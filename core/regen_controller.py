"""Regenerative braking controller with friction-brake blending.

For every braking time step the controller decides how much braking power
comes from the motor (regen) and how much from the friction brakes.

    P_brake_request = −P_wheel                (only when P_wheel < 0)

    Regen available = min( motor generating limit,
                           battery charge limit,
                           strategy limit )          <- "the three regen limits"
    P_regen   = f_fade(v) · min(P_brake_request, Regen available)
    P_friction = P_brake_request − P_regen

Low-speed fade (motor back-EMF too low to regenerate usefully):
    f_fade = 0                         v ≤ v_fade_end
           = (v − v_end)/(v_start − v_end)  in between
           = 1                         v ≥ v_fade_start

Strategies
----------
* friction_only : baseline, P_regen = 0 always.
* series        : "regen-first" - motor takes as much as it can; friction
                  only tops up. Maximum recovery.
* parallel      : motor takes a fixed share (parallel_ratio) of every brake
                  request, friction takes the rest (simple hydraulic systems).
* levels        : like paddle-shifter regen levels in real EVs
                  (Off / Low / Medium / High / One-pedal). Each level caps the
                  regen deceleration: P_strategy = m · a_max · v.
In every strategy a calibration cap `max_regen_power_kw` also applies.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

STRATEGIES: tuple[str, ...] = ("friction_only", "series", "parallel", "levels")
REGEN_LEVELS: tuple[str, ...] = ("Off", "Low", "Medium", "High", "One-pedal")

# Reason codes shown in the braking-event table
REASON_NONE = "Full regen"
REASON_DISABLED = "Regen disabled"
REASON_MOTOR = "Motor limit"
REASON_FADE = "Low-speed fade"
REASON_STRATEGY = "Strategy limit"


@dataclass
class RegenParams:
    """Calibration parameters of the regen controller (these are what OTA changes)."""

    strategy: str = "series"
    parallel_ratio: float = 0.30
    regen_level: str = "High"
    fade_start_kmh: float = 10.0
    fade_end_kmh: float = 5.0
    max_regen_power_kw: float = 70.0
    version: str = "1.0.0"
    levels: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> "RegenParams":
        r = cfg["regen"]
        params = cls(
            strategy=r["strategy"],
            parallel_ratio=r["parallel_ratio"],
            regen_level=r["regen_level"],
            fade_start_kmh=r["fade_start_kmh"],
            fade_end_kmh=r["fade_end_kmh"],
            max_regen_power_kw=r["max_regen_power_kw"],
            version=r.get("version", "1.0.0"),
            levels=copy.deepcopy(r["levels"]),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(params, key, value)
        return params

    def tunables(self) -> dict[str, Any]:
        """The calibration values that an OTA package may change."""
        d = asdict(self)
        d.pop("levels")
        return d

    def copy(self) -> "RegenParams":
        return copy.deepcopy(self)


@dataclass(frozen=True)
class BrakeContext:
    """Everything the controller needs to know for one braking step (wheel-side powers, W)."""

    p_request_w: float          # braking power demanded at the wheels (>0)
    speed_mps: float
    mass_kg: float
    motor_limit_w: float        # motor generating limit referred to the wheel
    battery_limit_w: float      # battery charge acceptance referred to the wheel
    battery_reason: str         # "SoC limit" / "Temperature derate" / "Battery power limit"


@dataclass(frozen=True)
class BrakeSplit:
    p_regen_w: float            # wheel-side power taken by the motor
    p_friction_w: float         # wheel-side power turned into heat by the brake pads
    reason: str                 # why regen was below the request
    fade_factor: float
    strategy_limit_w: float


class RegenController:
    """Splits a braking request between motor (regen) and friction brakes."""

    def __init__(self, params: RegenParams) -> None:
        self.params = params

    # -------------------------------------------------------------------
    def fade_window_kmh(self) -> tuple[float, float]:
        """(fade_start, fade_end) - One-pedal level may override them."""
        p = self.params
        start, end = p.fade_start_kmh, p.fade_end_kmh
        if p.strategy == "levels":
            lvl = p.levels.get(p.regen_level, {})
            start = lvl.get("fade_start_kmh", start)
            end = lvl.get("fade_end_kmh", end)
        return start, end

    def fade_factor(self, speed_mps: float) -> float:
        start, end = self.fade_window_kmh()
        v_kmh = speed_mps * 3.6
        if v_kmh <= end:
            return 0.0
        if v_kmh >= start or start <= end:
            return 1.0
        return (v_kmh - end) / (start - end)

    def strategy_limit_w(self, ctx: BrakeContext) -> float:
        """Upper bound on regen power imposed by the selected strategy/calibration."""
        p = self.params
        cap = p.max_regen_power_kw * 1000.0
        if p.strategy == "friction_only":
            return 0.0
        if p.strategy == "series":
            return cap
        if p.strategy == "parallel":
            return min(cap, max(0.0, min(1.0, p.parallel_ratio)) * ctx.p_request_w)
        if p.strategy == "levels":
            a_max = p.levels.get(p.regen_level, {}).get("max_decel_mps2", 0.0)
            return min(cap, ctx.mass_kg * a_max * ctx.speed_mps)  # P = m·a_max·v
        raise ValueError(f"Unknown regen strategy '{p.strategy}'")

    # -------------------------------------------------------------------
    def split(self, ctx: BrakeContext) -> BrakeSplit:
        request = max(0.0, ctx.p_request_w)
        if request <= 0.0:
            return BrakeSplit(0.0, 0.0, REASON_NONE, 1.0, 0.0)
        if self.params.strategy == "friction_only":
            return BrakeSplit(0.0, request, REASON_DISABLED, 1.0, 0.0)

        strat = self.strategy_limit_w(ctx)
        limits = {
            REASON_MOTOR: max(0.0, ctx.motor_limit_w),
            ctx.battery_reason: max(0.0, ctx.battery_limit_w),
            REASON_STRATEGY: strat,
        }
        binding_reason, binding_value = min(limits.items(), key=lambda kv: kv[1])
        unfaded = min(request, binding_value)
        fade = self.fade_factor(ctx.speed_mps)
        regen = fade * unfaded

        if regen >= request * (1.0 - 1e-9):
            reason = REASON_NONE
        elif fade <= 0.0:
            reason = REASON_FADE
        elif unfaded < request:
            reason = binding_reason
        else:
            reason = REASON_FADE
        if reason == REASON_STRATEGY and self.params.strategy == "levels" and self.params.regen_level == "Off":
            reason = REASON_DISABLED
        return BrakeSplit(regen, request - regen, reason, fade, strat)
