"""Simulated over-the-air (OTA) update of the regen calibration, mid-drive.

Life-cycle of a package (mirrors real SDV OTA flows such as Uptane / UNECE R156):

    SCHEDULED -> (time reached) -> VALIDATING
        - integrity: SHA-256 checksum of the payload must match
        - version : must be newer than the running version (no silent downgrade)
        - safety  : every parameter must be inside the allowed range
    -> INSTALLED  (old calibration kept as rollback image)   or   REJECTED
    -> health monitor for `health_window_s`; if it fails -> ROLLED_BACK
    -> manual rollback can also be scheduled at any time.

Only *calibration* parameters of the regen controller are touched - exactly
the kind of change an OEM pushes OTA without a workshop visit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from core.regen_controller import REGEN_LEVELS, RegenParams

OTA_TUNABLE_KEYS: tuple[str, ...] = (
    "strategy", "parallel_ratio", "regen_level", "fade_start_kmh", "fade_end_kmh", "max_regen_power_kw",
)


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def payload_checksum(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class OtaPackage:
    version: str
    params: dict[str, Any]
    description: str = ""
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = payload_checksum(self.params)


@dataclass(frozen=True)
class OtaEvent:
    t_s: float
    kind: str           # SCHEDULED / INSTALLED / REJECTED / ROLLED_BACK / HEALTH_OK
    version: str
    detail: str


def validate_package(pkg: OtaPackage, current: RegenParams, limits: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors (empty list = package is valid)."""
    errors: list[str] = []
    if payload_checksum(pkg.params) != pkg.checksum:
        errors.append("Checksum mismatch - payload corrupted or tampered.")
    try:
        if _version_tuple(pkg.version) <= _version_tuple(current.version):
            errors.append(f"Version {pkg.version} is not newer than running {current.version}.")
    except ValueError:
        errors.append(f"Malformed version string '{pkg.version}'.")

    for key, value in pkg.params.items():
        if key not in OTA_TUNABLE_KEYS:
            errors.append(f"Parameter '{key}' is not OTA-tunable.")
            continue
        if key == "strategy":
            if value not in limits["allowed_strategies"]:
                errors.append(f"strategy '{value}' not allowed.")
        elif key == "regen_level":
            if value not in REGEN_LEVELS:
                errors.append(f"regen_level '{value}' unknown.")
        else:
            lo, hi = limits[key]
            if not isinstance(value, (int, float)) or not (lo <= float(value) <= hi):
                errors.append(f"{key} = {value} outside safe range [{lo}, {hi}].")

    # cross-parameter checks on the resulting calibration
    merged = {**current.tunables(), **pkg.params}
    if merged["fade_end_kmh"] >= merged["fade_start_kmh"]:
        errors.append("fade_end_kmh must be lower than fade_start_kmh.")
    if merged["strategy"] == "levels":
        decel = current.levels.get(merged["regen_level"], {}).get("max_decel_mps2", 0.0)
        if decel > limits["max_safe_regen_decel_mps2"]:
            errors.append(f"Level {merged['regen_level']} decel {decel} m/s² exceeds safe limit.")
    return errors


def param_diff(old: RegenParams, new: RegenParams) -> dict[str, tuple[Any, Any]]:
    a, b = old.tunables(), new.tunables()
    return {k: (a[k], b[k]) for k in a if a[k] != b[k]}


@dataclass
class OtaManager:
    """Simulator hook that installs scheduled OTA packages and handles rollback."""

    limits: dict[str, Any]
    health_window_s: float = 30.0
    max_charge_current_a: float = float("inf")
    schedule: list[tuple[float, OtaPackage]] = field(default_factory=list)
    rollback_at: list[float] = field(default_factory=list)
    events: list[OtaEvent] = field(default_factory=list)
    diffs: list[dict[str, Any]] = field(default_factory=list)
    _rollback_image: RegenParams | None = None
    _installed_at: float | None = None

    def schedule_update(self, at_t_s: float, pkg: OtaPackage) -> None:
        self.schedule.append((at_t_s, pkg))
        self.schedule.sort(key=lambda x: x[0])
        self.events.append(OtaEvent(0.0, "SCHEDULED", pkg.version, f"install at t = {at_t_s:.0f} s: {pkg.description}"))

    def schedule_rollback(self, at_t_s: float) -> None:
        self.rollback_at.append(at_t_s)

    # ------------------------------------------------------------------
    def install(self, controller: Any, pkg: OtaPackage, t_s: float) -> bool:
        current: RegenParams = controller.params
        errors = validate_package(pkg, current, self.limits)
        if errors:
            self.events.append(OtaEvent(float(t_s), "REJECTED", pkg.version, "; ".join(errors)))
            return False
        new = current.copy()
        for k, v in pkg.params.items():
            setattr(new, k, v)
        new.version = pkg.version
        diff = param_diff(current, new)
        self._rollback_image = current.copy()
        controller.params = new
        self._installed_at = t_s
        self.diffs.append({"t_s": t_s, "from": current.version, "to": new.version, "diff": diff})
        self.events.append(OtaEvent(float(t_s), "INSTALLED", new.version,
                                    ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in diff.items()) or "no change"))
        return True

    def rollback(self, controller: Any, t_s: float, reason: str) -> bool:
        if self._rollback_image is None:
            return False
        bad = controller.params.version
        controller.params = self._rollback_image
        self._rollback_image = None
        self._installed_at = None
        self.events.append(OtaEvent(float(t_s), "ROLLED_BACK", controller.params.version, f"from {bad}: {reason}"))
        return True

    def __call__(self, sim: Any, row: dict[str, Any]) -> None:
        t = row["t_s"]
        while self.schedule and self.schedule[0][0] <= t:
            _, pkg = self.schedule.pop(0)
            self.install(sim.controller, pkg, t)
        if self.rollback_at and self.rollback_at[0] <= t:
            self.rollback_at.pop(0)
            self.rollback(sim.controller, t, "manual rollback requested")
        # post-install health monitor: charge current must stay within the pack rating
        if self._installed_at is not None:
            if -row["current_a"] > self.max_charge_current_a:
                self.rollback(sim.controller, t, f"health check failed: charge current {-row['current_a']:.0f} A")
            elif t - self._installed_at >= self.health_window_s:
                self.events.append(OtaEvent(float(t), "HEALTH_OK", sim.controller.params.version,
                                            f"{self.health_window_s:.0f} s monitor passed"))
                self._installed_at = None
