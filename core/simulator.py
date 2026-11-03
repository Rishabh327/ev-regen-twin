"""Time-step simulator: ties drive cycle, vehicle, motor, battery and regen controller together.

One step (Δt = 1 s) does:
    1. speed trace     -> v, a                  (drive cycle)
    2. road load       -> F_total, P_wheel      (vehicle_dynamics)
    3a. P_wheel ≥ 0    -> traction: P_elec = P_wheel / (η_d·η_m)
    3b. P_wheel < 0    -> braking: controller splits request into regen + friction
                          P_elec = −P_regen · η_d · η_m
    4. P_battery = P_elec + P_aux               (auxiliary loads always on)
    5. battery.step(P_battery)                  (current, voltage, SoC, losses)
    6. hooks (OTA manager, twin correction, VSS signal publisher) see the result

Sign conventions inside the model: battery current/power > 0 = discharge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from core.battery_model import BatteryModel, BatteryParams
from core.motor_model import MotorModel, MotorParams
from core.regen_controller import BrakeContext, RegenController, RegenParams
from core.vehicle_dynamics import VehicleDynamics, VehicleParams


class StepHook(Protocol):
    """Anything called after every step: hook(simulator, row). It may modify sim state or the row."""

    def __call__(self, sim: "Simulator", row: dict[str, Any]) -> None: ...


@dataclass
class SimulationResult:
    label: str
    df: pd.DataFrame
    initial_soc_pct: float
    vehicle: VehicleParams
    battery: BatteryParams
    final_params: RegenParams
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def final_soc_pct(self) -> float:
        return float(self.df["soc_pct"].iloc[-1]) if len(self.df) else self.initial_soc_pct


class Simulator:
    """EV energy model for one vehicle configuration and one regen calibration."""

    def __init__(
        self,
        cfg: dict[str, Any],
        regen_params: RegenParams | None = None,
        initial_soc_pct: float | None = None,
        ambient_temp_c: float | None = None,
        road_grade_pct: float | None = None,
        payload_kg: float | None = None,
        mass_scale: float = 1.0,
        r0_scale: float = 1.0,
        label: str = "",
    ) -> None:
        sim_cfg = cfg["simulation"]
        self.cfg = cfg
        self.label = label
        self.dt = float(sim_cfg["dt_s"])
        self.initial_soc_pct = float(sim_cfg["initial_soc_pct"] if initial_soc_pct is None else initial_soc_pct)
        self.ambient_temp_c = float(sim_cfg["ambient_temp_c"] if ambient_temp_c is None else ambient_temp_c)
        self.road_grade_pct = float(sim_cfg["road_grade_pct"] if road_grade_pct is None else road_grade_pct)

        self.vehicle_params = VehicleParams.from_config(cfg, payload_kg=payload_kg, mass_scale=mass_scale)
        self.battery_params = BatteryParams.from_config(cfg, r0_scale=r0_scale)
        self.dynamics = VehicleDynamics(self.vehicle_params)
        self.motor = MotorModel(MotorParams.from_config(cfg))
        self.controller = RegenController(regen_params.copy() if regen_params else RegenParams.from_config(cfg))
        self.battery = BatteryModel(self.battery_params, self.initial_soc_pct / 100.0, self.ambient_temp_c)

    # ------------------------------------------------------------------
    def step(self, t_end: float, v0_kmh: float, v1_kmh: float, distance_km: float) -> dict[str, Any]:
        """Simulate one step from v0 to v1 and return a row of results."""
        vp = self.vehicle_params
        eta_d = vp.drivetrain_efficiency
        rl = self.dynamics.step(v0_kmh / 3.6, v1_kmh / 3.6, self.dt, self.road_grade_pct)
        v = rl.speed_mps
        omega, torque_wheel_equiv = self.dynamics.wheel_to_motor(v, rl.f_total)

        p_traction = p_brake_req = p_regen = p_friction = 0.0
        p_elec = 0.0                  # electrical power into motor (+) / out of motor (−)
        p_regen_elec = 0.0            # electrical power returned by regen (+)
        unmet_w = 0.0
        eta_m = 0.0
        torque_motor = 0.0
        reason = ""
        fade = 1.0

        if rl.p_wheel >= 0.0:
            # ---------------- traction ----------------
            p_traction = rl.p_wheel
            p_shaft = p_traction / eta_d                     # P_shaft = P_wheel / η_d
            p_max = self.motor.max_power_w(omega)
            if p_shaft > p_max:                             # car cannot follow the trace
                unmet_w = (p_shaft - p_max) * eta_d
                p_shaft = p_max
            if p_shaft > 0.0 and omega > 1e-6:
                torque_motor = p_shaft / omega
                eta_m = self.motor.efficiency(torque_motor, omega)
                p_elec = p_shaft / eta_m                     # P_batt = P_wheel / (η_d·η_m)
        else:
            # ---------------- braking ----------------
            p_brake_req = -rl.p_wheel
            req_torque_shaft = abs(torque_wheel_equiv) * eta_d
            eta_est = self.motor.efficiency(req_torque_shaft, omega, generating=True)
            batt_limit_term = self.battery.max_charge_power_w()
            ctx = BrakeContext(
                p_request_w=p_brake_req,
                speed_mps=v,
                mass_kg=vp.mass_kg,
                motor_limit_w=self.motor.max_power_w(omega) / eta_d,
                battery_limit_w=batt_limit_term / (eta_d * eta_est),
                battery_reason=self.battery.charge_limit_reason(),
            )
            split = self.controller.split(ctx)
            p_regen, reason, fade = split.p_regen_w, split.reason, split.fade_factor
            if p_regen > 0.0 and omega > 1e-6:
                p_shaft = p_regen * eta_d
                torque_motor = -p_shaft / omega
                eta_m = self.motor.efficiency(p_shaft / omega, omega, generating=True)
                p_regen_elec = p_shaft * eta_m               # P_batt = P_wheel · η_d · η_m
                if p_regen_elec > batt_limit_term:           # respect the battery limit exactly
                    p_regen_elec = batt_limit_term
                    p_regen = p_regen_elec / (eta_d * eta_m)
                    reason = ctx.battery_reason
            else:
                p_regen = 0.0
            p_friction = p_brake_req - p_regen
            p_elec = -p_regen_elec

        p_batt_request = p_elec + vp.aux_power_w
        bstep = self.battery.step(p_batt_request, self.dt)
        distance_km += v * self.dt / 1000.0

        return {
            "t_s": t_end,
            "speed_kmh": v1_kmh,
            "avg_speed_kmh": v * 3.6,
            "accel_mps2": rl.accel_mps2,
            "distance_km": distance_km,
            "f_rolling_n": rl.f_rolling,
            "f_aero_n": rl.f_aero,
            "f_grade_n": rl.f_grade,
            "f_inertia_n": rl.f_inertia,
            "f_total_n": rl.f_total,
            "p_wheel_w": rl.p_wheel,
            "p_rolling_w": rl.f_rolling * v,
            "p_aero_w": rl.f_aero * v,
            "p_grade_w": rl.f_grade * v,
            "p_inertia_w": rl.f_inertia * v,
            "p_traction_w": p_traction,
            "p_unmet_w": unmet_w,
            "p_brake_req_w": p_brake_req,
            "p_regen_wheel_w": p_regen,
            "p_friction_w": p_friction,
            "p_regen_batt_w": p_regen_elec,
            "p_motor_elec_w": p_elec,
            "p_aux_w": vp.aux_power_w,
            "p_batt_w": bstep.power_w,
            "p_batt_chem_w": bstep.chem_power_w,
            "p_batt_loss_w": bstep.loss_w,
            "current_a": bstep.current_a,
            "voltage_v": bstep.voltage_v,
            "ocv_v": bstep.ocv_v,
            "soc_pct": 100.0 * bstep.soc,
            "motor_speed_rpm": omega * 60.0 / (2.0 * 3.141592653589793),
            "motor_torque_nm": torque_motor,
            "motor_eff": eta_m,
            "fade_factor": fade,
            "regen_reason": reason,
            "charge_limit_w": self.battery.max_charge_power_w(),
            "regen_version": self.controller.params.version,
            "regen_strategy": self.controller.params.strategy,
        }

    # ------------------------------------------------------------------
    def run(self, cycle: pd.DataFrame, hooks: Iterable[StepHook | Callable[..., None]] = ()) -> SimulationResult:
        """Run a whole drive cycle. Hooks are called after every step: hook(sim, row)."""
        hooks = list(hooks)
        speeds = cycle["speed_kmh"].to_numpy(dtype=float)
        times = cycle["time_s"].to_numpy(dtype=float)
        rows: list[dict[str, Any]] = []
        distance = 0.0
        for k in range(len(speeds) - 1):
            row = self.step(times[k + 1], speeds[k], speeds[k + 1], distance)
            distance = row["distance_km"]
            for hook in hooks:
                hook(self, row)
            rows.append(row)
        df = pd.DataFrame(rows)
        flags = {
            "battery_depleted": self.battery.depleted,
            "trace_miss_steps": int((df["p_unmet_w"] > 1.0).sum()) if len(df) else 0,
        }
        return SimulationResult(
            label=self.label,
            df=df,
            initial_soc_pct=self.initial_soc_pct,
            vehicle=self.vehicle_params,
            battery=self.battery_params,
            final_params=self.controller.params.copy(),
            flags=flags,
        )


def run_case(
    cfg: dict[str, Any],
    cycle: pd.DataFrame,
    strategy: str,
    label: str | None = None,
    hooks: Iterable[StepHook | Callable[..., None]] = (),
    regen_overrides: dict[str, Any] | None = None,
    **sim_kwargs: Any,
) -> SimulationResult:
    """Convenience wrapper: build params for a strategy and run the cycle."""
    overrides = dict(regen_overrides or {})
    overrides["strategy"] = strategy
    params = RegenParams.from_config(cfg, **overrides)
    sim = Simulator(cfg, params, label=label or strategy, **sim_kwargs)
    return sim.run(cycle, hooks)
