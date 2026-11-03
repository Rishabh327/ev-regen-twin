"""Longitudinal vehicle dynamics: road-load forces and wheel power.

Equations (θ = road angle, v = speed in m/s, a = acceleration in m/s²)
----------------------------------------------------------------------
    F_rolling = m · g · Crr · cos(θ)        tyre rolling resistance
    F_aero    = 0.5 · ρ · Cd · A · v²       aerodynamic drag
    F_grade   = m · g · sin(θ)              gravity component on a slope
    F_inertia = m · δ · a                   force to accelerate the car
                                            (δ = rotational mass factor ≈ 1.05
                                             accounts for spinning wheels,
                                             gears and motor rotor)
    F_total   = F_rolling + F_aero + F_grade + F_inertia

    P_wheel   = F_total · v                 power at the wheels
        P_wheel > 0  -> the motor must push (traction)
        P_wheel < 0  -> the car must be slowed down (braking)

Discretisation
--------------
For the step from t_k to t_k+1 (Δt) we use the step-average speed
v = (v_k + v_k+1)/2 and a = (v_k+1 − v_k)/Δt. With this choice
F_inertia · v · Δt = ½·m·δ·(v_k+1² − v_k²) EXACTLY, i.e. kinetic energy is
conserved perfectly by the discretisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VehicleParams:
    """Vehicle body parameters (all ASSUMED values, see config JSON)."""

    mass_kg: float                  # total test mass = curb + driver + payload
    rotational_mass_factor: float   # δ (-)
    drag_coefficient: float         # Cd (-)
    frontal_area_m2: float          # A (m²)
    rolling_resistance_coeff: float  # Crr (-)
    wheel_radius_m: float           # r (m)
    gear_ratio: float               # G (-), motor rpm / wheel rpm
    drivetrain_efficiency: float    # η_drivetrain (-), gearbox + shafts
    aux_power_w: float              # HVAC, 12 V loads, electronics (W)
    air_density_kg_m3: float = 1.2  # ρ
    gravity_m_s2: float = 9.81      # g

    @classmethod
    def from_config(cls, cfg: dict[str, Any], payload_kg: float | None = None,
                    mass_scale: float = 1.0) -> "VehicleParams":
        v = cfg["vehicle"]
        payload = v.get("payload_kg", 0.0) if payload_kg is None else payload_kg
        mass = (v["curb_mass_kg"] + v["driver_mass_kg"] + payload) * mass_scale
        return cls(
            mass_kg=mass,
            rotational_mass_factor=v["rotational_mass_factor"],
            drag_coefficient=v["drag_coefficient"],
            frontal_area_m2=v["frontal_area_m2"],
            rolling_resistance_coeff=v["rolling_resistance_coeff"],
            wheel_radius_m=v["wheel_radius_m"],
            gear_ratio=v["gear_ratio"],
            drivetrain_efficiency=v["drivetrain_efficiency"],
            aux_power_w=v["aux_power_w"],
            air_density_kg_m3=v.get("air_density_kg_m3", 1.2),
            gravity_m_s2=v.get("gravity_m_s2", 9.81),
        )


@dataclass(frozen=True)
class RoadLoad:
    """Forces (N) and powers (W) for one time step."""

    f_rolling: float
    f_aero: float
    f_grade: float
    f_inertia: float
    f_total: float
    p_wheel: float      # F_total · v (W)
    speed_mps: float    # step-average speed used
    accel_mps2: float


class VehicleDynamics:
    """Computes road-load forces for a given speed, acceleration and grade."""

    def __init__(self, params: VehicleParams) -> None:
        self.p = params

    @staticmethod
    def grade_angle_rad(grade_pct: float) -> float:
        """Road grade in % (rise/run × 100) -> angle θ in radians."""
        return math.atan(grade_pct / 100.0)

    def road_load(self, speed_mps: float, accel_mps2: float, grade_pct: float = 0.0) -> RoadLoad:
        p = self.p
        theta = self.grade_angle_rad(grade_pct)
        moving = speed_mps > 1e-3
        # F_rolling = m·g·Crr·cos(θ)   (only while rolling; a parked car has no rolling loss)
        f_roll = p.mass_kg * p.gravity_m_s2 * p.rolling_resistance_coeff * math.cos(theta) if moving else 0.0
        # F_aero = 0.5·ρ·Cd·A·v²
        f_aero = 0.5 * p.air_density_kg_m3 * p.drag_coefficient * p.frontal_area_m2 * speed_mps ** 2
        # F_grade = m·g·sin(θ)   (positive uphill, negative downhill)
        f_grade = p.mass_kg * p.gravity_m_s2 * math.sin(theta) if moving else 0.0
        # F_inertia = m·δ·a
        f_inertia = p.mass_kg * p.rotational_mass_factor * accel_mps2
        f_total = f_roll + f_aero + f_grade + f_inertia
        # P_wheel = F_total·v
        p_wheel = f_total * speed_mps
        return RoadLoad(f_roll, f_aero, f_grade, f_inertia, f_total, p_wheel, speed_mps, accel_mps2)

    def step(self, v_start_mps: float, v_end_mps: float, dt_s: float, grade_pct: float = 0.0) -> RoadLoad:
        """Road load for one discrete step (average speed, constant acceleration)."""
        v_avg = 0.5 * (v_start_mps + v_end_mps)
        accel = (v_end_mps - v_start_mps) / dt_s
        return self.road_load(v_avg, accel, grade_pct)

    def wheel_to_motor(self, speed_mps: float, wheel_force_n: float) -> tuple[float, float]:
        """Convert wheel speed/force to motor shaft speed (rad/s) and torque (N·m), ignoring losses.

        ω_motor = (v / r) · G          T_motor = F · r / G
        """
        omega = speed_mps / self.p.wheel_radius_m * self.p.gear_ratio
        torque = wheel_force_n * self.p.wheel_radius_m / self.p.gear_ratio
        return omega, torque

    def kinetic_energy_j(self, speed_mps: float) -> float:
        """Kinetic energy including rotating parts: ½·m·δ·v²."""
        return 0.5 * self.p.mass_kg * self.p.rotational_mass_factor * speed_mps ** 2
