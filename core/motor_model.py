"""Traction motor + inverter model: torque/power limits and an efficiency map.

Limits (same for motoring and generating, as requested)
-------------------------------------------------------
    T_max(ω) = min(T_peak, P_peak / ω)     constant torque, then constant power
    P_max(ω) = min(P_peak, T_peak · ω)
    Above max_speed_rpm the motor can deliver nothing.

Efficiency map (physics-based loss model, motor + inverter combined)
--------------------------------------------------------------------
    P_loss = k_c·T²  +  k_i·ω  +  k_w·ω³  +  P_0
             copper    iron      windage    inverter / constant losses
    Motoring  : η = P_mech / (P_mech + P_loss)
    Generating: η = (P_mech − P_loss) / P_mech
    η is clipped to [min_efficiency, max_efficiency] to stay well-behaved.

Battery-side power (η_d = drivetrain efficiency, η_m = motor+inverter)
---------------------------------------------------------------------
    Motoring  : P_batt = P_wheel / (η_d · η_m)
    Generating: P_batt = P_wheel · η_d · η_m     (flows back into battery)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotorParams:
    peak_power_w: float
    peak_torque_nm: float
    max_speed_rad_s: float
    k_copper: float
    k_iron: float
    k_windage: float
    p_constant_w: float
    min_efficiency: float
    max_efficiency: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MotorParams":
        m = cfg["motor"]
        return cls(
            peak_power_w=m["peak_power_kw"] * 1000.0,
            peak_torque_nm=m["peak_torque_nm"],
            max_speed_rad_s=m["max_speed_rpm"] * 2.0 * math.pi / 60.0,
            k_copper=m["loss_copper_coeff_w_per_nm2"],
            k_iron=m["loss_iron_coeff_w_per_rad_s"],
            k_windage=m["loss_windage_coeff_w_per_rad3_s3"],
            p_constant_w=m["loss_constant_w"],
            min_efficiency=m["min_efficiency"],
            max_efficiency=m["max_efficiency"],
        )


class MotorModel:
    """Permanent-magnet synchronous motor + inverter (lumped)."""

    def __init__(self, params: MotorParams) -> None:
        self.p = params

    # ---------------- limits ----------------
    def max_torque_nm(self, omega_rad_s: float) -> float:
        """T_max(ω) = min(T_peak, P_peak/ω); zero above max speed."""
        omega = abs(omega_rad_s)
        if omega > self.p.max_speed_rad_s:
            return 0.0
        if omega < 1e-6:
            return self.p.peak_torque_nm
        return min(self.p.peak_torque_nm, self.p.peak_power_w / omega)

    def max_power_w(self, omega_rad_s: float) -> float:
        """P_max(ω) = min(P_peak, T_peak·ω) at the motor shaft (mechanical)."""
        omega = abs(omega_rad_s)
        if omega > self.p.max_speed_rad_s:
            return 0.0
        return min(self.p.peak_power_w, self.p.peak_torque_nm * omega)

    # ---------------- efficiency ----------------
    def loss_w(self, torque_nm: float, omega_rad_s: float) -> float:
        """P_loss = k_c·T² + k_i·ω + k_w·ω³ + P_0."""
        t, w = abs(torque_nm), abs(omega_rad_s)
        return self.p.k_copper * t * t + self.p.k_iron * w + self.p.k_windage * w ** 3 + self.p.p_constant_w

    def efficiency(self, torque_nm: float, omega_rad_s: float, generating: bool = False) -> float:
        """Combined motor + inverter efficiency at an operating point."""
        p_mech = abs(torque_nm * omega_rad_s)
        if p_mech < 1.0:
            return self.p.min_efficiency
        loss = self.loss_w(torque_nm, omega_rad_s)
        eta = (p_mech - loss) / p_mech if generating else p_mech / (p_mech + loss)
        return float(min(self.p.max_efficiency, max(self.p.min_efficiency, eta)))

    def efficiency_map(self, n_speed: int = 60, n_torque: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grid of motoring efficiency for plotting: (speed_rpm, torque_nm, eta[torque, speed]).

        Points outside the torque envelope are NaN.
        """
        omegas = np.linspace(1.0, self.p.max_speed_rad_s, n_speed)
        torques = np.linspace(1.0, self.p.peak_torque_nm, n_torque)
        eta = np.full((n_torque, n_speed), np.nan)
        for j, w in enumerate(omegas):
            t_lim = self.max_torque_nm(w)
            for i, t in enumerate(torques):
                if t <= t_lim:
                    eta[i, j] = self.efficiency(t, w)
        return omegas * 60.0 / (2.0 * math.pi), torques, eta
