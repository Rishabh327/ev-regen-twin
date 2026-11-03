r"""Traction battery: 1-RC Thevenin equivalent circuit + Coulomb-counting SoC.

Circuit
-------
            R0          R1
   OCV ---/\/\/---+---/\/\/---+---- (+) terminal, voltage V
                  |           |
                  +----||-----+
                       C1
    V    = OCV(SoC) − I·R0 − V1          (I > 0 = discharge, I < 0 = charge)
    dV1/dt = −V1/(R1·C1) + I/C1          (RC branch = slow "polarisation")
    exact discrete update (zero-order hold):
    V1(k+1) = V1(k)·e^(−Δt/τ) + R1·I·(1 − e^(−Δt/τ)),  τ = R1·C1

Current from requested terminal power P (P = V·I):
    R0·I² − (OCV − V1)·I + P = 0
    I = [(OCV − V1) − sqrt((OCV − V1)² − 4·R0·P)] / (2·R0)

State of charge (Coulomb counting, Q = capacity in Ah)
    discharge: SoC(k+1) = SoC(k) − I·Δt / (3600·Q)
    charge   : SoC(k+1) = SoC(k) − η_charge·I·Δt / (3600·Q)   (I < 0, so SoC rises)

Losses: P_loss = I²·R0 + V1·I     (heat in R0 + energy into the RC branch)
Energy identity per step: OCV·I = V·I + P_loss  (checked in tests).

Charge acceptance (maximum regen power the battery will take)
    P_charge_max = P_charge_rated · f_SoC(SoC) · f_T(T)
    f_SoC tapers above 90 % and reaches ~0 at 98-100 %; f_T derates in cold/hot.
Temperature also raises resistance:  R0(T) = R0_25 · exp(k·(25 − T)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BatteryParams:
    chemistry: str
    capacity_kwh: float
    nominal_voltage_v: float
    capacity_ah: float
    n_series: int
    ocv_soc_pct: np.ndarray
    ocv_pack_v: np.ndarray
    r0_ohm: float
    r1_ohm: float
    c1_farad: float
    charge_efficiency: float
    max_discharge_power_w: float
    max_charge_power_w: float
    usable_soc_window: float
    r0_temp_coeff: float
    taper_soc_pct: np.ndarray
    taper_factor: np.ndarray
    derate_temp_c: np.ndarray
    derate_factor: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict[str, Any], r0_scale: float = 1.0) -> "BatteryParams":
        b = cfg["battery"]
        chem = b["chemistry"].upper()
        table = b["ocv_tables"][chem]
        n_series = int(round(b["nominal_voltage_v"] / table["cell_nominal_v"]))
        return cls(
            chemistry=chem,
            capacity_kwh=b["capacity_kwh"],
            nominal_voltage_v=b["nominal_voltage_v"],
            # Q (Ah) = E (Wh) / V_nominal
            capacity_ah=b["capacity_kwh"] * 1000.0 / b["nominal_voltage_v"],
            n_series=n_series,
            ocv_soc_pct=np.asarray(table["soc_pct"], dtype=float),
            ocv_pack_v=np.asarray(table["cell_ocv_v"], dtype=float) * n_series,
            r0_ohm=b["r0_ohm"] * r0_scale,
            r1_ohm=b["r1_ohm"],
            c1_farad=b["c1_farad"],
            charge_efficiency=b["charge_efficiency"],
            max_discharge_power_w=b["max_discharge_power_kw"] * 1000.0,
            max_charge_power_w=b["max_charge_power_kw"] * 1000.0,
            usable_soc_window=b["usable_soc_window"],
            r0_temp_coeff=b["r0_temp_coeff_per_c"],
            taper_soc_pct=np.asarray(b["soc_taper"]["soc_pct"], dtype=float),
            taper_factor=np.asarray(b["soc_taper"]["factor"], dtype=float),
            derate_temp_c=np.asarray(b["temp_derate"]["temp_c"], dtype=float),
            derate_factor=np.asarray(b["temp_derate"]["factor"], dtype=float),
        )


@dataclass(frozen=True)
class BatteryStep:
    current_a: float      # + discharge, − charge
    voltage_v: float      # terminal voltage
    ocv_v: float
    soc: float            # 0..1 after the step
    loss_w: float         # I²R0 + V1·I
    power_w: float        # terminal power actually delivered (after limits)
    chem_power_w: float   # OCV·I
    limited: bool         # True if the request had to be clipped


class BatteryModel:
    """Equivalent-circuit battery with state (SoC, V1)."""

    def __init__(self, params: BatteryParams, initial_soc: float, temperature_c: float = 25.0) -> None:
        self.p = params
        self.soc = float(min(1.0, max(0.0, initial_soc)))
        self.v1 = 0.0
        self.temperature_c = temperature_c
        self.depleted = False

    # ---------------- look-ups ----------------
    def ocv(self, soc: float | None = None) -> float:
        s = self.soc if soc is None else soc
        return float(np.interp(100.0 * s, self.p.ocv_soc_pct, self.p.ocv_pack_v))

    @property
    def r0(self) -> float:
        """R0(T) = R0_25·exp(k·(25 − T)), clipped to [0.8, 3]×R0_25."""
        factor = math.exp(self.p.r0_temp_coeff * (25.0 - self.temperature_c))
        return self.p.r0_ohm * min(3.0, max(0.8, factor))

    def soc_taper_factor(self, soc: float | None = None) -> float:
        s = self.soc if soc is None else soc
        return float(np.interp(100.0 * s, self.p.taper_soc_pct, self.p.taper_factor))

    def temp_derate_factor(self) -> float:
        return float(np.interp(self.temperature_c, self.p.derate_temp_c, self.p.derate_factor))

    def max_charge_power_w(self) -> float:
        """P_charge_max = P_rated · f_SoC · f_T  (terminal power the pack accepts)."""
        return self.p.max_charge_power_w * self.soc_taper_factor() * self.temp_derate_factor()

    def charge_limit_reason(self) -> str:
        """Which factor is limiting charge acceptance right now."""
        f_soc, f_t = self.soc_taper_factor(), self.temp_derate_factor()
        if f_soc < 1.0 and f_soc <= f_t:
            return "SoC limit"
        if f_t < 1.0:
            return "Temperature derate"
        return "Battery power limit"

    def max_discharge_power_w(self) -> float:
        return self.p.max_discharge_power_w * (0.0 if self.soc <= 0.0 else 1.0)

    # ---------------- dynamics ----------------
    def _solve_current(self, power_w: float, ocv: float) -> tuple[float, bool]:
        """Solve R0·I² − (OCV−V1)·I + P = 0 for the physically meaningful root."""
        e = ocv - self.v1
        r0 = self.r0
        disc = e * e - 4.0 * r0 * power_w
        if disc < 0.0:  # more power requested than the pack can ever deliver
            return e / (2.0 * r0), True
        return (e - math.sqrt(disc)) / (2.0 * r0), False

    def step(self, power_w: float, dt_s: float) -> BatteryStep:
        """Advance the battery by dt with a requested terminal power (W, + = discharge)."""
        limited = False
        if power_w > 0 and power_w > self.max_discharge_power_w():
            power_w, limited = self.max_discharge_power_w(), True
        if power_w < 0 and -power_w > self.p.max_charge_power_w:
            power_w, limited = -self.p.max_charge_power_w, True

        ocv = self.ocv()
        current, clipped = self._solve_current(power_w, ocv)
        limited = limited or clipped
        voltage = ocv - current * self.r0 - self.v1
        loss = current * current * self.r0 + self.v1 * current

        # RC branch (exact zero-order-hold update)
        tau = self.p.r1_ohm * self.p.c1_farad
        decay = math.exp(-dt_s / tau)
        self.v1 = self.v1 * decay + self.p.r1_ohm * current * (1.0 - decay)

        # Coulomb counting
        eta = 1.0 if current >= 0 else self.p.charge_efficiency
        self.soc -= eta * current * dt_s / (3600.0 * self.p.capacity_ah)
        if self.soc <= 0.0:
            self.soc, self.depleted = 0.0, True
        self.soc = min(1.0, self.soc)

        return BatteryStep(current, voltage, ocv, self.soc, loss, voltage * current, ocv * current, limited)
