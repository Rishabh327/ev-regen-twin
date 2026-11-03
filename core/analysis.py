"""Post-processing: energy totals, KPIs, braking-event table and sanity checks.

All energies are reported in Wh (watt-hours) unless the name says kWh.
    E = Σ P·Δt / 3600     (P in W, Δt in s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.simulator import SimulationResult


def _wh(series: pd.Series, dt_s: float = 1.0) -> float:
    return float(series.sum() * dt_s / 3600.0)


# --------------------------------------------------------------------------
# Energy totals
# --------------------------------------------------------------------------
def energy_totals(res: SimulationResult, dt_s: float = 1.0) -> dict[str, float]:
    """Integrate every power column into energy (Wh)."""
    df = res.df
    p_batt = df["p_batt_w"]
    motor_in = df["p_motor_elec_w"].clip(lower=0.0)
    e = {
        "traction_wheel_wh": _wh(df["p_traction_w"], dt_s),
        "unmet_traction_wh": _wh(df["p_unmet_w"], dt_s),
        "brake_request_wh": _wh(df["p_brake_req_w"], dt_s),
        "regen_wheel_wh": _wh(df["p_regen_wheel_w"], dt_s),
        "friction_wh": _wh(df["p_friction_w"], dt_s),
        "regen_batt_wh": _wh(df["p_regen_batt_w"], dt_s),
        "motor_in_wh": _wh(motor_in, dt_s),
        "aux_wh": _wh(df["p_aux_w"], dt_s),
        "batt_out_wh": _wh(p_batt.clip(lower=0.0), dt_s),
        "batt_in_wh": _wh((-p_batt).clip(lower=0.0), dt_s),
        "batt_net_wh": _wh(p_batt, dt_s),
        "batt_loss_wh": _wh(df["p_batt_loss_w"], dt_s),
        "batt_chem_net_wh": _wh(df["p_batt_chem_w"], dt_s),
        "aero_wh": _wh(df["p_aero_w"], dt_s),
        "rolling_wh": _wh(df["p_rolling_w"], dt_s),
        "grade_wh": _wh(df["p_grade_w"], dt_s),
        "kinetic_change_wh": _wh(df["p_inertia_w"], dt_s),
    }
    # motor+inverter+gear losses while driving, and while regenerating
    e["drive_conversion_loss_wh"] = e["motor_in_wh"] - (e["traction_wheel_wh"] - e["unmet_traction_wh"])
    e["regen_conversion_loss_wh"] = e["regen_wheel_wh"] - e["regen_batt_wh"]
    return e


# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
def compute_kpis(res: SimulationResult, dt_s: float = 1.0) -> dict[str, float]:
    e = energy_totals(res, dt_s)
    dist = float(res.df["distance_km"].iloc[-1]) if len(res.df) else 0.0
    wh_per_km = e["batt_net_wh"] / dist if dist > 0 else float("nan")
    usable_kwh = res.battery.capacity_kwh * res.battery.usable_soc_window
    range_km = usable_kwh * 1000.0 / wh_per_km if wh_per_km and wh_per_km > 0 else float("inf")
    return {
        "initial_soc_pct": res.initial_soc_pct,
        "final_soc_pct": res.final_soc_pct,
        "delta_soc_pct": res.initial_soc_pct - res.final_soc_pct,
        "distance_km": dist,
        "kwh_consumed_net": e["batt_net_wh"] / 1000.0,
        "kwh_drawn_gross": e["batt_out_wh"] / 1000.0,
        "kwh_recovered": e["regen_batt_wh"] / 1000.0,
        "kwh_brake_request": e["brake_request_wh"] / 1000.0,
        "kwh_friction": e["friction_wh"] / 1000.0,
        "kwh_batt_loss": e["batt_loss_wh"] / 1000.0,
        "recovery_eff_pct": 100.0 * e["regen_batt_wh"] / e["brake_request_wh"] if e["brake_request_wh"] > 0 else 0.0,
        "regen_share_pct": 100.0 * e["regen_batt_wh"] / e["motor_in_wh"] if e["motor_in_wh"] > 0 else 0.0,
        "wh_per_km": wh_per_km,
        "range_km": range_km,
    }


def compare_kpis(regen: dict[str, float], friction: dict[str, float]) -> dict[str, float]:
    """Regen-vs-friction comparison numbers for the KPI cards."""
    rg = regen["range_km"] / friction["range_km"] - 1.0 if friction["range_km"] > 0 else 0.0
    saved = friction["kwh_friction"] - regen["kwh_friction"]
    return {
        "range_gain_pct": 100.0 * rg,
        "soc_saved_pct": regen["final_soc_pct"] - friction["final_soc_pct"],
        "energy_saved_kwh": friction["kwh_consumed_net"] - regen["kwh_consumed_net"],
        "friction_saved_kwh": saved,
        "friction_saved_pct": 100.0 * saved / friction["kwh_friction"] if friction["kwh_friction"] > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# Braking events
# --------------------------------------------------------------------------
def braking_events(res: SimulationResult, dt_s: float = 1.0, min_request_wh: float = 0.5) -> pd.DataFrame:
    """One row per continuous braking event (consecutive steps with P_wheel < 0)."""
    df = res.df
    braking = (df["p_brake_req_w"] > 0).to_numpy()
    mass_eff = res.vehicle.mass_kg * res.vehicle.rotational_mass_factor
    speeds = df["speed_kmh"].to_numpy()
    rows: list[dict[str, Any]] = []
    k, n = 0, len(df)
    while k < n:
        if not braking[k]:
            k += 1
            continue
        start = k
        while k < n and braking[k]:
            k += 1
        seg = df.iloc[start:k]
        request_wh = _wh(seg["p_brake_req_w"], dt_s)
        if request_wh < min_request_wh:
            continue
        v_start = (speeds[start - 1] if start > 0 else 0.0) / 3.6
        v_end = speeds[k - 1] / 3.6
        ke_wh = max(0.0, 0.5 * mass_eff * (v_start ** 2 - v_end ** 2)) / 3600.0
        recovered_wh = _wh(seg["p_regen_batt_w"], dt_s)
        shortfall = (seg["p_brake_req_w"] - seg["p_regen_wheel_w"]).to_numpy()
        reasons = seg["regen_reason"].to_numpy()
        by_reason: dict[str, float] = {}
        for r, s in zip(reasons, shortfall):
            if r and r != "Full regen" and s > 0:
                by_reason[r] = by_reason.get(r, 0.0) + s
        total_short = sum(by_reason.values()) * dt_s / 3600.0
        if not by_reason or total_short < 0.02 * request_wh:
            limit = "None (full regen)"
        else:
            ordered = sorted(by_reason.items(), key=lambda kv: -kv[1])
            limit = " + ".join(r for r, s in ordered if s >= 0.25 * ordered[0][1])
        rows.append({
            "event": len(rows) + 1,
            "start_s": float(seg["t_s"].iloc[0] - dt_s),
            "end_s": float(seg["t_s"].iloc[-1]),
            "duration_s": float(len(seg) * dt_s),
            "v_start_kmh": v_start * 3.6,
            "v_end_kmh": v_end * 3.6,
            "kinetic_energy_wh": ke_wh,
            "brake_request_wh": request_wh,
            "recovered_wh": recovered_wh,
            "friction_wh": _wh(seg["p_friction_w"], dt_s),
            "recovery_pct": 100.0 * recovered_wh / ke_wh if ke_wh > 0 else 0.0,
            "regen_limited_by": limit,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sanity check
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SanityItem:
    metric: str
    value: float
    expected: tuple[float, float]
    status: str          # "OK", "WARN" (slightly outside), "FLAG" (far outside)
    note: str


def sanity_check(kpis_regen: dict[str, float], mean_speed_kmh: float, cfg: dict[str, Any],
                 strategy: str = "series") -> list[SanityItem]:
    """Compare results with typical published ranges and flag anything far outside."""
    s = cfg["sanity"]
    far = s["far_outside_factor"]
    if mean_speed_kmh < s["stop_go_mean_speed_kmh"]:
        kind, share_rng = "dense stop-and-go cycle", tuple(s["stop_go_regen_share_pct"])
    elif mean_speed_kmh < s["city_mean_speed_kmh"]:
        kind, share_rng = "city cycle", tuple(s["city_regen_share_pct"])
    else:
        kind, share_rng = "highway-heavy / mixed cycle", tuple(s["highway_regen_share_pct"])

    def judge(name: str, value: float, rng: tuple[float, float], note: str) -> SanityItem:
        lo, hi = rng
        if lo <= value <= hi:
            status = "OK"
        elif lo * (1 - far) <= value <= hi * (1 + far):
            status = "WARN"
        else:
            status = "FLAG"
        return SanityItem(name, value, (lo, hi), status, note)

    items = [
        judge("Regen recovered as % of traction energy", kpis_regen["regen_share_pct"], share_rng,
              f"Typical for a {kind} (mean speed {mean_speed_kmh:.1f} km/h)."),
        judge("Consumption (Wh/km, battery terminal)", kpis_regen["wh_per_km"], tuple(s["wh_per_km"]),
              "Compact EVs typically use ~100-180 Wh/km on test cycles."),
    ]
    if strategy == "series":
        items.append(judge("Braking-energy recovery efficiency (%)", kpis_regen["recovery_eff_pct"], (55.0, 90.0),
                           "Wheel braking energy reaching the battery terminals (series blending)."))
    return items


def format_sanity(items: list[SanityItem]) -> str:
    lines = ["SANITY CHECK (expected realistic ranges)"]
    for it in items:
        lines.append(f"  [{it.status:4s}] {it.metric:45s} = {it.value:7.1f}   expected {it.expected[0]:.0f}-"
                     f"{it.expected[1]:.0f}   ({it.note})")
    if any(it.status == "FLAG" for it in items):
        lines.append("  !! At least one result is FAR outside realistic ranges - check parameters.")
    return "\n".join(lines)


def soc_series_with_start(res: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    """Time and SoC arrays including the initial point at t = 0 (for plotting)."""
    t = np.concatenate([[0.0], res.df["t_s"].to_numpy()])
    soc = np.concatenate([[res.initial_soc_pct], res.df["soc_pct"].to_numpy()])
    return t, soc
