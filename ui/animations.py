"""JavaScript animations embedded with st.iframe (self-contained HTML, our own templates only).

The simulation result is serialised into compact column arrays and injected into an HTML
template from ui/assets; all motion then runs client-side at 60 fps with requestAnimationFrame,
so playback stays smooth regardless of the server.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.simulator import SimulationResult

ASSETS = Path(__file__).resolve().parent / "assets"
MAX_POINTS = 3000          # longer trips are down-sampled to keep the page light


def _template(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def _round(series: pd.Series | np.ndarray, digits: int) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(series, dtype=float)]


def energy_flow_payload(regen: SimulationResult, friction: SimulationResult, aux_kw: float,
                        wh_per_km: float) -> dict:
    """Column arrays (kW, Wh, %) for the energy-flow animation, aligned on the regen run's time axis."""
    df, fdf = regen.df, friction.df
    stride = max(1, math.ceil(len(df) / MAX_POINTS))
    idx = np.arange(0, len(df), stride)
    reasons = sorted({str(r) for r in df["regen_reason"].unique()})
    rindex = {r: i for i, r in enumerate(reasons)}
    pick = lambda s: np.asarray(s)[idx]  # noqa: E731 - tiny local helper
    return {
        "t": _round(pick(df["t_s"]), 1),
        "v": _round(pick(df["speed_kmh"]), 2),
        "soc": _round(pick(df["soc_pct"]), 3),
        "socf": _round(pick(fdf["soc_pct"]), 3),
        "pb": _round(pick(df["p_batt_w"]) / 1000.0, 3),
        "pm": _round(pick(df["p_motor_elec_w"]) / 1000.0, 3),
        "prb": _round(pick(df["p_regen_batt_w"]) / 1000.0, 3),
        "pfr": _round(pick(df["p_friction_w"]) / 1000.0, 3),
        "prw": _round(pick(df["p_regen_wheel_w"]) / 1000.0, 3),
        "ptr": _round(pick(df["p_traction_w"]) / 1000.0, 3),
        "cur": _round(pick(df["current_a"]), 1),
        "vo": _round(pick(df["voltage_v"]), 1),
        "rpm": _round(pick(df["motor_speed_rpm"]), 0),
        "tq": _round(pick(df["motor_torque_nm"]), 1),
        "er": _round(pick(df["p_regen_batt_w"].cumsum()) / 3600.0, 1),
        "ef": _round(pick(df["p_friction_w"].cumsum()) / 3600.0, 1),
        "eff": _round(pick(fdf["p_friction_w"].cumsum()) / 3600.0, 1),
        "d": _round(pick(df["distance_km"]), 3),
        "rs": [rindex[str(r)] for r in pick(df["regen_reason"])],
        "reasons": reasons,
        "meta": {"soc0": float(regen.initial_soc_pct), "aux_kw": round(aux_kw, 3), "wh_per_km": round(wh_per_km, 2)},
    }


def energy_flow(regen: SimulationResult, friction: SimulationResult, aux_kw: float, wh_per_km: float,
                height: int = 760) -> None:
    payload = json.dumps(energy_flow_payload(regen, friction, aux_kw, wh_per_km), separators=(",", ":"))
    st.iframe(_template("energy_flow.html").replace("__DATA__", payload), height=height)


def regen_explainer(height: int = 430) -> None:
    st.iframe(_template("regen_explainer.html"), height=height)
