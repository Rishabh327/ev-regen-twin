"""EV Energy-Recovery Digital Twin - Streamlit dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from core import PROJECT_ROOT, load_config  # noqa: E402
from core.analysis import (  # noqa: E402
    braking_events, compare_kpis, compute_kpis, energy_totals, sanity_check, soc_series_with_start,
)
from core.drive_cycles import BUILTIN_CYCLES, cycle_stats, load_cycle, parse_uploaded_csv, repeat_cycle  # noqa: E402
from core.motor_model import MotorModel, MotorParams  # noqa: E402
from core.regen_controller import REGEN_LEVELS, RegenParams  # noqa: E402
from core.simulator import SimulationResult, Simulator  # noqa: E402
from sdv.ota_update import OtaManager, OtaPackage  # noqa: E402
from sdv.signals import VSS_CATALOG, SignalBus, SignalPublisher, catalog_frame  # noqa: E402
from sdv.twin_sync import TwinSyncResult, run_twin_sync  # noqa: E402

st.set_page_config(page_title="EV Regen Digital Twin", page_icon="🔋", layout="wide")

C_REGEN = "#1a9850"      # green  = regenerative braking case
C_FRIC = "#d73027"       # red    = friction-only baseline
C_OTA = "#7b3294"        # purple = OTA case
C_SPEED = "#4575b4"
C_BRAKE_SHADE = "rgba(253, 174, 97, 0.25)"
STRATEGY_LABELS = {
    "series": "Series / regen-first (max recovery)",
    "parallel": "Parallel (fixed regen share)",
    "levels": "Regen levels (Off/Low/Med/High/One-pedal)",
}


# =============================================================================
# Simulation (cached)
# =============================================================================
@st.cache_data(show_spinner="Running digital twin…")
def simulate_everything(cycle: pd.DataFrame, sim_inputs: dict[str, Any], regen_inputs: dict[str, Any],
                        ota_inputs: dict[str, Any], twin_inputs: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    common = dict(initial_soc_pct=sim_inputs["initial_soc"], ambient_temp_c=sim_inputs["temp"],
                  road_grade_pct=sim_inputs["grade"], payload_kg=sim_inputs["payload"])
    regen_params = RegenParams.from_config(cfg, **regen_inputs)
    fric_params = RegenParams.from_config(cfg, strategy="friction_only")

    # Baseline and regen case. The regen case also publishes VSS signals to the bus.
    friction = Simulator(cfg, fric_params, label="Friction only", **common).run(cycle)
    bus = SignalBus(record_history=True)
    consumer_log: dict[str, int] = {"brake_msgs": 0, "battery_msgs": 0}
    bus.subscribe("Vehicle.Chassis.Brake.*", lambda s: consumer_log.__setitem__("brake_msgs", consumer_log["brake_msgs"] + 1))
    bus.subscribe("Vehicle.Powertrain.TractionBattery.*",
                  lambda s: consumer_log.__setitem__("battery_msgs", consumer_log["battery_msgs"] + 1))
    regen = Simulator(cfg, regen_params, label="Regenerative braking", **common).run(cycle, hooks=[SignalPublisher(bus)])

    out: dict[str, Any] = {
        "friction": friction, "regen": regen,
        "bus_history": bus.history_frame(),
        "bus_stats": {"messages": bus.message_count, "subscribers": bus.subscriber_count, **consumer_log},
        "ota": None, "twin": None,
    }

    # OTA case: same regen run, but an OTA package arrives mid-drive
    if ota_inputs["enabled"]:
        params_new = dict(ota_inputs["params"])
        if ota_inputs["inject_invalid"]:
            params_new.update({"max_regen_power_kw": 250.0, "parallel_ratio": 1.5})
        manager = OtaManager(cfg["ota_limits"], max_charge_current_a=cfg["battery"]["max_charge_power_kw"] * 1000.0
                             / cfg["battery"]["nominal_voltage_v"] * 1.2)
        manager.schedule_update(ota_inputs["t"], OtaPackage(ota_inputs["version"], params_new,
                                                            "regen calibration update"))
        if ota_inputs["rollback"]:
            manager.schedule_rollback(ota_inputs["rollback_t"])
        ota_res = Simulator(cfg, regen_params, label="Regen + OTA", **common).run(cycle, hooks=[manager])
        out["ota"] = {"result": ota_res, "events": [e.__dict__ for e in manager.events], "diffs": manager.diffs}

    if twin_inputs["enabled"]:
        out["twin"] = run_twin_sync(cfg, cycle, regen_params, sim_inputs["initial_soc"], sim_inputs["temp"],
                                    sim_inputs["grade"], sim_inputs["payload"], method=twin_inputs["method"])
    return out


# =============================================================================
# Sidebar
# =============================================================================
def sidebar(cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, dict, dict, dict, dict]:
    sb = st.sidebar
    sb.title("🔋 EV Regen Twin")
    sb.caption("Software Defined Vehicles - course project")

    sb.header("1 · Drive cycle")
    options = {info.label: key for key, info in BUILTIN_CYCLES.items()}
    options["Upload my own CSV…"] = "UPLOAD"
    choice = sb.selectbox("Drive cycle", list(options), index=0)
    key = options[choice]
    if key == "UPLOAD":
        up = sb.file_uploader("CSV with columns time_s, speed_kmh", type=["csv"])
        if up is None:
            sb.info("No file yet - using WLTP until you upload one.")
            base, name = load_cycle("WLTP"), "WLTP Class 3b"
        else:
            try:
                base, name = parse_uploaded_csv(up.getvalue()), up.name
            except Exception as exc:  # noqa: BLE001 - show any parse error to the user
                sb.error(f"Could not read CSV: {exc}")
                base, name = load_cycle("WLTP"), "WLTP Class 3b"
    else:
        base, name = load_cycle(key), choice
        if not BUILTIN_CYCLES[key].exact:
            sb.caption(f"ⓘ {BUILTIN_CYCLES[key].note}")
    repeats = sb.slider("Number of cycle repeats", 1, 5, 1)
    cycle = repeat_cycle(base, repeats)
    duration = int(cycle["time_s"].iloc[-1])

    sb.header("2 · Vehicle & environment")
    sim_inputs = {
        "initial_soc": sb.slider("Initial SoC (%)", 10, 100, int(cfg["simulation"]["initial_soc_pct"])),
        "temp": sb.slider("Ambient temperature (°C)", -20, 50, int(cfg["simulation"]["ambient_temp_c"])),
        "payload": sb.slider("Payload (kg, on top of driver)", 0, 400, 0, step=25),
        "grade": sb.slider("Road grade (%)", -6.0, 6.0, 0.0, step=0.5),
    }

    sb.header("3 · Regen strategy")
    strategy = sb.selectbox("Strategy", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get)
    regen_inputs: dict[str, Any] = {"strategy": strategy}
    if strategy == "parallel":
        regen_inputs["parallel_ratio"] = sb.slider("Regen share of braking", 0.0, 1.0,
                                                   float(cfg["regen"]["parallel_ratio"]), 0.05)
    if strategy == "levels":
        regen_inputs["regen_level"] = sb.select_slider("Regen level", list(REGEN_LEVELS),
                                                       value=cfg["regen"]["regen_level"])
    with sb.expander("Advanced calibration"):
        regen_inputs["fade_start_kmh"] = st.slider("Fade start (km/h)", 2.0, 25.0, float(cfg["regen"]["fade_start_kmh"]))
        regen_inputs["fade_end_kmh"] = st.slider("Fade end - regen = 0 (km/h)", 0.0, 20.0,
                                                 float(cfg["regen"]["fade_end_kmh"]))
        regen_inputs["max_regen_power_kw"] = st.slider("Max regen power (kW)", 5.0, 105.0,
                                                       float(cfg["regen"]["max_regen_power_kw"]))
        if regen_inputs["fade_end_kmh"] >= regen_inputs["fade_start_kmh"]:
            st.warning("Fade end must be below fade start - using start − 1 km/h.")
            regen_inputs["fade_end_kmh"] = max(0.0, regen_inputs["fade_start_kmh"] - 1.0)

    sb.header("4 · OTA update")
    ota_enabled = sb.checkbox("Run OTA update during the drive", value=True)
    ota_inputs: dict[str, Any] = {"enabled": ota_enabled, "params": {}, "inject_invalid": False,
                                  "rollback": False, "rollback_t": 0.0, "t": 0.0, "version": "1.1.0"}
    if ota_enabled:
        ota_inputs["t"] = float(sb.number_input("Run OTA update at t = (s)", 0, duration, min(600, duration // 2), 10))
        ota_inputs["version"] = sb.text_input("New calibration version", "1.1.0")
        new_strat = sb.selectbox("New strategy after OTA", list(STRATEGY_LABELS), index=2,
                                 format_func=STRATEGY_LABELS.get, key="ota_strat")
        p: dict[str, Any] = {"strategy": new_strat}
        if new_strat == "levels":
            p["regen_level"] = sb.select_slider("New regen level", list(REGEN_LEVELS), value="Low", key="ota_lvl")
        if new_strat == "parallel":
            p["parallel_ratio"] = sb.slider("New regen share", 0.0, 1.0, 0.5, 0.05, key="ota_ratio")
        p["fade_start_kmh"] = sb.slider("New fade start (km/h)", 2.0, 25.0, 12.0, key="ota_fs")
        p["fade_end_kmh"] = sb.slider("New fade end (km/h)", 0.0, 20.0, 6.0, key="ota_fe")
        ota_inputs["params"] = p
        ota_inputs["inject_invalid"] = sb.checkbox("Inject UNSAFE values (demo: validator rejects)")
        ota_inputs["rollback"] = sb.checkbox("Roll back later")
        if ota_inputs["rollback"]:
            ota_inputs["rollback_t"] = float(sb.number_input("Rollback at t = (s)", 0, duration,
                                                             min(int(ota_inputs["t"]) + 300, duration), 10))

    sb.header("5 · Digital-twin sync")
    twin_inputs = {"enabled": sb.checkbox("Run twin vs physical vehicle", value=True),
                   "method": sb.radio("Correction method", ["kalman", "resync"], horizontal=True,
                                      format_func={"kalman": "Kalman blend", "resync": "Hard re-sync"}.get)}
    return cycle, f"{name}{f' ×{repeats}' if repeats > 1 else ''}", sim_inputs, regen_inputs, ota_inputs, twin_inputs


# =============================================================================
# KPI cards
# =============================================================================
def kpi_cards(kr: dict[str, float], kf: dict[str, float], cmp: dict[str, float]) -> None:
    c = st.columns(5)
    c[0].metric("Final SoC · regen", f"{kr['final_soc_pct']:.2f} %", f"{cmp['soc_saved_pct']:+.2f} % vs friction")
    c[1].metric("Final SoC · friction-only", f"{kf['final_soc_pct']:.2f} %")
    c[2].metric("kWh consumed (net)", f"{kr['kwh_consumed_net']:.3f}",
                f"{-cmp['energy_saved_kwh']:+.3f} kWh vs {kf['kwh_consumed_net']:.3f}", delta_color="inverse")
    c[3].metric("kWh recovered", f"{kr['kwh_recovered']:.3f}", f"{kr['regen_share_pct']:.1f} % of traction energy",
                delta_color="off")
    c[4].metric("Recovery efficiency", f"{kr['recovery_eff_pct']:.1f} %", "of wheel braking energy", delta_color="off")
    c = st.columns(5)
    c[0].metric("Consumption · regen (Wh/km)", f"{kr['wh_per_km']:.1f}",
                f"{kr['wh_per_km'] - kf['wh_per_km']:+.1f} vs {kf['wh_per_km']:.1f}", delta_color="inverse")
    c[1].metric("Est. range · regen", f"{kr['range_km']:.0f} km")
    c[2].metric("Est. range · friction-only", f"{kf['range_km']:.0f} km")
    c[3].metric("Range gain", f"{cmp['range_gain_pct']:+.1f} %")
    c[4].metric("Friction-brake energy saved", f"{cmp['friction_saved_kwh']:.3f} kWh",
                f"{cmp['friction_saved_pct']:.0f} % less pad wear (proxy)", delta_color="off")


# =============================================================================
# Tab 1: Live twin animation
# =============================================================================
LIVE_SIGNALS = [
    ("Vehicle.Speed", "speed_kmh", 1.0),
    ("Vehicle.Powertrain.TractionBattery.StateOfCharge.Current", "soc_pct", 1.0),
    ("Vehicle.Powertrain.TractionBattery.CurrentCurrent", "current_a", -1.0),
    ("Vehicle.Powertrain.TractionBattery.CurrentVoltage", "voltage_v", 1.0),
    ("Vehicle.Powertrain.TractionBattery.CurrentPower", "p_batt_w", -1.0),
    ("Vehicle.Powertrain.TractionMotor.Speed", "motor_speed_rpm", 1.0),
    ("Vehicle.Powertrain.TractionMotor.Torque", "motor_torque_nm", 1.0),
    ("Vehicle.Chassis.Brake.RegenPower", "p_regen_wheel_w", 1.0),
    ("Vehicle.Chassis.Brake.FrictionPower", "p_friction_w", 1.0),
    ("Vehicle.Chassis.Brake.RegenLimitReason", "regen_reason", None),
]


def _flow_label(p_batt_kw: float) -> tuple[str, str]:
    if p_batt_kw > 1.0:
        return "Battery → Motor (driving)", "#4575b4"
    if p_batt_kw < -0.05:
        return "Motor → Battery (REGEN)", C_REGEN
    return "Idle / aux loads only", "#999999"


def _live_values(r: pd.Series) -> list[str]:
    vals = []
    for path, col, sign in LIVE_SIGNALS:
        v = r[col]
        vals.append(str(v or "-") if sign is None else f"{sign * float(v):,.2f}")
    return vals


def live_figure(df: pd.DataFrame, speedup: float, max_frames: int = 300) -> go.Figure:
    n = len(df)
    stride = max(1, math.ceil(n / max_frames))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    vmax = max(140.0, float(df["speed_kmh"].max()) * 1.1)
    p_lo = min(-50.0, float(df["p_batt_w"].min()) / 1000 * 1.1)
    p_hi = max(60.0, float(df["p_batt_w"].max()) / 1000 * 1.1)

    # Only indicator + table traces are animated: Plotly's redraw-animation does not render
    # xy subplots reliably next to them, so the speed/SoC trace is a separate static chart.
    fig = make_subplots(
        rows=2, cols=3, row_heights=[0.5, 0.5], vertical_spacing=0.08,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "table", "colspan": 3}, None, None]],
    )

    def gauges(r: pd.Series) -> list[go.Indicator]:
        pkw = r["p_batt_w"] / 1000.0
        label, color = _flow_label(pkw)
        return [
            go.Indicator(mode="gauge+number", value=r["speed_kmh"], title={"text": "Speed (km/h)"},
                         gauge={"axis": {"range": [0, vmax]}, "bar": {"color": C_SPEED}}),
            go.Indicator(mode="gauge+number", value=r["soc_pct"], number={"valueformat": ".2f", "suffix": " %"},
                         title={"text": "State of Charge"},
                         gauge={"axis": {"range": [0, 100]}, "bar": {"color": C_REGEN},
                                "steps": [{"range": [0, 20], "color": "#fde0dd"},
                                          {"range": [90, 100], "color": "#fff7bc"}]}),
            go.Indicator(mode="gauge+number", value=pkw, number={"valueformat": ".1f", "suffix": " kW"},
                         title={"text": f"Power flow<br><span style='font-size:0.8em;color:{color}'>{label}</span>"},
                         gauge={"axis": {"range": [p_lo, p_hi]}, "bar": {"color": color},
                                "threshold": {"line": {"color": "black", "width": 2}, "value": 0}}),
        ]

    def table(r: pd.Series) -> go.Table:
        units = [VSS_CATALOG[p].unit for p, _, _ in LIVE_SIGNALS]
        names = [p + (" (custom)" if VSS_CATALOG[p].custom else "") for p, _, _ in LIVE_SIGNALS]
        return go.Table(header={"values": [f"<b>VSS signal @ t = {r['t_s']:.0f} s</b>", "<b>Value</b>", "<b>Unit</b>"],
                                "align": "left", "fill_color": "#2c3e50", "font": {"color": "white"}},
                        cells={"values": [names, _live_values(r), units], "align": "left", "height": 22,
                               "fill_color": "#f4f6f9", "font": {"color": "#111111"}},
                        columnwidth=[3.2, 1, 0.6])

    first = df.iloc[0]
    for i, g in enumerate(gauges(first)):
        fig.add_trace(g, row=1, col=i + 1)
    fig.add_trace(table(first), row=2, col=1)

    frames = [go.Frame(name=str(int(df.iloc[k]["t_s"])), data=[*gauges(df.iloc[k]), table(df.iloc[k])],
                       traces=[0, 1, 2, 3]) for k in idx]
    fig.frames = frames
    dur = max(30, int(stride * 1000 / speedup))
    fig.update_layout(
        height=720, margin={"t": 110, "b": 10, "l": 10, "r": 10}, showlegend=False,
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.0, "y": 1.17, "xanchor": "left", "yanchor": "top",
            "bgcolor": "#eeeeee", "font": {"color": "#111111"}, "pad": {"t": 10},
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": dur, "redraw": True}, "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.16, "len": 0.84, "y": 1.19, "yanchor": "top", "pad": {"t": 0},
            "currentvalue": {"prefix": "t = ", "suffix": " s"},
            "steps": [{"label": f.name, "method": "animate",
                       "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                      for f in frames],
        }],
    )
    return fig


def trip_overview_figure(df: pd.DataFrame) -> go.Figure:
    """Static speed + SoC + regen-power overview shown under the animation."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["p_regen_batt_w"] / 1000.0, name="Regen power to battery (kW)",
                             fill="tozeroy", line={"color": C_REGEN, "width": 0.5}, opacity=0.5))
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["speed_kmh"], name="Speed (km/h)", line={"color": C_SPEED}))
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_pct"], name="SoC (%)", line={"color": "#e08214", "width": 2}),
                  secondary_y=True)
    fig.update_layout(height=320, margin={"t": 30, "b": 10}, legend={"orientation": "h", "y": 1.15},
                      hovermode="x unified")
    fig.update_yaxes(title_text="km/h · kW", secondary_y=False)
    fig.update_yaxes(title_text="SoC (%)", secondary_y=True)
    fig.update_xaxes(title_text="Time (s)")
    return fig


def tab_live(sim: dict[str, Any]) -> None:
    regen: SimulationResult = sim["regen"]
    st.subheader("Live twin playback - regenerative-braking case")
    c1, c2 = st.columns([3, 1])
    speedup = c2.select_slider("Playback speed", [1, 2, 5, 10, 20, 50, 100], value=20,
                               format_func=lambda x: f"{x}× real time")
    c1.caption("Press ▶ Play inside the chart. Gauges, power-flow indicator and the VSS signal table update "
               "every frame; drag the time slider to scrub. (Actual playback rate depends on your computer.)")
    st.plotly_chart(live_figure(regen.df, speedup), width="stretch")
    st.plotly_chart(trip_overview_figure(regen.df), width="stretch")

    st.markdown("##### Signal bus snapshot (what an SDV app would read from the data broker)")
    hist: pd.DataFrame = sim["bus_history"]
    stats = sim["bus_stats"]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Messages published", f"{stats['messages']:,}")
    s2.metric("Subscribers", stats["subscribers"])
    s3.metric("Brake.* messages received", f"{stats['brake_msgs']:,}")
    s4.metric("TractionBattery.* received", f"{stats['battery_msgs']:,}")
    if not hist.empty:
        t_pick = st.slider("Snapshot time (s)", int(hist["t_s"].min()), int(hist["t_s"].max()),
                           int(hist["t_s"].min()))
        row = hist.loc[(hist["t_s"] - t_pick).abs().idxmin()]
        snap = pd.DataFrame([{"VSS path": p, "value": str(row[p]), "unit": VSS_CATALOG[p].unit,
                              "custom?": "custom" if VSS_CATALOG[p].custom else ""}
                             for p in VSS_CATALOG if p in row.index])
        st.dataframe(snap, width="stretch", hide_index=True)


# =============================================================================
# Tab 2: SoC comparison
# =============================================================================
def tab_soc(sim: dict[str, Any]) -> go.Figure:
    regen: SimulationResult = sim["regen"]
    fric: SimulationResult = sim["friction"]
    t_r, soc_r = soc_series_with_start(regen)
    t_f, soc_f = soc_series_with_start(fric)
    brake = (regen.df["p_brake_req_w"] > 0).astype(float)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                        specs=[[{"secondary_y": True}], [{}], [{}]],
                        subplot_titles=("SoC: regenerative vs friction-only braking",
                                        "SoC advantage of regen (percentage points)", "Speed trace"))
    fig.add_trace(go.Scatter(x=regen.df["t_s"], y=brake, fill="tozeroy", mode="none", fillcolor=C_BRAKE_SHADE,
                             name="Braking event", line_shape="hv"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=t_f, y=soc_f, name="Friction-only", line={"color": C_FRIC, "width": 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_r, y=soc_r, name="Regenerative", line={"color": C_REGEN, "width": 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_r, y=soc_r - soc_f, name="ΔSoC", fill="tozeroy",
                             line={"color": C_REGEN, "width": 1}), row=2, col=1)
    fig.add_trace(go.Scatter(x=regen.df["t_s"], y=regen.df["speed_kmh"], name="Speed",
                             line={"color": C_SPEED, "width": 1}), row=3, col=1)
    fig.update_yaxes(title_text="SoC (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 1], visible=False, row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Δ (pp)", row=2, col=1)
    fig.update_yaxes(title_text="km/h", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(height=720, legend={"orientation": "h", "y": 1.08}, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    cum = make_subplots(specs=[[{"secondary_y": False}]])
    cum.add_trace(go.Scatter(x=regen.df["t_s"], y=regen.df["p_regen_batt_w"].cumsum() / 3600.0,
                             name="Energy recovered to battery (Wh)", line={"color": C_REGEN}))
    cum.add_trace(go.Scatter(x=regen.df["t_s"], y=regen.df["p_friction_w"].cumsum() / 3600.0,
                             name="Friction heat · regen case (Wh)", line={"color": "#fdae61"}))
    cum.add_trace(go.Scatter(x=fric.df["t_s"], y=fric.df["p_friction_w"].cumsum() / 3600.0,
                             name="Friction heat · friction-only (Wh)", line={"color": C_FRIC, "dash": "dash"}))
    cum.update_layout(title="Cumulative braking energy", height=380, yaxis_title="Wh", xaxis_title="Time (s)",
                      legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(cum, width="stretch")
    return fig


# =============================================================================
# Tab 3: Energy breakdown
# =============================================================================
def sankey_figure(e: dict[str, float]) -> go.Figure:
    nodes = ["Battery (gross out)", "Battery I²R loss", "Auxiliaries (HVAC, 12 V)", "Motor + inverter",
             "Drive conversion loss", "Wheels (traction)", "Aerodynamic drag", "Rolling resistance",
             "Climbing (grade)", "Friction-brake heat", "Regen (motor as generator)", "Regen conversion loss",
             "♻ Recovered → battery", "Gravity (downhill)", "Kinetic energy left"]
    ix = {n: i for i, n in enumerate(nodes)}
    links: list[tuple[str, str, float]] = [
        ("Battery (gross out)", "Battery I²R loss", e["batt_loss_wh"]),
        ("Battery (gross out)", "Auxiliaries (HVAC, 12 V)", e["aux_wh"]),
        ("Battery (gross out)", "Motor + inverter", e["motor_in_wh"]),
        ("Motor + inverter", "Drive conversion loss", e["drive_conversion_loss_wh"]),
        ("Motor + inverter", "Wheels (traction)", e["traction_wheel_wh"] - e["unmet_traction_wh"]),
        ("Wheels (traction)", "Aerodynamic drag", e["aero_wh"]),
        ("Wheels (traction)", "Rolling resistance", e["rolling_wh"]),
        ("Wheels (traction)", "Friction-brake heat", e["friction_wh"]),
        ("Wheels (traction)", "Regen (motor as generator)", e["regen_wheel_wh"]),
        ("Regen (motor as generator)", "Regen conversion loss", e["regen_conversion_loss_wh"]),
        ("Regen (motor as generator)", "♻ Recovered → battery", e["regen_batt_wh"]),
    ]
    if e["grade_wh"] >= 0:
        links.append(("Wheels (traction)", "Climbing (grade)", e["grade_wh"]))
    else:
        links.append(("Gravity (downhill)", "Wheels (traction)", -e["grade_wh"]))
    if e["kinetic_change_wh"] > 0.5:
        links.append(("Wheels (traction)", "Kinetic energy left", e["kinetic_change_wh"]))
    links = [(a, b, v) for a, b, v in links if v > 0.05]
    colors = ["#4575b4", "#bbbbbb", "#bbbbbb", "#74add1", "#bbbbbb", "#313695", "#abd9e9", "#abd9e9",
              "#abd9e9", C_FRIC, "#66bd63", "#bbbbbb", C_REGEN, "#abd9e9", "#abd9e9"]
    fig = go.Figure(go.Sankey(
        valueformat=".0f", valuesuffix=" Wh",
        node={"label": nodes, "pad": 18, "thickness": 18, "color": colors},
        link={"source": [ix[a] for a, _, _ in links], "target": [ix[b] for _, b, _ in links],
              "value": [v for _, _, v in links],
              "color": ["rgba(26,152,80,0.35)" if "Regen" in a or "Recovered" in b or "Regen" in b
                        else "rgba(215,48,39,0.30)" if "Friction" in b else "rgba(150,150,150,0.25)"
                        for a, b, _ in links]},
    ))
    fig.update_layout(height=520, margin={"t": 30, "b": 10})
    return fig


def tab_energy(sim: dict[str, Any], cfg: dict[str, Any]) -> go.Figure:
    er, ef = energy_totals(sim["regen"]), energy_totals(sim["friction"])
    st.subheader("Where does the energy go? (regenerative case)")
    st.caption("All values in Wh over the whole drive. Green = recovered energy, red = wasted as brake heat.")
    st.plotly_chart(sankey_figure(er), width="stretch")

    cats = [("Net battery energy", "batt_net_wh"), ("Recovered by regen", "regen_batt_wh"),
            ("Friction-brake heat", "friction_wh"), ("Aerodynamic drag", "aero_wh"), ("Rolling", "rolling_wh"),
            ("Battery I²R loss", "batt_loss_wh"), ("Drive conversion loss", "drive_conversion_loss_wh"),
            ("Auxiliaries", "aux_wh")]
    bars = go.Figure([
        go.Bar(name="Friction-only", x=[c for c, _ in cats], y=[ef[k] for _, k in cats], marker_color=C_FRIC),
        go.Bar(name="Regenerative", x=[c for c, _ in cats], y=[er[k] for _, k in cats], marker_color=C_REGEN),
    ])
    bars.update_layout(barmode="group", height=420, yaxis_title="Energy (Wh)", title="Energy categories: both cases",
                       legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(bars, width="stretch")

    c1, c2 = st.columns(2)
    reasons = sim["regen"].df.query("p_brake_req_w > 0")
    lost = (reasons["p_brake_req_w"] - reasons["p_regen_wheel_w"]).groupby(reasons["regen_reason"]).sum() / 3600
    pie = go.Figure(go.Pie(labels=lost.index, values=lost.values, hole=0.45))
    pie.update_layout(title="Braking energy NOT recovered, by limiting reason (Wh)", height=380)
    c1.plotly_chart(pie, width="stretch")

    motor = MotorModel(MotorParams.from_config(cfg))
    rpm, tq, eta = motor.efficiency_map()
    df = sim["regen"].df
    emap = go.Figure(go.Contour(x=rpm, y=tq, z=eta * 100, colorscale="Viridis", contours={"start": 70, "end": 96,
                                "size": 2, "showlabels": True}, colorbar={"title": "η %"}))
    op = df[df["motor_eff"] > 0]
    emap.add_trace(go.Scatter(x=op["motor_speed_rpm"], y=op["motor_torque_nm"].abs(), mode="markers",
                              marker={"size": 3, "color": np.where(op["motor_torque_nm"] < 0, C_REGEN, "white")},
                              name="Operating points (green = regen)"))
    emap.update_layout(title="Motor+inverter efficiency map with operating points", height=380,
                       xaxis_title="Motor speed (rpm)", yaxis_title="|Torque| (N·m)", showlegend=False)
    c2.plotly_chart(emap, width="stretch")
    return bars


# =============================================================================
# Tab 4: Braking events
# =============================================================================
def tab_events(sim: dict[str, Any]) -> None:
    ev = braking_events(sim["regen"])
    if ev.empty:
        st.info("No braking events in this cycle.")
        return
    st.subheader(f"{len(ev)} braking events")
    a, b, c, d = st.columns(4)
    a.metric("Kinetic energy available", f"{ev['kinetic_energy_wh'].sum():.0f} Wh")
    b.metric("Recovered to battery", f"{ev['recovered_wh'].sum():.0f} Wh")
    c.metric("Mean recovery per event", f"{ev['recovery_pct'].mean():.1f} %")
    d.metric("Events fully regen", f"{(ev['regen_limited_by'] == 'None (full regen)').sum()} / {len(ev)}")
    reasons = sorted(ev["regen_limited_by"].unique())
    pick = st.multiselect("Filter by limiting reason", reasons, default=reasons)
    view = ev[ev["regen_limited_by"].isin(pick)]
    st.dataframe(
        view.style.format({"start_s": "{:.0f}", "end_s": "{:.0f}", "duration_s": "{:.0f}", "v_start_kmh": "{:.1f}",
                           "v_end_kmh": "{:.1f}", "kinetic_energy_wh": "{:.1f}", "brake_request_wh": "{:.1f}",
                           "recovered_wh": "{:.1f}", "friction_wh": "{:.1f}", "recovery_pct": "{:.1f}"})
        .background_gradient(subset=["recovery_pct"], cmap="RdYlGn", vmin=0, vmax=100),
        width="stretch", hide_index=True, height=420,
        column_config={"kinetic_energy_wh": "KE available (Wh)", "recovered_wh": "Recovered (Wh)",
                       "recovery_pct": "Recovery %", "regen_limited_by": "Regen limited by"},
    )
    fig = go.Figure(go.Scatter(x=ev["v_start_kmh"], y=ev["recovery_pct"], mode="markers",
                               marker={"size": np.clip(ev["kinetic_energy_wh"], 4, 30)},
                               text=ev["regen_limited_by"], hovertemplate="v₀ %{x:.0f} km/h<br>%{y:.0f} %<br>%{text}"))
    fig.update_layout(title="Recovery % vs braking start speed (bubble = kinetic energy)", height=380,
                      xaxis_title="Speed at start of braking (km/h)", yaxis_title="Recovered / kinetic energy (%)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Low-speed stops lose relatively more (fade + road loads eat the energy); very high SoC or cold "
               "batteries show 'SoC limit' / 'Temperature derate'.")


# =============================================================================
# Tab 5: OTA & twin sync
# =============================================================================
def tab_ota_twin(sim: dict[str, Any]) -> None:
    st.subheader("Over-the-air (OTA) calibration update")
    ota = sim["ota"]
    if ota is None:
        st.info("Enable 'Run OTA update during the drive' in the sidebar.")
    else:
        res: SimulationResult = ota["result"]
        events = pd.DataFrame(ota["events"])
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown("**OTA event log**")
            st.dataframe(events.rename(columns={"t_s": "t (s)"}), hide_index=True, width="stretch")
            if ota["diffs"]:
                d = ota["diffs"][0]
                st.markdown(f"**Parameter diff** `{d['from']}` → `{d['to']}`")
                st.dataframe(pd.DataFrame([{"parameter": k, "before": str(a), "after": str(b)}
                                           for k, (a, b) in d["diff"].items()]), hide_index=True,
                             width="stretch")
            elif (events["kind"] == "REJECTED").any():
                st.error("Package REJECTED by the validator - the car kept running the old calibration.")
        with col_b:
            t_b, soc_b = soc_series_with_start(sim["regen"])
            t_o, soc_o = soc_series_with_start(res)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_b, y=soc_b, name="Regen, no OTA", line={"color": C_REGEN}))
            fig.add_trace(go.Scatter(x=t_o, y=soc_o, name="Regen with OTA", line={"color": C_OTA, "dash": "dash"}))
            colors = {"INSTALLED": C_OTA, "REJECTED": C_FRIC, "ROLLED_BACK": "#f46d43", "HEALTH_OK": "#999999"}
            for _, e in events[events["kind"] != "SCHEDULED"].iterrows():
                fig.add_vline(x=e["t_s"], line_dash="dot", line_color=colors.get(e["kind"], "grey"),
                              annotation_text=f"{e['kind']} {e['version']}", annotation_textangle=-90)
            fig.update_layout(title="Before/after effect of the OTA update on SoC", height=420,
                              xaxis_title="Time (s)", yaxis_title="SoC (%)", legend={"orientation": "h", "y": -0.2})
            st.plotly_chart(fig, width="stretch")
            dsoc = res.final_soc_pct - sim["regen"].final_soc_pct
            st.metric("Final SoC effect of OTA", f"{res.final_soc_pct:.2f} %", f"{dsoc:+.3f} pp vs no OTA")

        vers = res.df[["t_s", "regen_version"]]
        tl = go.Figure(go.Scatter(x=vers["t_s"], y=vers["regen_version"], mode="lines", line_shape="hv",
                                  line={"color": C_OTA, "width": 3}))
        tl.update_layout(title="Running calibration version (Vehicle.Software.RegenCalibration.Version)",
                         height=220, xaxis_title="Time (s)", margin={"t": 40, "b": 30})
        st.plotly_chart(tl, width="stretch")

    st.divider()
    st.subheader("Digital-twin synchronisation: twin vs 'physical' vehicle")
    twin: TwinSyncResult | None = sim["twin"]
    if twin is None:
        st.info("Enable 'Run twin vs physical vehicle' in the sidebar.")
        return
    m = twin.metrics
    a, b, c, d = st.columns(4)
    a.metric("RMSE open-loop twin", f"{m['rmse_open_pct']:.3f} %")
    b.metric("RMSE corrected twin", f"{m['rmse_corrected_pct']:.3f} %",
             f"{m['rmse_corrected_pct'] - m['rmse_open_pct']:+.3f}", delta_color="inverse")
    c.metric("RMSE raw SoC sensor", f"{m['rmse_raw_sensor_pct']:.3f} %")
    d.metric("Sync updates", f"{m['n_syncs']:.0f}", f"final gain K = {m['final_kalman_gain']:.2f}", delta_color="off")
    df = twin.df
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06,
                        subplot_titles=("SoC", "Twin error vs physical truth (pp)"))
    meas = df.iloc[:: twin.sync_period_s]
    fig.add_trace(go.Scatter(x=meas["t_s"], y=meas["soc_measured_pct"], mode="markers", name="Measured (noisy)",
                             marker={"size": 4, "color": "#999999"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_true_pct"], name="Physical vehicle (truth)",
                             line={"color": "black", "width": 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_twin_open_pct"], name="Twin, open loop",
                             line={"color": C_FRIC, "dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_twin_corrected_pct"], name=f"Twin, corrected ({twin.method})",
                             line={"color": C_REGEN}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_twin_open_pct"] - df["soc_true_pct"], name="error open",
                             line={"color": C_FRIC}, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_twin_corrected_pct"] - df["soc_true_pct"], name="error corr",
                             line={"color": C_REGEN}, showlegend=False), row=2, col=1)
    fig.update_layout(height=620, legend={"orientation": "h", "y": 1.08}, hovermode="x unified")
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    st.plotly_chart(fig, width="stretch")
    tw = load_config()["twin"]
    st.caption(f"Physical vehicle has hidden +{tw['mass_mismatch_pct']:.0f} % mass and +{tw['r0_mismatch_pct']:.0f} % "
               f"battery resistance; sensors add noise (SoC σ = {tw['soc_noise_std_pct']} %). The twin starts with a "
               f"{tw['initial_soc_error_pct']} % SoC error. Without feedback it never recovers; with feedback every "
               f"{tw['sync_period_s']} s it converges - that feedback loop is what makes it a digital TWIN.")


# =============================================================================
# Tab 6: Theory
# =============================================================================
BLOCK_DIAGRAM = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  cycle [label="Drive cycle\\nv(t)"]; dyn [label="Vehicle dynamics\\nF_roll+F_aero+F_grade+F_inertia"];
  ctrl [label="Regen controller\\n(blending strategy)", fillcolor="#e5f5e0"];
  motor [label="Motor + inverter\\nη(T, ω), limits"]; fric [label="Friction brakes\\n(heat)", fillcolor="#fde0dd"];
  batt [label="Battery 1-RC ECM\\nOCV, R0, R1‖C1, SoC"];
  bus [label="VSS signal bus\\n(pub/sub)", fillcolor="#fff7bc"]; ota [label="OTA manager\\nvalidate / rollback", fillcolor="#f2e5f7"];
  twin [label="Twin sync\\nKalman blend", fillcolor="#f2e5f7"]; phys [label="Physical vehicle\\n(sensors + noise)", shape=component];
  dash [label="Streamlit dashboard", shape=note];
  cycle -> dyn -> ctrl [label="P_wheel"]; ctrl -> motor [label="P_regen / traction"]; ctrl -> fric [label="P_friction"];
  motor -> batt [label="P_batt"]; batt -> ctrl [label="charge limit", style=dashed];
  batt -> bus; motor -> bus; ctrl -> bus; bus -> dash; ota -> ctrl [label="new calibration"];
  phys -> twin [label="measured SoC"]; twin -> batt [label="SoC correction", style=dashed]; twin -> dash;
}
"""


def tab_theory() -> None:
    st.subheader("System block diagram")
    st.graphviz_chart(BLOCK_DIAGRAM, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1 · Road-load forces (mnemonic **R-A-G-I**: Rolling, Aero, Grade, Inertia)")
        st.latex(r"F_{total}=F_{roll}+F_{aero}+F_{grade}+F_{inertia}")
        st.latex(r"F_{roll}=m\,g\,C_{rr}\cos\theta \qquad F_{aero}=\tfrac12\rho\,C_d\,A\,v^2")
        st.latex(r"F_{grade}=m\,g\sin\theta \qquad F_{inertia}=m\,\delta\,a")
        st.latex(r"P_{wheel}=F_{total}\cdot v\quad(<0\Rightarrow\text{braking})")
        st.markdown("#### 2 · Motor / inverter")
        st.latex(r"T_{max}(\omega)=\min\!\left(T_{peak},\frac{P_{peak}}{\omega}\right),\quad "
                 r"\omega_m=\frac{v}{r}G")
        st.latex(r"P_{loss}=k_cT^2+k_i\omega+k_w\omega^3+P_0")
        st.latex(r"\text{Motoring: }P_{batt}=\frac{P_{wheel}}{\eta_d\,\eta_m}\qquad "
                 r"\text{Generating: }P_{batt}=P_{wheel}\,\eta_d\,\eta_m")
    with c2:
        st.markdown("#### 3 · Battery (1-RC Thevenin)")
        st.latex(r"V=OCV(SoC)-I R_0-V_1,\qquad \frac{dV_1}{dt}=-\frac{V_1}{R_1C_1}+\frac{I}{C_1}")
        st.latex(r"I=\frac{(OCV-V_1)-\sqrt{(OCV-V_1)^2-4R_0P}}{2R_0}")
        st.latex(r"SoC_{k+1}=SoC_k-\frac{\eta_c\,I\,\Delta t}{3600\,Q_{Ah}}")
        st.latex(r"P_{chg,max}=P_{rated}\cdot f_{SoC}(SoC)\cdot f_T(T)")
        st.markdown("#### 4 · Regen blending (three limits: **M-B-S** = Motor, Battery, Strategy)")
        st.latex(r"P_{regen}=f_{fade}(v)\cdot\min(P_{req},P_{motor},P_{batt},P_{strategy})")
        st.latex(r"P_{friction}=P_{req}-P_{regen}")
        st.latex(r"\eta_{recovery}=\frac{E_{regen\to batt}}{E_{brake\,request}}\times100\%")
        st.markdown("#### 5 · Twin correction (scalar Kalman)")
        st.latex(r"P\leftarrow P+Q\Delta t,\;K=\frac{P}{P+R},\;SoC\leftarrow SoC+K(z-SoC)")
    st.markdown("#### VSS signal catalogue used on the bus")
    st.dataframe(catalog_frame(), width="stretch", hide_index=True)


# =============================================================================
# Export
# =============================================================================
def export_results(sim: dict[str, Any], cycle_name: str, kpis: dict[str, dict[str, float]]) -> tuple[bytes, bytes, str]:
    regen, fric = sim["regen"].df, sim["friction"].df
    out = regen.add_prefix("regen_").rename(columns={"regen_t_s": "t_s"})
    for col in ["soc_pct", "p_batt_w", "current_a", "voltage_v", "p_friction_w"]:
        out[f"friction_{col}"] = fric[col].to_numpy()
    csv_bytes = out.to_csv(index=False).encode("utf-8")

    t_r, soc_r = soc_series_with_start(sim["regen"])
    t_f, soc_f = soc_series_with_start(sim["friction"])
    er, ef = energy_totals(sim["regen"]), energy_totals(sim["friction"])
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].plot(t_f, soc_f, color=C_FRIC, label="Friction-only")
    ax[0, 0].plot(t_r, soc_r, color=C_REGEN, label="Regenerative")
    ax[0, 0].fill_between(regen["t_s"], soc_r.min(), soc_r.max(), where=regen["p_brake_req_w"] > 0,
                          color="orange", alpha=0.15, label="Braking")
    ax[0, 0].set(title="State of charge", xlabel="Time (s)", ylabel="SoC (%)")
    ax[0, 0].legend()
    ax[0, 1].plot(regen["t_s"], regen["speed_kmh"], color=C_SPEED)
    ax[0, 1].set(title=f"Drive cycle: {cycle_name}", xlabel="Time (s)", ylabel="Speed (km/h)")
    ax[1, 0].plot(regen["t_s"], regen["p_batt_w"] / 1000, color="#555555", lw=0.8)
    ax[1, 0].fill_between(regen["t_s"], 0, -regen["p_regen_batt_w"] / 1000, color=C_REGEN, alpha=0.6, label="Regen")
    ax[1, 0].set(title="Battery power (+ discharge)", xlabel="Time (s)", ylabel="kW")
    ax[1, 0].legend()
    labels = ["Net battery", "Recovered", "Friction heat", "Aero", "Rolling"]
    keys = ["batt_net_wh", "regen_batt_wh", "friction_wh", "aero_wh", "rolling_wh"]
    x = np.arange(len(labels))
    ax[1, 1].bar(x - 0.2, [ef[k] for k in keys], 0.4, color=C_FRIC, label="Friction-only")
    ax[1, 1].bar(x + 0.2, [er[k] for k in keys], 0.4, color=C_REGEN, label="Regenerative")
    ax[1, 1].set_xticks(x, labels)
    ax[1, 1].set(title="Energy (Wh)")
    ax[1, 1].legend()
    kr, kf = kpis["regen"], kpis["friction"]
    fig.suptitle(f"EV Regen Digital Twin - {cycle_name}:  final SoC {kr['final_soc_pct']:.2f} % vs "
                 f"{kf['final_soc_pct']:.2f} %,  {kr['wh_per_km']:.0f} vs {kf['wh_per_km']:.0f} Wh/km,  "
                 f"recovered {kr['kwh_recovered']:.2f} kWh", fontsize=12)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    png_bytes = buf.getvalue()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() else "_" for ch in cycle_name)[:30]
    folder = PROJECT_ROOT / "exports"
    folder.mkdir(exist_ok=True)
    (folder / f"results_{safe}_{stamp}.csv").write_bytes(csv_bytes)
    (folder / f"charts_{safe}_{stamp}.png").write_bytes(png_bytes)
    return csv_bytes, png_bytes, f"exports/results_{safe}_{stamp}.csv  +  exports/charts_{safe}_{stamp}.png"


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    cfg = load_config()
    cycle, cycle_name, sim_inputs, regen_inputs, ota_inputs, twin_inputs = sidebar(cfg)
    sim = simulate_everything(cycle, sim_inputs, regen_inputs, ota_inputs, twin_inputs)
    kr, kf = compute_kpis(sim["regen"]), compute_kpis(sim["friction"])
    cmp = compare_kpis(kr, kf)
    stats = cycle_stats(cycle)

    st.title("EV Energy-Recovery Digital Twin")
    st.caption(f"**{cycle_name}** · {stats['distance_km']:.2f} km · {stats['duration_s']:.0f} s · "
               f"max {stats['max_speed_kmh']:.0f} km/h · mean {stats['mean_speed_kmh']:.1f} km/h · "
               f"strategy: {STRATEGY_LABELS[regen_inputs['strategy']]} · vehicle: {cfg['vehicle']['name']}")
    kpi_cards(kr, kf, cmp)

    items = sanity_check(kr, stats["mean_speed_kmh"], cfg, regen_inputs["strategy"])
    icon = {"OK": "✅", "WARN": "⚠️", "FLAG": "🚩"}
    with st.expander("Sanity check against realistic ranges  " + " ".join(icon[i.status] for i in items)):
        for it in items:
            st.write(f"{icon[it.status]} **{it.metric}** = {it.value:.1f} (expected {it.expected[0]:.0f}–"
                     f"{it.expected[1]:.0f}) - {it.note}")
        flags = sim["regen"].flags
        if flags["trace_miss_steps"]:
            st.warning(f"Vehicle could not follow the speed trace for {flags['trace_miss_steps']} s (motor power limit).")
        if flags["battery_depleted"]:
            st.error("Battery reached 0 % SoC during the drive.")

    if st.button("💾 Export results (CSV + PNG)"):
        csv_b, png_b, where = export_results(sim, cycle_name, {"regen": kr, "friction": kf})
        st.session_state["export"] = (csv_b, png_b, where)
    if "export" in st.session_state:
        csv_b, png_b, where = st.session_state["export"]
        st.success(f"Saved to {where}")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV", csv_b, "ev_regen_results.csv", "text/csv")
        d2.download_button("⬇ Download PNG", png_b, "ev_regen_charts.png", "image/png")

    tabs = st.tabs(["🚗 Live Twin", "📉 SoC Comparison", "⚡ Energy Breakdown", "🛑 Braking Events",
                    "📡 OTA & Twin Sync", "📘 Theory"])
    with tabs[0]:
        tab_live(sim)
    with tabs[1]:
        tab_soc(sim)
    with tabs[2]:
        tab_energy(sim, cfg)
    with tabs[3]:
        tab_events(sim)
    with tabs[4]:
        tab_ota_twin(sim)
    with tabs[5]:
        tab_theory()


if __name__ == "__main__":
    main()
