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
from core.regen_controller import REGEN_LEVELS, BrakeContext, RegenParams  # noqa: E402
from core.simulator import SimulationResult, Simulator  # noqa: E402
from sdv.ota_update import OtaManager, OtaPackage  # noqa: E402
from sdv.signals import VSS_CATALOG, SignalBus, SignalPublisher, catalog_frame  # noqa: E402
from sdv.twin_sync import TwinSyncResult, run_twin_sync  # noqa: E402
from ui import theme  # noqa: E402  (also registers the Plotly template)
from ui.animations import energy_flow, regen_explainer  # noqa: E402

st.set_page_config(page_title="EV Regen Digital Twin", page_icon="🔋", layout="wide")

C_REGEN = theme.REGEN    # green  = regenerative braking case
C_FRIC = theme.FRIC      # red    = friction-only baseline
C_OTA = theme.OTA        # purple = OTA case
C_SPEED = theme.DRIVE
C_SOC = theme.VOLT
C_BRAKE_SHADE = theme.BRAKE_SHADE
def show_html(body: str) -> None:
    """Render our own trusted HTML/CSS/SVG widgets unsanitised (the sanitiser drops <style> and <svg>)."""
    st.html(body, unsafe_allow_javascript=True)


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
    sb.html(f"""<div style="display:flex;gap:10px;align-items:center;margin:2px 0 6px">
      <div style="width:38px;height:38px;border-radius:11px;display:grid;place-items:center;font-size:1.25rem;
                  background:linear-gradient(135deg,{theme.REGEN},{theme.DRIVE})">⚡</div>
      <div><div style="font:700 1.1rem {theme.FONT_HEAD};color:{theme.TEXT}">EV Regen Twin</div>
      <div style="font:.72rem {theme.FONT_MONO};color:{theme.MUTED}">Software-Defined Vehicles</div></div></div>""")

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
    show_html(theme.kpi_grid_html(kr, kf, cmp))


# =============================================================================
# Tab 1: Live twin animation
# =============================================================================
def trip_overview_figure(df: pd.DataFrame) -> go.Figure:
    """Static speed + SoC + regen-power overview shown under the animation."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["p_regen_batt_w"] / 1000.0, name="Regen power to battery (kW)",
                             fill="tozeroy", line={"color": C_REGEN, "width": 0.5}, opacity=0.5))
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["speed_kmh"], name="Speed (km/h)", line={"color": C_SPEED}))
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_pct"], name="SoC (%)", line={"color": C_SOC, "width": 2}),
                  secondary_y=True)
    fig.update_layout(height=320, margin={"t": 30, "b": 10}, legend={"orientation": "h", "y": 1.15},
                      hovermode="x unified")
    fig.update_yaxes(title_text="km/h · kW", secondary_y=False)
    fig.update_yaxes(title_text="SoC (%)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Time (s)")
    return fig


def tab_live(sim: dict[str, Any], cfg: dict[str, Any], kr: dict[str, float]) -> None:
    regen: SimulationResult = sim["regen"]
    show_html(theme.section("Live digital twin", "Watch the energy move, second by second",
                          "The car replays the simulated trip. Blue dots carry battery power to the wheels, green dots "
                          "carry braking energy back into the battery and red glow is energy lost as brake heat. "
                          "The battery race on the right compares the same trip with friction brakes only.",
                          theme.REGEN))
    energy_flow(regen, sim["friction"], cfg["vehicle"]["aux_power_w"] / 1000.0, kr["wh_per_km"])
    st.plotly_chart(trip_overview_figure(regen.df), width="stretch", theme=None)

    show_html(theme.section("Signal bus", "What an SDV app would read from the data broker",
                          "Every simulation step publishes COVESA VSS signals on an in-memory pub/sub bus.", theme.VOLT))
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
# Tab 2: How regen works (animated explainer + blending playground)
# =============================================================================
def blend_preview(cfg: dict[str, Any], regen_inputs: dict[str, Any], v_kmh: float, p_req_kw: float,
                  soc_pct: float, temp_c: float) -> tuple[BrakeContext, Any]:
    """One braking instant evaluated with the real motor, battery and controller models."""
    sim = Simulator(cfg, RegenParams.from_config(cfg, **regen_inputs), initial_soc_pct=soc_pct, ambient_temp_c=temp_c)
    v = max(v_kmh / 3.6, 0.01)
    p_req = p_req_kw * 1000.0
    omega, torque_wheel_equiv = sim.dynamics.wheel_to_motor(v, -p_req / v)
    eta_d = sim.vehicle_params.drivetrain_efficiency
    eta_est = sim.motor.efficiency(abs(torque_wheel_equiv) * eta_d, omega, generating=True)
    ctx = BrakeContext(p_request_w=p_req, speed_mps=v, mass_kg=sim.vehicle_params.mass_kg,
                       motor_limit_w=sim.motor.max_power_w(omega) / eta_d,
                       battery_limit_w=sim.battery.max_charge_power_w() / (eta_d * eta_est) if eta_est > 0 else 0.0,
                       battery_reason=sim.battery.charge_limit_reason())
    return ctx, sim.controller.split(ctx)


def tab_how(cfg: dict[str, Any], regen_inputs: dict[str, Any], sim_inputs: dict[str, Any]) -> None:
    show_html(theme.section("Concept in 30 seconds", "How regenerative braking works",
                          "Four short animated steps. They play automatically; click a step to jump to it.", theme.DRIVE))
    regen_explainer()

    show_html(theme.section("Try it yourself", "Regen blending playground",
                          "Set up one braking moment. The real controller from the simulation decides how much the "
                          "motor can recover and how much the friction brakes must burn. Uses the strategy chosen "
                          "in the sidebar.", theme.VOLT))
    c1, c2, c3, c4 = st.columns(4)
    v = c1.slider("Vehicle speed (km/h)", 0, 130, 50, key="pg_v")
    p = c2.slider("Braking power requested (kW)", 1, 120, 30, key="pg_p",
                  help="Gentle stop ≈ 10–20 kW, firm stop ≈ 40–60 kW, emergency stop ≈ 100 kW+")
    soc = c3.slider("Battery level (%)", 0, 100, int(sim_inputs["initial_soc"]), key="pg_soc")
    temp = c4.slider("Battery temperature (°C)", -20, 50, int(sim_inputs["temp"]), key="pg_t")
    ctx, split = blend_preview(cfg, regen_inputs, v, p, soc, temp)
    fade = split.fade_factor
    limits = [("Motor generating limit", ctx.motor_limit_w / 1000.0),
              (f"Battery acceptance ({ctx.battery_reason.lower()})", ctx.battery_limit_w / 1000.0),
              ("Strategy limit", split.strategy_limit_w / 1000.0)]
    winner_name, _ = min(limits, key=lambda kv: kv[1])
    regen_kw, fric_kw = split.p_regen_w / 1000.0, split.p_friction_w / 1000.0
    reason_text = {
        "Full regen": "the motor can absorb everything, so <b>no energy is wasted</b>.",
        "Low-speed fade": f"at {v} km/h regen is faded to <b>{fade * 100:.0f} %</b>; the pads finish the stop.",
        "Motor limit": "the <b>motor</b> is the bottleneck: it can't generate that much braking power at this speed.",
        "Strategy limit": "the <b>strategy</b> chosen in the sidebar caps the regen share.",
        "SoC limit": "the battery is nearly <b>full</b> and tapers its charging power.",
        "Battery power limit": "the <b>battery</b> can't absorb charging power any faster.",
        "Temperature derate": "the battery is too <b>cold/hot</b> to charge quickly.",
        "Regen disabled": "regen is <b>switched off</b> by the selected level.",
    }.get(split.reason, split.reason)
    verdict = (f"Motor recovers <b>{regen_kw:.1f} kW</b> ({100 * regen_kw / p:.0f} %), friction brakes burn "
               f"<b>{fric_kw:.1f} kW</b>. Why: {reason_text}")
    show_html(theme.blend_bars_html(float(p), limits, winner_name if split.reason not in ("Full regen", "Low-speed fade")
                                  else "", regen_kw, fric_kw, verdict))


# =============================================================================
# Tab 3: SoC comparison
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
    st.plotly_chart(fig, width="stretch", theme=None)

    cum = make_subplots(specs=[[{"secondary_y": False}]])
    cum.add_trace(go.Scatter(x=regen.df["t_s"], y=regen.df["p_regen_batt_w"].cumsum() / 3600.0,
                             name="Energy recovered to battery (Wh)", line={"color": C_REGEN}))
    cum.add_trace(go.Scatter(x=regen.df["t_s"], y=regen.df["p_friction_w"].cumsum() / 3600.0,
                             name="Friction heat · regen case (Wh)", line={"color": theme.VOLT}))
    cum.add_trace(go.Scatter(x=fric.df["t_s"], y=fric.df["p_friction_w"].cumsum() / 3600.0,
                             name="Friction heat · friction-only (Wh)", line={"color": C_FRIC, "dash": "dash"}))
    cum.update_layout(title="Cumulative braking energy", height=380, yaxis_title="Wh", xaxis_title="Time (s)",
                      legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(cum, width="stretch", theme=None)
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
    colors = [theme.VOLT, "#4a5568", "#4a5568", C_SPEED, "#4a5568", "#1e6fa8", "#5b7a99", "#5b7a99",
              "#5b7a99", C_FRIC, "#1fae78", "#4a5568", C_REGEN, "#5b7a99", "#5b7a99"]
    fig = go.Figure(go.Sankey(
        valueformat=".0f", valuesuffix=" Wh",
        node={"label": nodes, "pad": 18, "thickness": 18, "color": colors, "line": {"width": 0}},
        link={"source": [ix[a] for a, _, _ in links], "target": [ix[b] for _, b, _ in links],
              "value": [v for _, _, v in links],
              "color": ["rgba(46,229,157,0.35)" if "Regen" in a or "Recovered" in b or "Regen" in b
                        else "rgba(255,93,93,0.35)" if "Friction" in b else "rgba(138,151,171,0.18)"
                        for a, b, _ in links]},
    ))
    fig.update_layout(height=520, margin={"t": 30, "b": 10})
    return fig


def tab_energy(sim: dict[str, Any], cfg: dict[str, Any]) -> go.Figure:
    er, ef = energy_totals(sim["regen"]), energy_totals(sim["friction"])
    show_html(theme.section("Energy breakdown", "Where does every watt-hour go?",
                          "Regenerative case, whole drive, in Wh. Green = recovered into the battery, "
                          "red = wasted as brake heat, grey = unavoidable losses.", C_REGEN))
    st.plotly_chart(sankey_figure(er), width="stretch", theme=None)

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
    st.plotly_chart(bars, width="stretch", theme=None)

    c1, c2 = st.columns(2)
    reasons = sim["regen"].df.query("p_brake_req_w > 0")
    lost = (reasons["p_brake_req_w"] - reasons["p_regen_wheel_w"]).groupby(reasons["regen_reason"]).sum() / 3600
    pie = go.Figure(go.Pie(labels=lost.index, values=lost.values, hole=0.45))
    pie.update_layout(title="Braking energy NOT recovered, by limiting reason (Wh)", height=380)
    c1.plotly_chart(pie, width="stretch", theme=None)

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
    c2.plotly_chart(emap, width="stretch", theme=None)
    return bars


# =============================================================================
# Tab 4: Braking events
# =============================================================================
def tab_events(sim: dict[str, Any]) -> None:
    ev = braking_events(sim["regen"])
    if ev.empty:
        st.info("No braking events in this cycle.")
        return
    show_html(theme.section("Braking events", f"{len(ev)} stops and slow-downs, one by one",
                          "Each row is one continuous braking event: how much motion energy was available, how much "
                          "came back, and which limit stopped regen from taking more.", C_FRIC))
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
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("Low-speed stops lose relatively more (fade + road loads eat the energy); very high SoC or cold "
               "batteries show 'SoC limit' / 'Temperature derate'.")


# =============================================================================
# Tab 5: OTA & twin sync
# =============================================================================
def tab_ota_twin(sim: dict[str, Any]) -> None:
    show_html(theme.section("Software-defined vehicle", "Over-the-air (OTA) calibration update",
                          "A new regen calibration arrives mid-drive. The pipeline below replays what the car's OTA "
                          "manager did with it.", C_OTA))
    ota = sim["ota"]
    if ota is None:
        st.info("Enable 'Run OTA update during the drive' in the sidebar.")
    else:
        res: SimulationResult = ota["result"]
        events = pd.DataFrame(ota["events"])
        show_html(theme.ota_pipeline_html(ota["events"]))
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
            colors = {"INSTALLED": C_OTA, "REJECTED": C_FRIC, "ROLLED_BACK": theme.VOLT, "HEALTH_OK": theme.MUTED}
            for _, e in events[events["kind"] != "SCHEDULED"].iterrows():
                fig.add_vline(x=e["t_s"], line_dash="dot", line_color=colors.get(e["kind"], "grey"),
                              annotation_text=f"{e['kind']} {e['version']}", annotation_textangle=-90)
            fig.update_layout(title="Before/after effect of the OTA update on SoC", height=420,
                              xaxis_title="Time (s)", yaxis_title="SoC (%)", legend={"orientation": "h", "y": -0.2})
            st.plotly_chart(fig, width="stretch", theme=None)
            dsoc = res.final_soc_pct - sim["regen"].final_soc_pct
            st.metric("Final SoC effect of OTA", f"{res.final_soc_pct:.2f} %", f"{dsoc:+.3f} pp vs no OTA")

        vers = res.df[["t_s", "regen_version"]]
        tl = go.Figure(go.Scatter(x=vers["t_s"], y=vers["regen_version"], mode="lines", line_shape="hv",
                                  line={"color": C_OTA, "width": 3}))
        tl.update_layout(title="Running calibration version (Vehicle.Software.RegenCalibration.Version)",
                         height=220, xaxis_title="Time (s)", margin={"t": 40, "b": 30})
        st.plotly_chart(tl, width="stretch", theme=None)

    st.divider()
    show_html(theme.section("Digital twin", "Twin vs 'physical' vehicle synchronisation",
                          "The twin only stays a twin if real measurements keep correcting it.", C_REGEN))
    twin: TwinSyncResult | None = sim["twin"]
    if twin is None:
        st.info("Enable 'Run twin vs physical vehicle' in the sidebar.")
        return
    m = twin.metrics
    show_html(theme.twin_sync_html(m, twin.sync_period_s, twin.method))
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
                             marker={"size": 4, "color": theme.MUTED}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t_s"], y=df["soc_true_pct"], name="Physical vehicle (truth)",
                             line={"color": theme.TEXT, "width": 2}), row=1, col=1)
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
    st.plotly_chart(fig, width="stretch", theme=None)
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
  rankdir=LR; bgcolor="transparent"; pad=0.2;
  node [shape=box, style="rounded,filled", fillcolor="#162031", color="#22304A", penwidth=1.5,
        fontname="Helvetica", fontcolor="#E7ECF3", fontsize=11];
  edge [color="#8A97AB", fontcolor="#8A97AB", fontname="Helvetica", fontsize=9, arrowsize=0.7];
  cycle [label="Drive cycle\\nv(t)", color="#38BDF8"]; dyn [label="Vehicle dynamics\\nF_roll+F_aero+F_grade+F_inertia", color="#38BDF8"];
  ctrl [label="Regen controller\\n(blending strategy)", color="#2EE59D", penwidth=2.5];
  motor [label="Motor + inverter\\nη(T, ω), limits", color="#38BDF8"]; fric [label="Friction brakes\\n(heat)", color="#FF5D5D"];
  batt [label="Battery 1-RC ECM\\nOCV, R0, R1‖C1, SoC", color="#FFB020"];
  bus [label="VSS signal bus\\n(pub/sub)", color="#FFB020"]; ota [label="OTA manager\\nvalidate / rollback", color="#B18CFF"];
  twin [label="Twin sync\\nKalman blend", color="#B18CFF"]; phys [label="Physical vehicle\\n(sensors + noise)", shape=component, color="#B18CFF"];
  dash [label="Streamlit dashboard", shape=note, color="#8A97AB"];
  cycle -> dyn -> ctrl [label="P_wheel"]; ctrl -> motor [label="P_regen / traction", color="#2EE59D"];
  ctrl -> fric [label="P_friction", color="#FF5D5D"];
  motor -> batt [label="P_batt", color="#2EE59D"]; batt -> ctrl [label="charge limit", style=dashed];
  batt -> bus; motor -> bus; ctrl -> bus; bus -> dash; ota -> ctrl [label="new calibration", color="#B18CFF"];
  phys -> twin [label="measured SoC"]; twin -> batt [label="SoC correction", style=dashed]; twin -> dash;
}
"""


def tab_theory() -> None:
    show_html(theme.section("Theory", "System block diagram and equations", "How the modules connect, and the "
                          "physics each one solves.", theme.DRIVE))
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

    # CSS travels with the hero block: a style-only st.html goes to a hidden container and is dropped
    show_html(theme.GLOBAL_CSS + theme.hero_html(cycle_name, stats, STRATEGY_LABELS[regen_inputs["strategy"]], cfg["vehicle"]["name"]))
    show_html(theme.how_to_read_html())
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

    tabs = st.tabs(["🚗 Live Twin", "🎓 How Regen Works", "📉 SoC Comparison", "⚡ Energy Breakdown",
                    "🛑 Braking Events", "📡 OTA & Twin Sync", "📘 Theory"])
    with tabs[0]:
        tab_live(sim, cfg, kr)
    with tabs[1]:
        tab_how(cfg, regen_inputs, sim_inputs)
    with tabs[2]:
        tab_soc(sim)
    with tabs[3]:
        tab_energy(sim, cfg)
    with tabs[4]:
        tab_events(sim)
    with tabs[5]:
        tab_ota_twin(sim)
    with tabs[6]:
        tab_theory()

if __name__ == "__main__":
    main()
