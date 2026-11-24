"""Visual identity of the dashboard: colour palette, global CSS, Plotly template and HTML widgets.

The look is an "EV instrument cluster": graphite background, amber accent for controls,
green for recovered energy, red for brake heat and sky-blue for driving power.
Every widget here is plain HTML + CSS (no JavaScript) so it can be rendered with st.html.
"""

from __future__ import annotations

import base64
import html
import math
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------- palette
INK = "#0A0E14"
PANEL = "#111824"
PANEL_2 = "#162031"
LINE = "#22304A"
TEXT = "#E7ECF3"
MUTED = "#8A97AB"
REGEN = "#2EE59D"      # recovered energy / regenerative case
FRIC = "#FF5D5D"       # friction brake heat / friction-only baseline
DRIVE = "#38BDF8"      # traction power, speed
VOLT = "#FFB020"       # accent, state of charge
OTA = "#B18CFF"        # software / OTA
BRAKE_SHADE = "rgba(255, 176, 32, 0.16)"

FONT_BODY = "'IBM Plex Sans', system-ui, sans-serif"
FONT_HEAD = "'Space Grotesk', 'IBM Plex Sans', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"


# ---------------------------------------------------------------- Plotly template
def _register_plotly_template() -> None:
    axis = {"gridcolor": "#1C2738", "zerolinecolor": "#2A3A55", "linecolor": LINE,
            "tickfont": {"color": MUTED}, "title": {"font": {"color": MUTED}}}
    pio.templates["evtwin"] = go.layout.Template(layout={
        "font": {"family": FONT_BODY, "color": TEXT, "size": 13},
        "title": {"font": {"family": FONT_HEAD, "size": 17, "color": TEXT}, "x": 0.01},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(17,24,36,0.55)",
        "colorway": [REGEN, FRIC, DRIVE, VOLT, OTA, MUTED],
        "xaxis": axis, "yaxis": axis,
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT}},
        "hoverlabel": {"bgcolor": PANEL_2, "bordercolor": LINE, "font": {"family": FONT_MONO, "color": TEXT}},
        "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
    })
    pio.templates.default = "plotly_dark+evtwin"


_register_plotly_template()


# ---------------------------------------------------------------- global CSS
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{
  --ink:{INK}; --panel:{PANEL}; --panel2:{PANEL_2}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
  --regen:{REGEN}; --fric:{FRIC}; --drive:{DRIVE}; --volt:{VOLT}; --ota:{OTA};
  --mono:{FONT_MONO}; --head:{FONT_HEAD};
}}
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(46,229,157,0.07), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(56,189,248,0.06), transparent 60%),
    linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px) 0 0 / 100% 32px,
    var(--ink);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] h2 {{
  font-size: 0.78rem !important; letter-spacing: .14em; text-transform: uppercase; color: var(--volt) !important;
  border-top: 1px solid var(--line); padding-top: 1rem !important; margin-top: .4rem;
}}
h1, h2, h3, h4, h5 {{ font-family: var(--head) !important; letter-spacing: -.01em; }}

/* tabs as a segmented control */
[data-baseweb="tab-list"] {{
  gap: 4px; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 5px;
  overflow-x: auto;
}}
button[data-baseweb="tab"] {{
  border-radius: 10px !important; padding: 8px 14px !important; height: auto !important;
  transition: background .25s, color .25s;
}}
button[data-baseweb="tab"][aria-selected="true"] {{ background: var(--panel2) !important; color: var(--volt) !important; }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none; }}

/* metrics as instrument cards */
[data-testid="stMetric"] {{
  background: linear-gradient(180deg, var(--panel2), var(--panel)); border: 1px solid var(--line);
  border-radius: 14px; padding: 14px 16px; animation: ev-rise .6s ease-out both;
}}
[data-testid="stMetricValue"] {{ font-family: var(--mono) !important; }}
[data-testid="stExpander"] details {{ border-radius: 14px !important; background: var(--panel); }}
[data-testid="stPlotlyChart"] {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 6px 8px;
  animation: ev-rise .7s ease-out both;
}}

@keyframes ev-rise {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes ev-grow {{ from {{ width: 0; }} }}
@keyframes ev-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes ev-road {{ to {{ background-position: -120px 0; }} }}
@keyframes ev-bob {{ 50% {{ transform: translateY(-1.5px); }} }}
@keyframes ev-pulse {{ 0% {{ opacity: .9; transform: scale(.6); }} 100% {{ opacity: 0; transform: scale(1.8); }} }}
@keyframes ev-blink {{ 50% {{ opacity: .35; }} }}
@keyframes ev-flow-r {{ from {{ left: -6%; opacity: 0; }} 15% {{ opacity: 1; }} 85% {{ opacity: 1; }} to {{ left: 100%; opacity: 0; }} }}
@keyframes ev-flow-l {{ from {{ left: 100%; opacity: 0; }} 15% {{ opacity: 1; }} 85% {{ opacity: 1; }} to {{ left: -6%; opacity: 0; }} }}
@keyframes ev-shake {{ 20%, 60% {{ transform: translateX(-3px); }} 40%, 80% {{ transform: translateX(3px); }} }}
@keyframes ev-ring {{ from {{ transform: rotate(-90deg) scale(.4); opacity: 0; }} }}

/* hero */
.ev-hero {{
  position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 20px;
  background: linear-gradient(120deg, #101a2a 0%, #0c121c 55%, #0f1d1a 100%); padding: 22px 26px 18px;
  animation: ev-rise .6s ease-out both;
}}
.ev-hero .kicker {{ font: 600 .72rem var(--mono); letter-spacing: .18em; color: var(--volt); text-transform: uppercase; }}
.ev-hero h1 {{ font: 700 2.1rem/1.1 var(--head); margin: 6px 0 6px; color: var(--text); padding: 0; }}
.ev-hero h1 em {{ font-style: normal; color: var(--regen); }}
.ev-hero p {{ color: var(--muted); margin: 0 0 12px; max-width: 640px; }}
.ev-chips {{ display: flex; flex-wrap: wrap; gap: 6px; position: relative; z-index: 2; }}
.ev-chip {{
  font: 500 .78rem var(--mono); color: var(--text); background: rgba(255,255,255,0.04);
  border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px;
}}
.ev-chip b {{ color: var(--volt); font-weight: 600; }}
.ev-hero-car {{ position: absolute; right: 26px; top: 18px; width: 300px; height: 120px; }}
.ev-hero-car .road {{
  position: absolute; left: 0; right: 0; bottom: 8px; height: 4px; border-radius: 2px;
  background: repeating-linear-gradient(90deg, rgba(231,236,243,.45) 0 40px, transparent 40px 120px);
  animation: ev-road .7s linear infinite;
}}
.ev-hero-car img {{ position: absolute; left: 50px; bottom: 12px; width: 200px; animation: ev-bob .5s ease-in-out infinite; }}
.ev-hero-car .pulse {{
  position: absolute; left: 118px; bottom: 22px; width: 64px; height: 64px; border-radius: 50%;
  border: 2px solid var(--regen); animation: ev-pulse 1.8s ease-out infinite;
}}
@media (max-width: 900px) {{ .ev-hero-car {{ display: none; }} .ev-hero h1 {{ font-size: 1.6rem; }} }}

/* KPI cards */
.ev-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 14px 0 6px; }}
.ev-kpi {{
  background: linear-gradient(180deg, var(--panel2), var(--panel)); border: 1px solid var(--line);
  border-radius: 16px; padding: 14px 16px 12px; position: relative; overflow: hidden;
  animation: ev-rise .6s ease-out both; animation-delay: calc(var(--i) * 80ms);
}}
.ev-kpi::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: var(--c); }}
.ev-kpi.ring .lbl {{ padding-right: 64px; }}
.ev-kpi .lbl {{ font: 600 .7rem var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }}
.ev-kpi .val {{ font: 600 1.75rem/1.2 var(--mono); color: var(--text); margin-top: 4px; }}
.ev-kpi .val small {{ font-size: .9rem; color: var(--muted); margin-left: 3px; }}
.ev-kpi .sub {{ font-size: .8rem; color: var(--muted); margin-top: 2px; }}
.ev-kpi .sub b {{ color: var(--c); font-weight: 600; }}
.ev-cmp {{ margin-top: 10px; display: grid; gap: 5px; }}
.ev-cmp-row {{ display: grid; grid-template-columns: 58px 1fr 52px; align-items: center; gap: 8px; font: .72rem var(--mono); color: var(--muted); }}
.ev-cmp-row b {{ color: var(--text); font-weight: 500; text-align: right; }}
.ev-bar {{ height: 7px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; }}
.ev-bar i {{ display: block; height: 100%; width: var(--w); background: var(--bc); border-radius: 4px; animation: ev-grow 1.3s cubic-bezier(.2,.8,.2,1) both; animation-delay: calc(var(--i) * 80ms + 200ms); }}
.ev-ring {{
  position: absolute; right: 14px; top: 14px; width: 54px; height: 54px; border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,0.07) 0);
  animation: ev-ring 1.6s cubic-bezier(.2,.8,.2,1) both;
  display: grid; place-items: center;
}}
.ev-ring::after {{ content: ""; position: absolute; inset: 6px; border-radius: 50%; background: var(--panel); }}
.ev-ring span {{ position: relative; z-index: 1; font: 600 .7rem var(--mono); color: var(--text); }}

/* section headings */
.ev-sec {{ margin: 8px 0 10px; animation: ev-rise .5s ease-out both; }}
.ev-sec .k {{ font: 600 .7rem var(--mono); letter-spacing: .16em; text-transform: uppercase; color: var(--c, var(--volt)); }}
.ev-sec h3 {{ font: 600 1.35rem var(--head); margin: 2px 0 2px; padding: 0; color: var(--text); }}
.ev-sec p {{ color: var(--muted); margin: 0; font-size: .92rem; }}

/* how-to-read strip */
.ev-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 6px 0 4px; }}
.ev-step {{ display: flex; gap: 10px; align-items: flex-start; background: var(--panel); border: 1px dashed var(--line); border-radius: 14px; padding: 10px 12px; animation: ev-rise .6s ease-out both; animation-delay: calc(var(--i) * 90ms); }}
.ev-step .n {{ flex: none; width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center; font: 600 .8rem var(--mono); background: var(--c); color: var(--ink); }}
.ev-step div {{ font-size: .82rem; color: var(--muted); }}
.ev-step b {{ color: var(--text); display: block; font-weight: 600; }}

/* OTA pipeline */
.ev-pipe {{ display: flex; align-items: stretch; gap: 0; overflow-x: auto; padding: 8px 2px 4px; }}
.ev-node {{
  flex: 1 1 0; min-width: 130px; background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 10px 12px; opacity: .35; animation: ev-on .45s ease-out forwards; animation-delay: calc(var(--i) * 550ms);
}}
.ev-node .ic {{ font-size: 1.2rem; }}
.ev-node b {{ display: block; font: 600 .85rem var(--head); color: var(--text); margin-top: 2px; }}
.ev-node span {{ font-size: .74rem; color: var(--muted); }}
.ev-node.ok {{ --st: var(--regen); }} .ev-node.bad {{ --st: var(--fric); }} .ev-node.warn {{ --st: var(--volt); }} .ev-node.skip {{ --st: var(--line); }}
.ev-node.bad {{ animation: ev-on .45s ease-out forwards, ev-shake .5s ease-in-out; animation-delay: calc(var(--i) * 550ms), calc(var(--i) * 550ms + 450ms); }}
.ev-node.skip {{ animation: none; opacity: .3; }}
.ev-node .st {{ font: 600 .7rem var(--mono); color: var(--st); letter-spacing: .08em; }}
@keyframes ev-on {{ to {{ opacity: 1; border-color: var(--st); box-shadow: 0 0 0 1px var(--st) inset, 0 6px 24px -10px var(--st); }} }}
.ev-link {{ flex: 0 0 34px; align-self: center; height: 3px; background: var(--line); position: relative; overflow: hidden; }}
.ev-link i {{ position: absolute; inset: 0; background: var(--st, var(--regen)); transform-origin: left; transform: scaleX(0); animation: ev-fill .5s linear forwards; animation-delay: calc(var(--i) * 550ms - 150ms); }}
.ev-link.skip i {{ display: none; }}
@keyframes ev-fill {{ to {{ transform: scaleX(1); }} }}

/* twin sync */
.ev-twin {{ display: grid; grid-template-columns: 1fr 1.3fr 1fr; gap: 10px; align-items: center; margin: 6px 0 12px; }}
.ev-twin-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px; text-align: center; }}
.ev-twin-card .ic {{ font-size: 1.8rem; }}
.ev-twin-card b {{ display: block; font: 600 .95rem var(--head); }}
.ev-twin-card span {{ font-size: .78rem; color: var(--muted); }}
.ev-chan {{ position: relative; height: 74px; }}
.ev-chan .lane {{ position: absolute; left: 0; right: 0; height: 2px; background: var(--line); }}
.ev-chan .lane.a {{ top: 22px; }} .ev-chan .lane.b {{ top: 52px; }}
.ev-chan .pk {{ position: absolute; width: 10px; height: 10px; border-radius: 50%; margin-top: -4px; }}
.ev-chan .pk.a {{ top: 22px; background: var(--drive); box-shadow: 0 0 10px var(--drive); animation: ev-flow-r 2.4s linear infinite; animation-delay: calc(var(--i) * 0.8s); }}
.ev-chan .pk.b {{ top: 52px; background: var(--regen); box-shadow: 0 0 10px var(--regen); animation: ev-flow-l 2.4s linear infinite; animation-delay: calc(var(--i) * 0.8s + .4s); }}
.ev-chan .t {{ position: absolute; left: 0; right: 0; text-align: center; font: .7rem var(--mono); color: var(--muted); }}
.ev-chan .t.a {{ top: 0; }} .ev-chan .t.b {{ top: 58px; }}
@media (max-width: 700px) {{ .ev-twin {{ grid-template-columns: 1fr; }} .ev-chan {{ height: 80px; }} }}

/* regen blend playground bars */
.ev-limits {{ display: grid; gap: 8px; margin-top: 6px; }}
.ev-lim {{ display: grid; grid-template-columns: 170px 1fr 80px; gap: 10px; align-items: center; font-size: .84rem; }}
.ev-lim .nm {{ color: var(--muted); }}
.ev-lim.win .nm {{ color: var(--volt); font-weight: 600; }}
.ev-lim .v {{ font-family: var(--mono); text-align: right; }}
.ev-lim .ev-bar {{ height: 12px; }}
.ev-lim.win .ev-bar {{ box-shadow: 0 0 0 1px var(--volt); }}
.ev-split {{ display: flex; height: 34px; border-radius: 10px; overflow: hidden; border: 1px solid var(--line); margin-top: 6px; }}
.ev-split div {{ display: grid; place-items: center; font: 600 .78rem var(--mono); color: var(--ink); white-space: nowrap; overflow: hidden; animation: ev-grow 1s cubic-bezier(.2,.8,.2,1) both; }}
.ev-verdict {{ margin-top: 10px; padding: 10px 14px; border-radius: 12px; border: 1px solid var(--line); background: var(--panel); font-size: .9rem; animation: ev-rise .5s ease-out both; }}
.ev-verdict b {{ color: var(--volt); }}
</style>
"""


# ---------------------------------------------------------------- helpers
def _e(text: Any) -> str:
    return html.escape(str(text))


def section(kicker: str, title: str, text: str = "", color: str = VOLT) -> str:
    body = f"<p>{text}</p>" if text else ""
    return f'<div class="ev-sec" style="--c:{color}"><div class="k">{_e(kicker)}</div><h3>{_e(title)}</h3>{body}</div>'


HERO_CAR_SVG = f"""<svg viewBox="0 0 200 70" xmlns="http://www.w3.org/2000/svg">
  <style>.w{{transform-box:fill-box;transform-origin:center;animation:s .45s linear infinite}}
    @keyframes s{{to{{transform:rotate(360deg)}}}}
    .b{{animation:g 1.6s ease-in-out infinite}} @keyframes g{{50%{{opacity:.35}}}}</style>
  <path d="M8 50 L14 34 Q18 28 28 27 L62 24 Q76 12 100 11 L134 12 Q150 14 162 26 L184 30 Q194 32 194 42 L194 50 Z"
        fill="#1b2a40" stroke="{DRIVE}" stroke-width="1.5"/>
  <path d="M70 25 Q82 15 100 15 L128 15 Q142 17 150 26 Z" fill="#0d1624" stroke="#2c4060"/>
  <rect class="b" x="44" y="44" width="110" height="5" rx="2" fill="{REGEN}"/>
  <rect x="186" y="34" width="6" height="4" rx="1" fill="{VOLT}"/>
  <rect x="9" y="36" width="5" height="5" rx="1" fill="{FRIC}"/>
  <g class="w"><circle cx="48" cy="52" r="12" fill="#0a0e14" stroke="#c9d3e0" stroke-width="2.5"/>
    <path d="M48 42 V62 M38 52 H58 M41 45 L55 59 M55 45 L41 59" stroke="#8A97AB" stroke-width="1.5"/></g>
  <g class="w"><circle cx="156" cy="52" r="12" fill="#0a0e14" stroke="#c9d3e0" stroke-width="2.5"/>
    <path d="M156 42 V62 M146 52 H166 M149 45 L163 59 M163 45 L149 59" stroke="#8A97AB" stroke-width="1.5"/></g>
</svg>"""
HERO_CAR_IMG = "data:image/svg+xml;base64," + base64.b64encode(HERO_CAR_SVG.encode("utf-8")).decode("ascii")


def hero_html(cycle_name: str, stats: dict[str, float], strategy_label: str, vehicle: str) -> str:
    chips = [
        ("cycle", cycle_name), ("distance", f"{stats['distance_km']:.2f} km"), ("duration", f"{stats['duration_s']:.0f} s"),
        ("top speed", f"{stats['max_speed_kmh']:.0f} km/h"), ("avg", f"{stats['mean_speed_kmh']:.1f} km/h"),
        ("strategy", strategy_label), ("vehicle", vehicle),
    ]
    chip_html = "".join(f'<span class="ev-chip">{_e(k)} <b>{_e(v)}</b></span>' for k, v in chips)
    return f"""
<div class="ev-hero">
  <div class="ev-hero-car"><div class="pulse"></div><img src="{HERO_CAR_IMG}" alt="EV"><div class="road"></div></div>
  <div class="kicker">Software-Defined Vehicle · Digital Twin</div>
  <h1>Every stop can <em>recharge</em> the battery.</h1>
  <p>A physics-based twin of a compact EV driving a standard test cycle twice: once braking with the motor
     (energy goes back into the battery) and once with friction brakes only (energy becomes heat).</p>
  <div class="ev-chips">{chip_html}</div>
</div>"""


def how_to_read_html() -> str:
    steps = [
        (DRIVE, "Pick a drive cycle", "Sidebar → choose WLTP, city traffic, etc. and the starting battery level."),
        (REGEN, "Watch the twin drive", "Live Twin tab → press Play and follow the energy flowing in and out."),
        (VOLT, "Compare the outcome", "Cards below show battery left, energy saved and extra range."),
        (OTA, "Push a software update", "OTA tab → see the car's braking logic change mid-drive."),
    ]
    return '<div class="ev-steps">' + "".join(
        f'<div class="ev-step" style="--i:{i};--c:{c}"><span class="n">{i + 1}</span><div><b>{_e(t)}</b>{_e(d)}</div></div>'
        for i, (c, t, d) in enumerate(steps)) + "</div>"


# ---------------------------------------------------------------- KPI cards
def _cmp_rows(i: int, a: float, b: float, fmt: str, scale: float | None = None) -> str:
    top = scale if scale is not None else max(abs(a), abs(b), 1e-9)
    rows = [("Regen", a, REGEN), ("Friction", b, FRIC)]
    return '<div class="ev-cmp">' + "".join(
        f'<div class="ev-cmp-row"><span>{name}</span><div class="ev-bar"><i style="--w:{max(0.0, min(100.0, 100 * v / top)):.1f}%;'
        f'--bc:{c};--i:{i}"></i></div><b>{format(v, fmt)}</b></div>' for name, v, c in rows) + "</div>"


def kpi_grid_html(kr: dict[str, float], kf: dict[str, float], cmp: dict[str, float]) -> str:
    free_km = kr["kwh_recovered"] * 1000.0 / kr["wh_per_km"] if kr["wh_per_km"] > 0 else 0.0
    rng_r = kr["range_km"] if math.isfinite(kr["range_km"]) else 0.0
    rng_f = kf["range_km"] if math.isfinite(kf["range_km"]) else 0.0
    cards = [
        (VOLT, "Battery left at the end", f"{kr['final_soc_pct']:.2f}", "%",
         f"<b>{cmp['soc_saved_pct']:+.2f} pp</b> more than friction-only",
         _cmp_rows(0, kr["final_soc_pct"], kf["final_soc_pct"], ".2f", 100.0), ""),
        (DRIVE, "Energy used from battery", f"{kr['kwh_consumed_net']:.3f}", "kWh",
         f"<b>{cmp['energy_saved_kwh']:.3f} kWh</b> saved by regen",
         _cmp_rows(1, kr["kwh_consumed_net"], kf["kwh_consumed_net"], ".3f"), ""),
        (REGEN, "Energy recovered by braking", f"{kr['kwh_recovered']:.3f}", "kWh",
         f"≈ <b>{free_km:.1f} km</b> of free driving",
         "", f'<div class="ev-ring" style="--p:{kr["recovery_eff_pct"]:.1f};--c:{REGEN}"><span>{kr["recovery_eff_pct"]:.0f}%</span></div>'),
        (DRIVE, "Consumption", f"{kr['wh_per_km']:.1f}", "Wh/km",
         f"<b>{kr['wh_per_km'] - kf['wh_per_km']:+.1f} Wh/km</b> vs friction-only",
         _cmp_rows(3, kr["wh_per_km"], kf["wh_per_km"], ".1f"), ""),
        (REGEN, "Estimated range", f"{rng_r:.0f}", "km",
         f"<b>{cmp['range_gain_pct']:+.1f} %</b> range gain",
         _cmp_rows(4, rng_r, rng_f, ".0f"), ""),
        (FRIC, "Wasted as brake heat", f"{kr['kwh_friction']:.3f}", "kWh",
         f"<b>{cmp['friction_saved_pct']:.0f} % less</b> pad wear (proxy)",
         _cmp_rows(5, kr["kwh_friction"], kf["kwh_friction"], ".3f"), ""),
    ]
    out = []
    for i, (c, lbl, val, unit, sub, bars, ring) in enumerate(cards):
        out.append(f'<div class="ev-kpi{' ring' if ring else ''}" style="--c:{c};--i:{i}">{ring}<div class="lbl">{_e(lbl)}</div>'
                   f'<div class="val">{val}<small>{unit}</small></div><div class="sub">{sub}</div>{bars}</div>')
    return '<div class="ev-kpis">' + "".join(out) + "</div>"


# ---------------------------------------------------------------- OTA pipeline
def ota_pipeline_html(events: list[dict[str, Any]]) -> str:
    kinds = {e["kind"]: e for e in events}
    steps = [["📦", "Package received", "signed calibration bundle"],
             ["🔐", "Checksum & version", "SHA-256 + no downgrade"],
             ["🛡️", "Safety validation", "whitelist + safe ranges"],
             ["⚙️", "Install", "hot-swap regen params"],
             ["🩺", "Health monitor", "30 s watch, auto-rollback"]]
    status = ["ok", "ok", "ok", "ok", "ok"]
    notes = ["DONE", "PASSED", "PASSED", "INSTALLED", "HEALTHY"]
    if "REJECTED" in kinds:
        msg = kinds["REJECTED"]["detail"].lower()
        fail_at = 1 if ("checksum" in msg or "version" in msg or "downgrade" in msg) else 2
        status = ["ok"] * fail_at + ["bad"] + ["skip"] * (4 - fail_at)
        notes = ["DONE", "PASSED", "PASSED"][:fail_at] + ["REJECTED"] + ["—"] * (4 - fail_at)
        steps[fail_at][2] = kinds["REJECTED"]["detail"][:90]
    elif "ROLLED_BACK" in kinds:
        status[4], notes[4] = "warn", "ROLLED BACK"
        steps[4][2] = f"t = {kinds['ROLLED_BACK']['t_s']:.0f} s · {kinds['ROLLED_BACK']['detail'][:70]}"
    elif "HEALTH_OK" not in kinds:
        status[4], notes[4] = "warn", "WATCHING"
    if "INSTALLED" in kinds:
        steps[3][2] = f"v{kinds['INSTALLED']['version']} at t = {kinds['INSTALLED']['t_s']:.0f} s"
    parts = []
    for i, ((ic, title, desc), st_, note) in enumerate(zip(steps, status, notes)):
        if i:
            parts.append(f'<div class="ev-link {st_ if st_ == "skip" else ""}" style="--i:{i};--st:var(--{ "regen" if st_ != "bad" else "fric"})"><i></i></div>')
        parts.append(f'<div class="ev-node {st_}" style="--i:{i}"><div class="ic">{ic}</div><b>{_e(title)}</b>'
                     f'<div class="st">{note}</div><span>{_e(desc)}</span></div>')
    return '<div class="ev-pipe">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------- twin sync
def twin_sync_html(metrics: dict[str, float], sync_period_s: int, method: str) -> str:
    packets_a = "".join(f'<span class="pk a" style="--i:{i}"></span>' for i in range(3))
    packets_b = "".join(f'<span class="pk b" style="--i:{i}"></span>' for i in range(3))
    label = "Kalman blend" if method == "kalman" else "hard re-sync"
    return f"""
<div class="ev-twin">
  <div class="ev-twin-card"><div class="ic">🚗</div><b>Physical vehicle</b>
    <span>hidden +5 % mass, +10 % battery resistance, noisy sensors</span></div>
  <div class="ev-chan">
    <div class="t a">measured SoC, current, voltage → every {sync_period_s} s</div>
    <div class="lane a"></div>{packets_a}
    <div class="lane b"></div>{packets_b}
    <div class="t b">← {label} correction (RMSE {metrics['rmse_open_pct']:.2f} → {metrics['rmse_corrected_pct']:.2f} %)</div>
  </div>
  <div class="ev-twin-card"><div class="ic">🖥️</div><b>Digital twin</b>
    <span>nominal model, starts with a SoC error, corrected by feedback</span></div>
</div>"""


# ---------------------------------------------------------------- regen blend playground
def blend_bars_html(request_kw: float, limits: list[tuple[str, float]], winner: str, regen_kw: float,
                    friction_kw: float, verdict: str) -> str:
    top = max([request_kw] + [v for _, v in limits if math.isfinite(v)] + [1.0])
    rows = [("Brake request (driver)", request_kw, DRIVE, False)] + [(n, v, VOLT if n == winner else MUTED, n == winner)
                                                                       for n, v in limits]
    bars = "".join(
        f'<div class="ev-lim {"win" if win else ""}"><span class="nm">{_e(n)}</span><div class="ev-bar">'
        f'<i style="--w:{100 * min(v, top) / top:.1f}%;--bc:{c};--i:{i}"></i></div><span class="v">{v:.1f} kW</span></div>'
        for i, (n, v, c, win) in enumerate(rows))
    total = max(regen_kw + friction_kw, 1e-9)
    split = (f'<div class="ev-split"><div style="width:{100 * regen_kw / total:.1f}%;background:{REGEN}">'
             f'{"♻ regen " + format(regen_kw, ".1f") + " kW" if regen_kw / total > 0.18 else ""}</div>'
             f'<div style="width:{100 * friction_kw / total:.1f}%;background:{FRIC}">'
             f'{"🔥 friction " + format(friction_kw, ".1f") + " kW" if friction_kw / total > 0.18 else ""}</div></div>')
    return f'<div class="ev-limits">{bars}</div>{split}<div class="ev-verdict">{verdict}</div>'
