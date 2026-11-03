"""Command-line runner: prints KPIs for regen vs friction-only braking + sanity check.

Examples
--------
    python run_cli.py                       # WLTP, default settings
    python run_cli.py --cycle MIDC --strategy levels --level Medium
    python run_cli.py --cycle UDDS --soc 100 --temp 0
    python run_cli.py --all                 # every built-in cycle
"""

from __future__ import annotations

import argparse

from core import load_config
from core.analysis import compare_kpis, compute_kpis, format_sanity, sanity_check
from core.drive_cycles import BUILTIN_CYCLES, cycle_stats, load_cycle, repeat_cycle
from core.regen_controller import REGEN_LEVELS
from core.simulator import run_case


def run_one(cycle_key: str, args: argparse.Namespace) -> None:
    cfg = load_config()
    cycle = repeat_cycle(load_cycle(cycle_key), args.repeats)
    stats = cycle_stats(cycle)
    common = dict(initial_soc_pct=args.soc, ambient_temp_c=args.temp, road_grade_pct=args.grade,
                  payload_kg=args.payload)
    overrides = {"regen_level": args.level, "parallel_ratio": args.ratio}
    regen = run_case(cfg, cycle, args.strategy, regen_overrides=overrides, **common)
    fric = run_case(cfg, cycle, "friction_only", **common)
    kr, kf = compute_kpis(regen), compute_kpis(fric)
    cmp = compare_kpis(kr, kf)

    info = BUILTIN_CYCLES[cycle_key]
    print("=" * 78)
    print(f"Cycle : {info.label} x{args.repeats}  ({'exact' if info.exact else 'approximation'})")
    print(f"        {stats['distance_km']:.2f} km, {stats['duration_s']:.0f} s, max {stats['max_speed_kmh']:.1f} km/h,"
          f" mean {stats['mean_speed_kmh']:.1f} km/h")
    print(f"Setup : strategy={args.strategy}, initial SoC={args.soc:.0f} %, T={args.temp:.0f} C, "
          f"grade={args.grade:.1f} %, payload={args.payload:.0f} kg")
    print("-" * 78)
    print(f"{'KPI':38s}{'Regen':>14s}{'Friction-only':>16s}")
    rows = [
        ("Final SoC (%)", "final_soc_pct", "{:.2f}"),
        ("SoC used (percentage points)", "delta_soc_pct", "{:.2f}"),
        ("Net battery energy consumed (kWh)", "kwh_consumed_net", "{:.3f}"),
        ("Energy recovered by regen (kWh)", "kwh_recovered", "{:.3f}"),
        ("Friction-brake heat (kWh)", "kwh_friction", "{:.3f}"),
        ("Recovery efficiency (% of braking)", "recovery_eff_pct", "{:.1f}"),
        ("Regen share of traction energy (%)", "regen_share_pct", "{:.1f}"),
        ("Consumption (Wh/km)", "wh_per_km", "{:.1f}"),
        ("Estimated range (km)", "range_km", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:38s}{fmt.format(kr[key]):>14s}{fmt.format(kf[key]):>16s}")
    print("-" * 78)
    print(f"Range gain with regen          : {cmp['range_gain_pct']:+.1f} %")
    print(f"SoC saved                      : {cmp['soc_saved_pct']:+.2f} percentage points")
    print(f"Friction-brake energy saved    : {cmp['friction_saved_kwh']:.3f} kWh "
          f"({cmp['friction_saved_pct']:.0f} % less pad wear, proxy)")
    print(format_sanity(sanity_check(kr, stats["mean_speed_kmh"], cfg, args.strategy)))


def main() -> None:
    p = argparse.ArgumentParser(description="EV regen digital twin - command-line runner")
    p.add_argument("--cycle", default="WLTP", choices=list(BUILTIN_CYCLES))
    p.add_argument("--all", action="store_true", help="run every built-in cycle")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--strategy", default="series", choices=["series", "parallel", "levels"])
    p.add_argument("--level", default="High", choices=list(REGEN_LEVELS))
    p.add_argument("--ratio", type=float, default=0.3, help="parallel regen share")
    p.add_argument("--soc", type=float, default=90.0)
    p.add_argument("--temp", type=float, default=25.0)
    p.add_argument("--grade", type=float, default=0.0)
    p.add_argument("--payload", type=float, default=0.0)
    args = p.parse_args()
    for key in (BUILTIN_CYCLES if args.all else [args.cycle]):
        run_one(key, args)


if __name__ == "__main__":
    main()
