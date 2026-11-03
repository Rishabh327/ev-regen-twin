"""Physics tests: energy conservation, SoC bounds, regen limits, regen vs friction-only."""

from __future__ import annotations

import pytest

from core.analysis import compute_kpis, energy_totals
from core.battery_model import BatteryModel, BatteryParams
from core.drive_cycles import BUILTIN_CYCLES, cycle_stats, load_cycle, repeat_cycle
from core.regen_controller import BrakeContext, RegenController, RegenParams
from core.simulator import Simulator, run_case


# ---------------------------------------------------------------- energy conservation
def test_wheel_energy_balance(cfg, wltp):
    """Sum(P_wheel * dt) = E_aero + E_roll + E_grade + dKE  (exact by construction)."""
    e = energy_totals(run_case(cfg, wltp, "series"))
    lhs = e["traction_wheel_wh"] - e["brake_request_wh"]
    rhs = e["aero_wh"] + e["rolling_wh"] + e["grade_wh"] + e["kinetic_change_wh"]
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-6)


def test_brake_split_conserves_energy(cfg, wltp):
    """Every watt of braking request goes either to regen or to friction."""
    e = energy_totals(run_case(cfg, wltp, "series"))
    assert e["regen_wheel_wh"] + e["friction_wh"] == pytest.approx(e["brake_request_wh"], rel=1e-9)


def test_battery_energy_balance(cfg, wltp):
    """Chemical energy (Sum OCV*I*dt) = terminal energy + internal losses."""
    e = energy_totals(run_case(cfg, wltp, "series"))
    assert e["batt_chem_net_wh"] == pytest.approx(e["batt_net_wh"] + e["batt_loss_wh"], rel=1e-9)


def test_soc_change_matches_energy(cfg, wltp):
    """Charge removed (dSoC x Ah) x mean OCV ~ chemical energy drawn (within 3 %)."""
    res = run_case(cfg, wltp, "series")
    e = energy_totals(res)
    charge_ah = (res.initial_soc_pct - res.final_soc_pct) / 100.0 * res.battery.capacity_ah
    soc_energy_wh = charge_ah * res.df["ocv_v"].mean()
    assert soc_energy_wh == pytest.approx(e["batt_chem_net_wh"], rel=0.03)


def test_conversion_losses_are_positive(cfg, wltp):
    """Motor input > wheel output, and regen at the battery < regen at the wheel (efficiency < 100 %)."""
    e = energy_totals(run_case(cfg, wltp, "series"))
    assert e["motor_in_wh"] > e["traction_wheel_wh"]
    assert e["regen_batt_wh"] < e["regen_wheel_wh"]


# ---------------------------------------------------------------- SoC bounds
@pytest.mark.parametrize("start_soc", [100.0, 50.0, 3.0])
def test_soc_stays_in_bounds(cfg, start_soc):
    cycle = repeat_cycle(load_cycle("WLTP"), 3 if start_soc < 10 else 1)
    res = run_case(cfg, cycle, "series", initial_soc_pct=start_soc)
    assert res.df["soc_pct"].max() <= 100.0
    assert res.df["soc_pct"].min() >= 0.0


def test_battery_cannot_overcharge(cfg):
    batt = BatteryModel(BatteryParams.from_config(cfg), initial_soc=0.999)
    for _ in range(600):
        batt.step(-40_000.0, 1.0)  # force heavy charging
    assert batt.soc <= 1.0


# ---------------------------------------------------------------- regen limits
def _ctx(speed_kmh: float, battery_limit_w: float = 1e9, reason: str = "Battery power limit") -> BrakeContext:
    return BrakeContext(p_request_w=30_000.0, speed_mps=speed_kmh / 3.6, mass_kg=1475.0,
                        motor_limit_w=1e9, battery_limit_w=battery_limit_w, battery_reason=reason)


def test_regen_zero_at_full_soc(cfg):
    """At SoC = 100 % the battery accepts nothing -> regen must be zero."""
    batt = BatteryModel(BatteryParams.from_config(cfg), initial_soc=1.0)
    assert batt.max_charge_power_w() == 0.0
    ctrl = RegenController(RegenParams.from_config(cfg, strategy="series"))
    split = ctrl.split(_ctx(60.0, battery_limit_w=batt.max_charge_power_w(), reason=batt.charge_limit_reason()))
    assert split.p_regen_w == 0.0
    assert split.p_friction_w == pytest.approx(30_000.0)
    assert split.reason == "SoC limit"


def test_regen_zero_at_full_soc_in_simulation(cfg, city):
    """While SoC is still 100 %, every braking step is friction only."""
    df = run_case(cfg, city, "series", initial_soc_pct=100.0).df
    soc_before = df["soc_pct"].shift(1, fill_value=100.0)
    full = df[(soc_before >= 100.0) & (df["p_brake_req_w"] > 0)]
    assert (full["p_regen_wheel_w"] == 0.0).all()


@pytest.mark.parametrize("speed_kmh", [0.0, 2.0, 4.9, 5.0])
def test_regen_zero_below_fade_speed(cfg, speed_kmh):
    ctrl = RegenController(RegenParams.from_config(cfg, strategy="series"))
    split = ctrl.split(_ctx(speed_kmh))
    assert split.p_regen_w == 0.0
    assert split.reason == "Low-speed fade"


def test_fade_ramp_is_partial_between_end_and_start(cfg):
    ctrl = RegenController(RegenParams.from_config(cfg, strategy="series"))
    assert 0.0 < ctrl.split(_ctx(7.5)).p_regen_w < 30_000.0


def test_regen_zero_below_fade_speed_in_simulation(cfg, wltp):
    df = run_case(cfg, wltp, "series").df
    slow = df[df["avg_speed_kmh"] <= cfg["regen"]["fade_end_kmh"]]
    assert (slow["p_regen_wheel_w"] == 0.0).all()


def test_parallel_share(cfg):
    ctrl = RegenController(RegenParams.from_config(cfg, strategy="parallel", parallel_ratio=0.3))
    assert ctrl.split(_ctx(60.0)).p_regen_w == pytest.approx(0.3 * 30_000.0)


def test_level_off_gives_no_regen(cfg):
    ctrl = RegenController(RegenParams.from_config(cfg, strategy="levels", regen_level="Off"))
    assert ctrl.split(_ctx(60.0)).p_regen_w == 0.0


# ---------------------------------------------------------------- friction-only baseline
@pytest.mark.parametrize("key", list(BUILTIN_CYCLES))
def test_friction_only_recovers_exactly_zero(cfg, key):
    res = run_case(cfg, load_cycle(key), "friction_only")
    assert res.df["p_regen_batt_w"].sum() == 0.0
    assert res.df["p_regen_wheel_w"].sum() == 0.0
    assert compute_kpis(res)["kwh_recovered"] == 0.0


# ---------------------------------------------------------------- regen >= friction-only
@pytest.mark.parametrize("key", list(BUILTIN_CYCLES))
@pytest.mark.parametrize("strategy,extra", [("series", {}), ("parallel", {"parallel_ratio": 0.3}),
                                            ("levels", {"regen_level": "Low"}),
                                            ("levels", {"regen_level": "One-pedal"})])
def test_regen_ends_with_higher_or_equal_soc(cfg, key, strategy, extra):
    cycle = load_cycle(key)
    regen = run_case(cfg, cycle, strategy, regen_overrides=extra)
    fric = run_case(cfg, cycle, "friction_only")
    assert regen.final_soc_pct >= fric.final_soc_pct


def test_series_recovers_at_least_as_much_as_parallel(cfg, wltp):
    series = compute_kpis(run_case(cfg, wltp, "series"))
    parallel = compute_kpis(run_case(cfg, wltp, "parallel", regen_overrides={"parallel_ratio": 0.3}))
    assert series["kwh_recovered"] >= parallel["kwh_recovered"]


# ---------------------------------------------------------------- drive cycles
def test_wltp_matches_published_statistics(wltp):
    s = cycle_stats(wltp)
    assert s["duration_s"] == 1800
    assert s["distance_km"] == pytest.approx(23.27, rel=0.005)
    assert s["max_speed_kmh"] == pytest.approx(131.3, abs=0.1)


def test_udds_matches_published_statistics():
    s = cycle_stats(load_cycle("UDDS"))
    assert s["duration_s"] == 1369
    assert s["distance_km"] == pytest.approx(11.99, rel=0.005)


def test_nedc_duration():
    assert cycle_stats(load_cycle("NEDC"))["duration_s"] == 1180


def test_simulator_row_count(cfg, city):
    assert len(Simulator(cfg).run(city).df) == len(city) - 1
