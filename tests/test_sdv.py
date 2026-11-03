"""SDV-layer tests: OTA validation/rollback, signal bus, twin synchronisation."""

from __future__ import annotations

import pytest

from core.drive_cycles import repeat_cycle
from core.regen_controller import RegenController, RegenParams
from core.simulator import Simulator
from sdv.ota_update import OtaManager, OtaPackage, validate_package
from sdv.signals import VSS_CATALOG, SignalBus, SignalPublisher
from sdv.twin_sync import run_twin_sync


# ---------------------------------------------------------------- OTA validator
@pytest.mark.parametrize("bad_params", [
    {"parallel_ratio": 1.5},
    {"parallel_ratio": -0.1},
    {"fade_start_kmh": 40.0},
    {"fade_end_kmh": 25.0},
    {"max_regen_power_kw": 250.0},
    {"strategy": "turbo"},
    {"regen_level": "Ludicrous"},
    {"fade_start_kmh": 6.0, "fade_end_kmh": 8.0},   # end above start
    {"battery_capacity": 50},                        # not OTA-tunable
])
def test_ota_validator_rejects_out_of_range(cfg, bad_params):
    current = RegenParams.from_config(cfg)
    errors = validate_package(OtaPackage("1.1.0", bad_params), current, cfg["ota_limits"])
    assert errors, f"expected rejection for {bad_params}"


def test_ota_validator_accepts_valid_package(cfg):
    current = RegenParams.from_config(cfg)
    pkg = OtaPackage("1.1.0", {"strategy": "levels", "regen_level": "Medium", "fade_start_kmh": 12.0})
    assert validate_package(pkg, current, cfg["ota_limits"]) == []


def test_ota_rejects_downgrade_and_tampering(cfg):
    current = RegenParams.from_config(cfg)
    assert validate_package(OtaPackage("0.9.0", {"parallel_ratio": 0.5}), current, cfg["ota_limits"])
    pkg = OtaPackage("1.1.0", {"parallel_ratio": 0.5})
    pkg.params["parallel_ratio"] = 0.9            # payload changed after signing
    assert any("Checksum" in e for e in validate_package(pkg, current, cfg["ota_limits"]))


def test_ota_rejected_package_leaves_params_unchanged(cfg):
    ctrl = RegenController(RegenParams.from_config(cfg))
    before = ctrl.params.tunables()
    mgr = OtaManager(cfg["ota_limits"])
    assert not mgr.install(ctrl, OtaPackage("2.0.0", {"max_regen_power_kw": 999.0}), 10.0)
    assert ctrl.params.tunables() == before
    assert mgr.events[-1].kind == "REJECTED"


def test_ota_install_and_rollback_mid_drive(cfg, wltp):
    mgr = OtaManager(cfg["ota_limits"])
    mgr.schedule_update(600, OtaPackage("1.1.0", {"strategy": "levels", "regen_level": "Low"}))
    mgr.schedule_rollback(1200)
    df = Simulator(cfg).run(wltp, hooks=[mgr]).df
    assert set(df.loc[df["t_s"] < 600, "regen_version"]) == {"1.0.0"}
    assert set(df.loc[(df["t_s"] > 600) & (df["t_s"] < 1200), "regen_version"]) == {"1.1.0"}
    assert set(df.loc[df["t_s"] > 1200, "regen_version"]) == {"1.0.0"}
    kinds = [e.kind for e in mgr.events]
    assert "INSTALLED" in kinds and "ROLLED_BACK" in kinds


def test_ota_changes_soc_outcome(cfg, wltp):
    """Switching to a weak regen level mid-drive must lower the final SoC."""
    base = Simulator(cfg).run(wltp)
    mgr = OtaManager(cfg["ota_limits"])
    mgr.schedule_update(300, OtaPackage("1.1.0", {"strategy": "levels", "regen_level": "Low"}))
    ota = Simulator(cfg).run(wltp, hooks=[mgr])
    assert ota.final_soc_pct < base.final_soc_pct


# ---------------------------------------------------------------- signal bus
def test_signal_bus_publish_subscribe(cfg, city):
    bus = SignalBus()
    received: list = []
    bus.subscribe("Vehicle.Chassis.Brake.*", received.append)
    res = Simulator(cfg).run(city, hooks=[SignalPublisher(bus)])
    steps = len(res.df)
    assert bus.message_count == steps * len(VSS_CATALOG)
    assert len(received) == steps * 3       # RegenPower, FrictionPower, RegenLimitReason
    assert bus.latest("Vehicle.Speed") == pytest.approx(res.df["speed_kmh"].iloc[-1], abs=0.01)
    # VSS sign convention: positive current = into the battery
    assert bus.latest("Vehicle.Powertrain.TractionBattery.CurrentCurrent") == pytest.approx(
        -res.df["current_a"].iloc[-1], abs=0.01)


def test_signal_bus_rejects_unknown_signal():
    with pytest.raises(KeyError):
        SignalBus().publish("Vehicle.Made.Up", 1.0, 0.0)


def test_custom_signals_are_marked():
    assert VSS_CATALOG["Vehicle.Chassis.Brake.RegenPower"].custom
    assert not VSS_CATALOG["Vehicle.Speed"].custom


# ---------------------------------------------------------------- twin sync
def test_twin_correction_reduces_error(cfg, wltp):
    cycle = repeat_cycle(wltp, 2)
    res = run_twin_sync(cfg, cycle, RegenParams.from_config(cfg), 90.0, 25.0, 0.0, 0.0, method="kalman")
    m = res.metrics
    assert m["rmse_corrected_pct"] < m["rmse_open_pct"]
    assert m["rmse_corrected_pct"] < m["rmse_raw_sensor_pct"]   # filter beats the raw noisy sensor
    assert abs(m["final_error_corrected_pct"]) < 0.5
