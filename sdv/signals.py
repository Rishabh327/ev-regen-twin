"""Vehicle signals in COVESA Vehicle Signal Specification (VSS) style + a tiny signal bus.

Why VSS?  In a software-defined vehicle, apps never read raw CAN frames.
They read *named, typed, unit-annotated* signals from a central data broker
(e.g. Eclipse Kuksa.val). Using VSS names here means the twin's outputs could
later be pushed to a real Kuksa broker with almost no code change.

Signals marked `custom=True` are NOT in the official VSS tree; they are
project-specific extensions (VSS explicitly allows private branches/overlays).

Sign convention (follows VSS): TractionBattery.CurrentCurrent and
CurrentPower are POSITIVE when energy flows INTO the battery (charging/regen)
and NEGATIVE when driving. The physics core uses the opposite sign
(discharge positive), so the publisher flips it.
"""

from __future__ import annotations

import fnmatch
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class SignalSpec:
    path: str
    unit: str
    datatype: str
    description: str
    custom: bool = False


VSS_CATALOG: dict[str, SignalSpec] = {s.path: s for s in [
    SignalSpec("Vehicle.Speed", "km/h", "float", "Vehicle speed."),
    SignalSpec("Vehicle.TraveledDistance", "km", "float", "Odometer distance travelled (this trip)."),
    SignalSpec("Vehicle.Acceleration.Longitudinal", "m/s^2", "float", "Longitudinal acceleration."),
    SignalSpec("Vehicle.Powertrain.TractionBattery.StateOfCharge.Current", "percent", "float",
               "Physical state of charge of the high-voltage battery."),
    SignalSpec("Vehicle.Powertrain.TractionBattery.CurrentCurrent", "A", "float",
               "Battery current. Positive = into battery (charging/regen)."),
    SignalSpec("Vehicle.Powertrain.TractionBattery.CurrentVoltage", "V", "float", "Battery terminal voltage."),
    SignalSpec("Vehicle.Powertrain.TractionBattery.CurrentPower", "W", "float",
               "Battery power. Positive = into battery."),
    SignalSpec("Vehicle.Powertrain.TractionBattery.Temperature.Average", "celsius", "float",
               "Average battery cell temperature (= ambient in this model)."),
    SignalSpec("Vehicle.Powertrain.TractionMotor.Speed", "rpm", "float", "Traction motor speed."),
    SignalSpec("Vehicle.Powertrain.TractionMotor.Torque", "Nm", "float",
               "Motor torque. Negative = generating (regen)."),
    SignalSpec("Vehicle.Chassis.Brake.RegenPower", "W", "float",
               "CUSTOM: wheel-side braking power taken by the motor.", custom=True),
    SignalSpec("Vehicle.Chassis.Brake.FrictionPower", "W", "float",
               "CUSTOM: braking power dissipated as heat in the friction brakes.", custom=True),
    SignalSpec("Vehicle.Chassis.Brake.RegenLimitReason", "-", "string",
               "CUSTOM: why regen was below the request (SoC/motor/fade/strategy).", custom=True),
    SignalSpec("Vehicle.Powertrain.TractionBattery.EnergyRegenerated", "Wh", "float",
               "CUSTOM: cumulative energy returned to the battery this trip.", custom=True),
    SignalSpec("Vehicle.Software.RegenCalibration.Version", "-", "string",
               "CUSTOM: version of the regen calibration currently running (changes on OTA).", custom=True),
]}


@dataclass(frozen=True)
class SignalSample:
    t_s: float
    path: str
    value: Any


Callback = Callable[[SignalSample], None]


class SignalBus:
    """Minimal in-memory publish/subscribe data broker (think: a toy Kuksa.val).

    * publish(path, value, t)  - producers (the vehicle model) push values
    * subscribe(pattern, cb)   - consumers get a callback; wildcards allowed,
                                 e.g. "Vehicle.Chassis.Brake.*"
    * latest(path)             - last known value (like a 'get' request)
    * history                  - optional full log for plotting/export
    """

    def __init__(self, record_history: bool = True, strict: bool = True) -> None:
        self._subs: dict[str, list[Callback]] = defaultdict(list)
        self._latest: dict[str, SignalSample] = {}
        self.record_history = record_history
        self.strict = strict
        self.history: list[SignalSample] = []
        self.message_count = 0

    def subscribe(self, pattern: str, callback: Callback) -> None:
        self._subs[pattern].append(callback)

    def publish(self, path: str, value: Any, t_s: float) -> None:
        if self.strict and path not in VSS_CATALOG:
            raise KeyError(f"Signal '{path}' is not in the VSS catalogue")
        sample = SignalSample(t_s, path, value)
        self._latest[path] = sample
        self.message_count += 1
        if self.record_history:
            self.history.append(sample)
        for pattern, callbacks in self._subs.items():
            if pattern == path or fnmatch.fnmatchcase(path, pattern):
                for cb in callbacks:
                    cb(sample)

    def latest(self, path: str) -> Any:
        s = self._latest.get(path)
        return None if s is None else s.value

    def snapshot(self) -> dict[str, Any]:
        return {p: s.value for p, s in self._latest.items()}

    @property
    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subs.values())

    def history_frame(self) -> pd.DataFrame:
        """History as a wide table: one row per time stamp, one column per signal."""
        if not self.history:
            return pd.DataFrame()
        long = pd.DataFrame([(s.t_s, s.path, s.value) for s in self.history], columns=["t_s", "path", "value"])
        return long.pivot_table(index="t_s", columns="path", values="value", aggfunc="last").reset_index()


def row_to_signals(row: dict[str, Any], battery_temp_c: float, regen_energy_wh: float) -> dict[str, Any]:
    """Map one simulator row to VSS signal values (with VSS sign conventions)."""
    return {
        "Vehicle.Speed": round(row["speed_kmh"], 2),
        "Vehicle.TraveledDistance": round(row["distance_km"], 4),
        "Vehicle.Acceleration.Longitudinal": round(row["accel_mps2"], 3),
        "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current": round(row["soc_pct"], 3),
        "Vehicle.Powertrain.TractionBattery.CurrentCurrent": round(-row["current_a"], 2),
        "Vehicle.Powertrain.TractionBattery.CurrentVoltage": round(row["voltage_v"], 2),
        "Vehicle.Powertrain.TractionBattery.CurrentPower": round(-row["p_batt_w"], 1),
        "Vehicle.Powertrain.TractionBattery.Temperature.Average": battery_temp_c,
        "Vehicle.Powertrain.TractionMotor.Speed": round(row["motor_speed_rpm"], 1),
        "Vehicle.Powertrain.TractionMotor.Torque": round(row["motor_torque_nm"], 2),
        "Vehicle.Chassis.Brake.RegenPower": round(row["p_regen_wheel_w"], 1),
        "Vehicle.Chassis.Brake.FrictionPower": round(row["p_friction_w"], 1),
        "Vehicle.Chassis.Brake.RegenLimitReason": row["regen_reason"] or "-",
        "Vehicle.Powertrain.TractionBattery.EnergyRegenerated": round(regen_energy_wh, 2),
        "Vehicle.Software.RegenCalibration.Version": row["regen_version"],
    }


class SignalPublisher:
    """Simulator hook that publishes every step's state to a SignalBus."""

    def __init__(self, bus: SignalBus) -> None:
        self.bus = bus
        self._regen_wh = 0.0

    def __call__(self, sim: Any, row: dict[str, Any]) -> None:
        self._regen_wh += row["p_regen_batt_w"] * sim.dt / 3600.0
        for path, value in row_to_signals(row, sim.ambient_temp_c, self._regen_wh).items():
            self.bus.publish(path, value, row["t_s"])


def catalog_frame() -> pd.DataFrame:
    """Catalogue as a table for the dashboard/README."""
    return pd.DataFrame([{
        "VSS path": s.path, "unit": s.unit, "type": s.datatype,
        "standard / custom": "CUSTOM extension" if s.custom else "VSS standard",
        "description": s.description,
    } for s in VSS_CATALOG.values()])
