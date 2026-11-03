"""Core physics models of the EV energy-recovery digital twin.

Modules
-------
drive_cycles      : load / generate standard drive cycles (speed vs time)
vehicle_dynamics  : road-load forces and wheel power
motor_model       : motor + inverter efficiency map and torque/power limits
battery_model     : 1-RC Thevenin equivalent-circuit battery with SoC
regen_controller  : regenerative / friction brake blending strategies
simulator         : time-step loop tying everything together
analysis          : KPIs, braking-event table and sanity checks
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "vehicle_default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the vehicle/battery/regen configuration JSON file.

    Keys starting with an underscore (e.g. "_comment") are documentation only.
    A deep copy is returned so callers may modify it freely.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return copy.deepcopy(cfg)
