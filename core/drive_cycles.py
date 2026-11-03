"""Drive cycles: load or generate speed-vs-time tables at 1 s resolution.

A drive cycle is a pandas DataFrame with two columns:
    time_s     : time in seconds (0, 1, 2, ...)
    speed_kmh  : vehicle speed in km/h

Exact / approximate status (also stated in README.md)
------------------------------------------------------
* NEDC   : EXACT shape. Built from the published regulation segments
           (4 x ECE-15 urban + 1 x EUDC extra-urban), 1180 s, ~10.9 km.
* MIDC   : APPROXIMATION of the Indian Modified Indian Driving Cycle:
           4 x ECE-15 + EUDC for low-powered vehicles (top speed 90 km/h).
* WLTP   : APPROXIMATION of WLTC Class 3b. Each of the 4 phases (Low,
           Medium, High, Extra-High) is built from micro-trips and then
           shape-fitted so that phase DURATION, MAX SPEED and DISTANCE match
           the published values exactly. Second-by-second wiggles differ.
* UDDS   : APPROXIMATION of US EPA UDDS (FTP-72). 1369 s, 11.99 km (7.45 mi), 91.2 km/h
           max are matched exactly; micro-trip details differ.
* CITY   : custom "city stop-and-go" cycle designed for this project.

Each cycle is built from "segments" (target_speed_kmh, duration_s): speed
ramps linearly from the previous speed to the target over `duration_s`.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from core import PROJECT_ROOT

DATA_DIR: Path = PROJECT_ROOT / "data"

Segment = tuple[float, int]  # (target speed km/h, duration s)


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def build_from_segments(segments: Iterable[Segment], start_speed_kmh: float = 0.0) -> np.ndarray:
    """Turn a list of (target_speed, duration) segments into a 1 Hz speed array.

    The returned array starts at t = 0 with `start_speed_kmh`, so its length is
    sum(durations) + 1.
    """
    speeds: list[float] = [start_speed_kmh]
    prev = start_speed_kmh
    for target, duration in segments:
        duration = int(duration)
        if duration <= 0:
            continue
        for i in range(1, duration + 1):
            speeds.append(prev + (target - prev) * i / duration)
        prev = float(target)
    return np.asarray(speeds, dtype=float)


def _pad_to_duration(speed: np.ndarray, duration_s: int) -> np.ndarray:
    """Pad with idle (0 km/h) or trim so the array covers exactly `duration_s` seconds."""
    target_len = duration_s + 1
    if len(speed) < target_len:
        speed = np.concatenate([speed, np.zeros(target_len - len(speed))])
    return speed[:target_len]


def _distance_km(speed_kmh: np.ndarray, dt_s: float = 1.0) -> float:
    """Distance by trapezoidal integration of speed."""
    v = speed_kmh / 3.6
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * dt_s) / 1000.0)


def _fit_shape(speed_kmh: np.ndarray, target_km: float, target_max_kmh: float) -> np.ndarray:
    """Shape-fit a speed trace so its max speed AND distance match published values.

    Transform used:  v' = v_max_target * (v / v_max)^p
    * p = 1 keeps the shape unchanged; p < 1 raises mid speeds (more distance),
      p > 1 lowers them. Zero stays zero and the peak stays at v_max_target.
    * p is found by bisection so the distance equals `target_km`.
    """
    v_max = float(np.max(speed_kmh))
    if v_max <= 0:
        return speed_kmh
    norm = speed_kmh / v_max

    def dist_for(p: float) -> float:
        return _distance_km(target_max_kmh * norm ** p)

    lo, hi = 0.2, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if dist_for(mid) > target_km:
            lo = mid  # too far -> increase exponent
        else:
            hi = mid
    return target_max_kmh * norm ** (0.5 * (lo + hi))


def _to_frame(speed_kmh: np.ndarray) -> pd.DataFrame:
    speed_kmh = np.clip(np.round(speed_kmh, 2), 0.0, None)
    return pd.DataFrame({"time_s": np.arange(len(speed_kmh), dtype=float), "speed_kmh": speed_kmh})


# --------------------------------------------------------------------------
# NEDC / ECE-15 / EUDC (exact regulation segments)
# --------------------------------------------------------------------------
# ECE-15 elementary urban cycle: 195 s, max 50 km/h, ~0.994 km
ECE15_SEGMENTS: list[Segment] = [
    (0, 11), (15, 4), (15, 8), (0, 5),
    (0, 21), (32, 12), (32, 24), (0, 11),
    (0, 21), (50, 26), (50, 12), (35, 8), (35, 13), (0, 12),
    (0, 7),
]

# EUDC extra-urban cycle: 400 s, max 120 km/h, ~6.955 km
EUDC_SEGMENTS: list[Segment] = [
    (0, 20), (70, 41), (70, 50), (50, 8), (50, 69), (70, 13), (70, 50),
    (100, 35), (100, 30), (120, 20), (120, 10), (0, 34), (0, 20),
]

# EUDC for low-powered vehicles (used in MIDC): top speed limited to 90 km/h, 400 s
EUDC_LOW_POWER_SEGMENTS: list[Segment] = [
    (0, 20), (70, 41), (70, 50), (50, 8), (50, 69), (70, 13), (70, 50),
    (90, 35), (90, 60), (0, 34), (0, 20),
]


def nedc() -> pd.DataFrame:
    """New European Driving Cycle: 4 x ECE-15 + EUDC = 1180 s (EXACT segments)."""
    return _to_frame(build_from_segments(ECE15_SEGMENTS * 4 + EUDC_SEGMENTS))


def midc() -> pd.DataFrame:
    """Modified Indian Driving Cycle (APPROXIMATION): 4 x ECE-15 + low-power EUDC (90 km/h)."""
    return _to_frame(build_from_segments(ECE15_SEGMENTS * 4 + EUDC_LOW_POWER_SEGMENTS))


# --------------------------------------------------------------------------
# WLTC Class 3b (approximation fitted to published phase statistics)
# --------------------------------------------------------------------------
# (duration s, distance km, max km/h) from the UN GTR No.15 phase tables
WLTP_PHASES: dict[str, tuple[int, float, float]] = {
    "Low": (589, 3.095, 56.5),
    "Medium": (433, 4.756, 76.6),
    "High": (455, 7.162, 97.4),
    "Extra-High": (323, 8.254, 131.3),
}

WLTP_SEGMENTS: dict[str, list[Segment]] = {
    "Low": [
        (0, 11), (15, 6), (25, 8), (25, 6), (12, 10), (0, 6),
        (0, 10), (30, 12), (42, 15), (42, 20), (30, 10), (0, 12),
        (0, 20), (25, 10), (45, 15), (56.5, 12), (50, 25), (40, 15), (50, 20), (30, 15), (0, 14),
        (0, 15), (20, 8), (35, 12), (35, 25), (20, 10), (0, 8),
        (0, 12), (30, 10), (48, 15), (48, 30), (35, 12), (45, 15), (30, 12), (0, 12),
        (0, 10), (20, 8), (32, 10), (32, 20), (15, 10), (0, 8),
    ],
    "Medium": [
        (0, 6), (30, 10), (50, 15), (65, 20), (60, 25), (70, 20), (76.6, 15), (70, 25),
        (55, 20), (35, 15), (0, 15),
        (0, 10), (25, 8), (45, 12), (55, 15), (55, 30), (65, 15), (60, 25), (45, 15),
        (30, 10), (50, 15), (50, 20), (25, 15), (0, 12),
    ],
    "High": [
        (0, 5), (35, 12), (60, 15), (80, 20), (85, 30), (97.4, 20), (90, 30), (80, 25),
        (85, 30), (70, 20), (75, 30), (60, 20), (40, 15), (0, 18),
        (0, 10), (30, 10), (60, 15), (80, 20), (80, 40), (65, 15), (0, 22),
    ],
    "Extra-High": [
        (0, 5), (40, 12), (70, 18), (95, 20), (110, 25), (120, 30), (125, 40),
        (131.3, 20), (125, 30), (115, 25), (90, 25), (60, 20), (0, 25),
    ],
}


def wltp_class3() -> pd.DataFrame:
    """WLTC Class 3b (APPROXIMATION). 1800 s, 23.27 km, max 131.3 km/h."""
    pieces: list[np.ndarray] = []
    for phase, (duration, dist_km, vmax) in WLTP_PHASES.items():
        raw = _pad_to_duration(build_from_segments(WLTP_SEGMENTS[phase]), duration)
        fitted = _fit_shape(raw, dist_km, vmax)
        # drop the first sample of later phases (it duplicates the previous end, 0 km/h)
        pieces.append(fitted if not pieces else fitted[1:])
    return _to_frame(np.concatenate(pieces))


# --------------------------------------------------------------------------
# US EPA UDDS (approximation fitted to published statistics)
# --------------------------------------------------------------------------
UDDS_DURATION_S = 1369
UDDS_DISTANCE_KM = 11.99
UDDS_MAX_KMH = 91.25

UDDS_SEGMENTS: list[Segment] = [
    # first transient part
    (0, 20), (30, 8), (48, 10), (50, 30), (40, 8), (52, 20), (48, 25), (0, 14), (0, 10),
    # the famous "big hill" to ~91 km/h
    (40, 12), (64, 18), (80, 25), (91.25, 20), (88, 30), (72, 15), (85, 20), (64, 20), (0, 25), (0, 15),
    # urban micro-trips
    (40, 10), (40, 15), (0, 12), (0, 15),
    (50, 15), (45, 20), (0, 14), (0, 20),
    (35, 10), (35, 15), (0, 10), (0, 15),
    (55, 15), (50, 25), (0, 15), (0, 20),
    (40, 12), (30, 10), (0, 10), (0, 15),
    (48, 15), (48, 30), (0, 14), (0, 20),
    (30, 10), (25, 10), (0, 8), (0, 15),
    (52, 18), (45, 20), (0, 15), (0, 20),
    (40, 12), (40, 25), (0, 12), (0, 18),
    (55, 18), (50, 30), (0, 16), (0, 20),
    (35, 10), (30, 15), (0, 10), (0, 15),
    (45, 14), (40, 20), (0, 12), (0, 20),
    (40, 12), (48, 20), (35, 10), (0, 12), (0, 15),
    (30, 10), (30, 12), (0, 9),
]


def udds() -> pd.DataFrame:
    """US EPA Urban Dynamometer Driving Schedule (APPROXIMATION). 1369 s, 11.99 km."""
    raw = _pad_to_duration(build_from_segments(UDDS_SEGMENTS), UDDS_DURATION_S)
    return _to_frame(_fit_shape(raw, UDDS_DISTANCE_KM, UDDS_MAX_KMH))


# --------------------------------------------------------------------------
# Custom city stop-and-go
# --------------------------------------------------------------------------
CITY_PATTERN: list[Segment] = [
    (0, 10), (25, 6), (25, 10), (0, 5),     # short hop to 25 km/h
    (0, 8), (40, 10), (40, 15), (0, 8),     # signal-to-signal at 40 km/h
    (0, 6), (15, 4), (15, 6), (0, 3),       # crawling traffic
]


def city_stop_and_go() -> pd.DataFrame:
    """Custom dense-traffic city cycle: 6 repeats of a 91 s pattern + idle = 600 s."""
    return _to_frame(_pad_to_duration(build_from_segments(CITY_PATTERN * 6), 600))


# --------------------------------------------------------------------------
# Registry, CSV I/O, statistics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CycleInfo:
    key: str
    label: str
    csv_name: str
    generator: Callable[[], pd.DataFrame]
    exact: bool
    note: str


BUILTIN_CYCLES: dict[str, CycleInfo] = {
    "WLTP": CycleInfo("WLTP", "WLTP Class 3b", "wltp_class3.csv", wltp_class3, False,
                      "Approximation fitted to published phase duration, distance and max speed."),
    "NEDC": CycleInfo("NEDC", "NEDC (4xECE-15 + EUDC)", "nedc.csv", nedc, True,
                      "Exact regulation segments."),
    "UDDS": CycleInfo("UDDS", "US EPA UDDS (FTP-72)", "udds.csv", udds, False,
                      "Approximation fitted to published duration, distance and max speed."),
    "MIDC": CycleInfo("MIDC", "Indian MIDC", "midc.csv", midc, False,
                      "Approximation: 4xECE-15 + low-power EUDC (90 km/h)."),
    "CITY": CycleInfo("CITY", "City stop-and-go (custom)", "city_stop_and_go.csv", city_stop_and_go, True,
                      "Custom cycle defined in this project."),
}


def load_cycle(key: str, prefer_csv: bool = True) -> pd.DataFrame:
    """Load a built-in cycle, from data/*.csv when present, otherwise generate it."""
    info = BUILTIN_CYCLES[key.upper()]
    csv_path = DATA_DIR / info.csv_name
    if prefer_csv and csv_path.exists():
        return normalise_cycle(pd.read_csv(csv_path))
    return info.generator()


def normalise_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Accept any CSV with a time column and a speed column; resample to 1 s.

    Column detection is case-insensitive: the first column containing "time"
    (or the first column) is time in seconds; the first containing "speed"
    (or the second column) is speed. If the speed header contains "m/s" or
    "mps" it is converted to km/h.
    """
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("Cycle CSV needs at least two columns: time and speed.")
    lower = [str(c).lower() for c in cols]
    t_col = next((c for c, l in zip(cols, lower) if "time" in l), cols[0])
    v_col = next((c for c, l in zip(cols, lower) if "speed" in l or "vel" in l), cols[1])
    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype=float)
    ok = ~(np.isnan(t) | np.isnan(v))
    t, v = t[ok], v[ok]
    if len(t) < 2:
        raise ValueError("Cycle CSV has fewer than two valid rows.")
    order = np.argsort(t)
    t, v = t[order], v[order]
    v_name = str(v_col).lower()
    if "m/s" in v_name or "mps" in v_name:
        v = v * 3.6
    t = t - t[0]
    t_new = np.arange(0.0, np.floor(t[-1]) + 1.0, 1.0)
    v_new = np.interp(t_new, t, np.clip(v, 0.0, None))
    return pd.DataFrame({"time_s": t_new, "speed_kmh": np.round(v_new, 3)})


def parse_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse a user-uploaded CSV (bytes from Streamlit's file_uploader)."""
    return normalise_cycle(pd.read_csv(StringIO(raw_bytes.decode("utf-8-sig"))))


def repeat_cycle(cycle: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """Concatenate a cycle `repeats` times end-to-end (time stays continuous)."""
    repeats = max(1, int(repeats))
    speed = cycle["speed_kmh"].to_numpy()
    parts = [speed] + [speed[1:] for _ in range(repeats - 1)]
    return _to_frame(np.concatenate(parts))


def cycle_stats(cycle: pd.DataFrame) -> dict[str, float]:
    """Basic statistics used in the README table and the dashboard."""
    v = cycle["speed_kmh"].to_numpy(dtype=float)
    duration = float(cycle["time_s"].iloc[-1] - cycle["time_s"].iloc[0])
    dist = _distance_km(v)
    moving = v > 0.5
    stops = int(np.sum((v[1:] <= 0.5) & (v[:-1] > 0.5)))
    return {
        "duration_s": duration,
        "distance_km": dist,
        "max_speed_kmh": float(v.max()),
        "mean_speed_kmh": dist / (duration / 3600.0) if duration > 0 else 0.0,
        "idle_pct": 100.0 * float(np.mean(~moving)),
        "stops": float(stops),
    }


def generate_all_csvs() -> None:
    """Regenerate every built-in cycle into data/*.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for info in BUILTIN_CYCLES.values():
        df = info.generator()
        df.to_csv(DATA_DIR / info.csv_name, index=False)
        s = cycle_stats(df)
        print(f"{info.key:5s} -> data/{info.csv_name:22s} {s['duration_s']:6.0f} s  "
              f"{s['distance_km']:6.2f} km  max {s['max_speed_kmh']:6.1f} km/h  "
              f"mean {s['mean_speed_kmh']:5.1f} km/h  stops {s['stops']:.0f}  "
              f"[{'exact' if info.exact else 'approximation'}]")


if __name__ == "__main__":
    generate_all_csvs()
