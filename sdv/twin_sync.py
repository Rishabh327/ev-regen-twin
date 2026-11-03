"""Digital-twin synchronisation: "physical vehicle" stream vs. the twin.

A simulation runs on its own. A digital TWIN is continuously fed with data
from its physical counterpart and corrects itself, so it stays a faithful
mirror of *that particular car*. Here:

1. "Physical vehicle"  = same model but with a hidden parameter mismatch
   (+5 % mass, +10 % battery resistance) and noisy sensors
   (SoC σ = 0.3 %, current σ = 1.5 A, voltage σ = 0.5 V).
2. Twin (open loop)    = nominal model, started with a 2 % SoC initial error
   (typical BMS initialisation uncertainty). It never looks at the car.
3. Twin (corrected)    = nominal model + correction every `sync_period_s`
   using the measured SoC:
     * "kalman" : scalar Kalman-style blend
            predict : P ← P + Q·Δt             (uncertainty grows while we only model)
            update  : K = P / (P + R)          (R = sensor noise variance)
                      SoC ← SoC + K·(SoC_meas − SoC)
                      P ← (1 − K)·P
     * "resync" : hard reset SoC ← SoC_meas (K = 1, simplest, but copies noise)

RMSE = sqrt( mean( (SoC_twin − SoC_true)² ) )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.regen_controller import RegenParams
from core.simulator import Simulator


@dataclass
class TwinSyncResult:
    df: pd.DataFrame
    metrics: dict[str, float]
    method: str
    sync_period_s: int


class KalmanSocCorrector:
    """Simulator hook: blends the twin's SoC with noisy measurements."""

    def __init__(self, measured_soc_pct: dict[float, float], sync_period_s: int, q_pct2_per_s: float,
                 r_pct2: float, method: str = "kalman", p0_pct2: float = 4.0) -> None:
        self.meas = measured_soc_pct
        self.period = max(1, int(sync_period_s))
        self.q = q_pct2_per_s
        self.r = r_pct2
        self.method = method
        self.p = p0_pct2            # initial uncertainty (%²), e.g. (2 %)²
        self.gains: list[tuple[float, float]] = []

    def __call__(self, sim: Simulator, row: dict[str, Any]) -> None:
        t = row["t_s"]
        self.p += self.q * sim.dt
        if int(round(t)) % self.period != 0 or t not in self.meas:
            return
        z = self.meas[t]
        soc_pct = sim.battery.soc * 100.0
        k = 1.0 if self.method == "resync" else self.p / (self.p + self.r)
        soc_pct += k * (z - soc_pct)
        self.p = (1.0 - k) * self.p
        sim.battery.soc = min(1.0, max(0.0, soc_pct / 100.0))
        row["soc_pct"] = sim.battery.soc * 100.0
        self.gains.append((t, k))


def run_twin_sync(cfg: dict[str, Any], cycle: pd.DataFrame, regen_params: RegenParams,
                  initial_soc_pct: float, ambient_temp_c: float, road_grade_pct: float,
                  payload_kg: float, method: str = "kalman") -> TwinSyncResult:
    tw = cfg["twin"]
    rng = np.random.default_rng(tw["random_seed"])
    common = dict(ambient_temp_c=ambient_temp_c, road_grade_pct=road_grade_pct, payload_kg=payload_kg)

    # 1. physical vehicle (hidden truth)
    physical = Simulator(cfg, regen_params, initial_soc_pct=initial_soc_pct,
                         mass_scale=1.0 + tw["mass_mismatch_pct"] / 100.0,
                         r0_scale=1.0 + tw["r0_mismatch_pct"] / 100.0, label="physical", **common).run(cycle)
    truth = physical.df
    n = len(truth)
    soc_meas = np.round(truth["soc_pct"].to_numpy() + rng.normal(0.0, tw["soc_noise_std_pct"], n), 1)
    i_meas = truth["current_a"].to_numpy() + rng.normal(0.0, tw["current_noise_std_a"], n)
    v_meas = truth["voltage_v"].to_numpy() + rng.normal(0.0, tw["voltage_noise_std_v"], n)
    meas_map = dict(zip(truth["t_s"].to_numpy(), soc_meas))

    twin_start = initial_soc_pct + tw["initial_soc_error_pct"]

    # 2. open-loop twin
    open_loop = Simulator(cfg, regen_params, initial_soc_pct=twin_start, label="twin-open", **common).run(cycle)

    # 3. corrected twin
    corrector = KalmanSocCorrector(meas_map, tw["sync_period_s"],
                                   q_pct2_per_s=tw["process_noise_std_pct_per_s"] ** 2,
                                   r_pct2=tw["soc_noise_std_pct"] ** 2, method=method,
                                   p0_pct2=max(tw["initial_soc_error_pct"], 0.5) ** 2)
    corrected = Simulator(cfg, regen_params, initial_soc_pct=twin_start, label="twin-corrected",
                          **common).run(cycle, hooks=[corrector])

    df = pd.DataFrame({
        "t_s": truth["t_s"],
        "soc_true_pct": truth["soc_pct"],
        "soc_measured_pct": soc_meas,
        "soc_twin_open_pct": open_loop.df["soc_pct"].to_numpy(),
        "soc_twin_corrected_pct": corrected.df["soc_pct"].to_numpy(),
        "current_measured_a": i_meas,
        "current_twin_a": corrected.df["current_a"].to_numpy(),
        "voltage_measured_v": v_meas,
        "voltage_twin_v": corrected.df["voltage_v"].to_numpy(),
    })
    err_open = df["soc_twin_open_pct"] - df["soc_true_pct"]
    err_corr = df["soc_twin_corrected_pct"] - df["soc_true_pct"]
    err_meas = df["soc_measured_pct"] - df["soc_true_pct"]
    metrics = {
        "rmse_open_pct": float(np.sqrt(np.mean(err_open ** 2))),
        "rmse_corrected_pct": float(np.sqrt(np.mean(err_corr ** 2))),
        "rmse_raw_sensor_pct": float(np.sqrt(np.mean(err_meas ** 2))),
        "max_abs_open_pct": float(err_open.abs().max()),
        "max_abs_corrected_pct": float(err_corr.abs().max()),
        "final_error_open_pct": float(err_open.iloc[-1]),
        "final_error_corrected_pct": float(err_corr.iloc[-1]),
        "current_rmse_a": float(np.sqrt(np.mean((df["current_twin_a"] - truth["current_a"]) ** 2))),
        "n_syncs": float(len(corrector.gains)),
        "final_kalman_gain": corrector.gains[-1][1] if corrector.gains else 0.0,
    }
    return TwinSyncResult(df, metrics, method, tw["sync_period_s"])
