"""Phase 1 data curation: session trimming, SG derivatives, tire-grip factor, and lap distance."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from src.config import CurationConfig
from src.data.motec_loader import TelemetryMetadata

logger = logging.getLogger(__name__)

TIRE_PRESS_COLS: tuple[str, ...] = (
    "TYRE_PRESS_LF",
    "TYRE_PRESS_RF",
    "TYRE_PRESS_LR",
    "TYRE_PRESS_RR",
)


def trim_session(
    df: pd.DataFrame,
    meta: TelemetryMetadata,
    mode: Literal["speed_threshold", "first_beacon", "none"] = "speed_threshold",
    speed_threshold_kmh: float = 100.0,
) -> pd.DataFrame:
    """Trim pit-lane exit or formation-lap transients from the start of a session."""
    if mode == "none" or df.empty:
        return df.reset_index(drop=True)

    if mode == "first_beacon":
        if not meta.beacon_markers_s:
            logger.warning("No Beacon Markers in %s; falling back to speed_threshold.", meta.filepath.name)
            mode = "speed_threshold"
        else:
            t0 = meta.beacon_markers_s[0]
            trimmed = df[df["Time"] >= t0].reset_index(drop=True)
            logger.info("Trimmed at first_beacon t=%.3fs (%d -> %d rows)", t0, len(df), len(trimmed))
            return trimmed

    if mode == "speed_threshold":
        threshold_ms = speed_threshold_kmh / 3.6
        hits = df.index[df["SPEED"] >= threshold_ms]
        if len(hits) == 0:
            logger.warning("SPEED never reached %.1f km/h; returning untrimmed data.", speed_threshold_kmh)
            return df.reset_index(drop=True)
        start_idx = int(hits[0])
        trimmed = df.iloc[start_idx:].reset_index(drop=True)
        logger.info(
            "Trimmed at SPEED >= %.1f km/h (t=%.3fs, %d -> %d rows)",
            speed_threshold_kmh,
            df.loc[start_idx, "Time"],
            len(df),
            len(trimmed),
        )
        return trimmed

    raise ValueError(f"Unknown trim mode: {mode}")


def compute_tire_grip_factor(
    df: pd.DataFrame,
    p_opt_psi: float = 27.0,
    sigma_p_psi: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 4-wheel mean tire pressure P_bar(t) and Gaussian grip factor eta_P(t)."""
    present = [c for c in TIRE_PRESS_COLS if c in df.columns]
    if not present:
        logger.warning("No TYRE_PRESS_* columns found; defaulting eta_P = 1.0.")
        ones = np.ones(len(df), dtype=float)
        return np.full(len(df), p_opt_psi, dtype=float), ones

    p_mean = df[present].mean(axis=1).to_numpy(dtype=float)
    eta_p = np.exp(-((p_mean - p_opt_psi) ** 2) / (2.0 * (sigma_p_psi**2)))
    return p_mean, eta_p


def reconstruct_lap_distance(
    time_s: np.ndarray,
    speed_ms: np.ndarray,
    beacon_markers_s: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct lap index and cumulative distance within each lap via trapezoidal integration."""
    n = len(time_s)
    lap_idx = np.zeros(n, dtype=int)
    lap_dist = np.zeros(n, dtype=float)

    if n < 2:
        return lap_idx, lap_dist

    markers = sorted([m for m in beacon_markers_s if time_s[0] <= m <= time_s[-1]])
    boundaries = [time_s[0]] + markers + [time_s[-1] + 1e-6]

    for k in range(len(boundaries) - 1):
        mask = (time_s >= boundaries[k]) & (time_s < boundaries[k + 1])
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        lap_idx[idx] = k
        if len(idx) > 1:
            lap_dist[idx] = cumulative_trapezoid(speed_ms[idx], time_s[idx], initial=0.0)

    return lap_idx, lap_dist


def curate_telemetry(
    df: pd.DataFrame,
    meta: TelemetryMetadata,
    cfg: CurationConfig,
    trim_mode: Literal["speed_threshold", "first_beacon", "none"] = "speed_threshold",
    speed_threshold_kmh: float = 100.0,
) -> pd.DataFrame:
    """Apply session trimming, Savitzky-Golay smoothing + r_dot, eta_P, and lap distance."""
    out = trim_session(df, meta, mode=trim_mode, speed_threshold_kmh=speed_threshold_kmh)
    if len(out) < cfg.savgol_window:
        raise ValueError(
            f"Telemetry has only {len(out)} samples after trimming (< window {cfg.savgol_window})."
        )

    t = out["Time"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))

    # Smooth kinematic & force channels without phase lag
    smooth_cols = [c for c in ("ROTY", "STEERANGLE", "SPEED", "G_LAT", "G_LON") if c in out.columns]
    for col in smooth_cols:
        out[col] = savgol_filter(out[col].to_numpy(dtype=float), cfg.savgol_window, cfg.savgol_polyorder)

    # Analytical first derivatives (rad/s^2 and m/s^3) from Savitzky-Golay polynomial
    out["R_DOT"] = savgol_filter(
        out["ROTY"].to_numpy(dtype=float),
        window_length=cfg.savgol_window,
        polyorder=cfg.savgol_polyorder,
        deriv=1,
        delta=dt,
    )
    out["AY_DOT"] = savgol_filter(
        out["G_LAT"].to_numpy(dtype=float),
        window_length=cfg.savgol_window,
        polyorder=cfg.savgol_polyorder,
        deriv=1,
        delta=dt,
    )

    p_mean, eta_p = compute_tire_grip_factor(out, cfg.p_opt_psi, cfg.sigma_p_psi)
    out["P_MEAN_PSI"] = p_mean
    out["ETA_P"] = eta_p

    lap_idx, lap_dist = reconstruct_lap_distance(t, out["SPEED"].to_numpy(dtype=float), meta.beacon_markers_s)
    out["LAP_INDEX"] = lap_idx
    out["LAP_DIST_M"] = lap_dist

    return out
