"""Georeferencing Z-score anomalies to physical circuit corners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sindy_piml.config import CornerRegion


@dataclass
class AnomalySummary:
    """Summary of >threshold Z-score anomalies mapped to track corners."""

    threshold_z: float
    total_anomalies: int
    total_frames: int
    corner_counts: dict[str, int]
    anomalies_df: pd.DataFrame


def _assign_corner(dist_m: float, corners: list[CornerRegion]) -> str:
    for c in corners:
        if c.start_m <= dist_m < c.end_m:
            return c.name
    return "Straight / Other"


def map_anomalies_to_corners(
    df: pd.DataFrame,
    z_scores: np.ndarray,
    corners: list[CornerRegion],
    threshold_z: float = 5.0,
) -> AnomalySummary:
    """Identify frames with |Z(t)| > threshold_z and map them to track corners."""
    mask = np.abs(z_scores) > threshold_z
    idx = np.flatnonzero(mask)

    if len(idx) == 0:
        return AnomalySummary(
            threshold_z=threshold_z,
            total_anomalies=0,
            total_frames=len(df),
            corner_counts={},
            anomalies_df=pd.DataFrame(columns=["Time", "LAP_INDEX", "LAP_DIST_M", "Z_SCORE", "CORNER"]),
        )

    sub = df.iloc[idx][["Time", "LAP_INDEX", "LAP_DIST_M", "SPEED", "G_LAT", "ROTY"]].copy()
    sub["Z_SCORE"] = z_scores[idx]
    sub["CORNER"] = [
        _assign_corner(float(d), corners) for d in sub["LAP_DIST_M"].to_numpy(dtype=float)
    ]

    counts = sub["CORNER"].value_counts().to_dict()
    return AnomalySummary(
        threshold_z=threshold_z,
        total_anomalies=len(sub),
        total_frames=len(df),
        corner_counts={str(k): int(v) for k, v in counts.items()},
        anomalies_df=sub.reset_index(drop=True),
    )
