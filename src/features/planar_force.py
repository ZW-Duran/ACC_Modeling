"""Stage 2: Whole-Vehicle Planar Force & Bicycle-Model Yaw Dynamics library."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import CandidateLibrary, FeatureMatrix


class PlanarForceLibrary(CandidateLibrary):
    """Whole-vehicle planar force + 2-DOF bicycle library for [delta, delta_dot, a_y, a_x, v, eta_P] -> r_dot.

    Combines causal steering torque & relaxation lead (delta, delta_dot), linear &
    speed-dependent yaw damping (r, r/v), lateral-slip velocity proxy ((a_y - v*r)/v),
    transient lateral force rate (a_y_dot), combined-slip friction-ellipse deficit
    scaled by tire pressure factor eta_P, and nonlinear front-tire saturation (delta*|delta|).
    """

    name = "planar_force"
    REQUIRED_COLS: tuple[str, ...] = (
        "STEERANGLE",
        "STEER_DOT",
        "ROTY",
        "G_LAT",
        "G_LON",
        "SPEED",
        "ETA_P",
        "R_DOT",
        "AY_DOT",
    )

    FEATURE_NAMES: list[str] = [
        "delta",
        "delta_dot",
        "r",
        "r/v",
        "(a_y-v*r)/v",
        "a_y_dot",
        "a_y*|a_x|/eta_P^2",
        "delta*|delta|",
    ]

    LATEX_NAMES: list[str] = [
        r"\delta",
        r"\dot{\delta}",
        r"r",
        r"\frac{r}{v}",
        r"\frac{a_y - v r}{v}",
        r"\dot{a}_y",
        r"\frac{a_y |a_x|}{\eta_P^2}",
        r"\delta |\delta|",
    ]

    def build(self, df: pd.DataFrame) -> FeatureMatrix:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for PlanarForceLibrary: {missing}")

        delta = df["STEERANGLE"].to_numpy(dtype=float)
        delta_dot = df["STEER_DOT"].to_numpy(dtype=float)
        r = df["ROTY"].to_numpy(dtype=float)
        ay = df["G_LAT"].to_numpy(dtype=float)
        ax = df["G_LON"].to_numpy(dtype=float)
        v = np.clip(df["SPEED"].to_numpy(dtype=float), 5.0, None)
        eta_p = np.clip(df["ETA_P"].to_numpy(dtype=float), 1e-3, None)
        r_dot = df["R_DOT"].to_numpy(dtype=float)
        ay_dot = df["AY_DOT"].to_numpy(dtype=float)

        theta = np.column_stack(
            [
                delta,
                delta_dot,
                r,
                r / v,
                (ay - v * r) / v,
                ay_dot,
                ay * np.abs(ax) / (eta_p**2),
                delta * np.abs(delta),
            ]
        )

        return FeatureMatrix(
            theta=theta,
            target=r_dot,
            feature_names=list(self.FEATURE_NAMES),
            latex_names=list(self.LATEX_NAMES),
            target_name="r_dot",
        )
