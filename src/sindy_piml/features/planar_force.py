"""Stage 2: Whole-Vehicle Planar Force & Friction-Ellipse Yaw Dynamics library."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sindy_piml.features.base import CandidateLibrary, FeatureMatrix


class PlanarForceLibrary(CandidateLibrary):
    """Whole-vehicle planar force candidate library for [a_y, a_x, v, eta_P] -> r_dot.

    Combines transient lateral force buildup (a_y_dot, a_y_dot/v), sideslip
    acceleration (a_y - v*r), friction-ellipse combined-slip deficit (a_y*|a_x|),
    pneumatic saturation scaled by tire pressure factor eta_P, and yaw damping.
    """

    name = "planar_force"
    REQUIRED_COLS: tuple[str, ...] = ("ROTY", "G_LAT", "G_LON", "SPEED", "ETA_P", "R_DOT", "AY_DOT")

    FEATURE_NAMES: list[str] = [
        "r",
        "a_y",
        "a_y_dot",
        "a_y_dot/v",
        "a_y-v*r",
        "a_y*|a_x|",
        "a_y*(a_x^2+a_y^2)/eta_P^2",
        "r*|r|",
    ]

    LATEX_NAMES: list[str] = [
        r"r",
        r"a_y",
        r"\dot{a}_y",
        r"\frac{\dot{a}_y}{v}",
        r"a_y - v r",
        r"a_y |a_x|",
        r"\frac{a_y (a_x^2 + a_y^2)}{\eta_P^2}",
        r"r |r|",
    ]

    def build(self, df: pd.DataFrame) -> FeatureMatrix:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for PlanarForceLibrary: {missing}")

        r = df["ROTY"].to_numpy(dtype=float)
        ay = df["G_LAT"].to_numpy(dtype=float)
        ax = df["G_LON"].to_numpy(dtype=float)
        v = np.clip(df["SPEED"].to_numpy(dtype=float), 5.0, None)
        eta_p = np.clip(df["ETA_P"].to_numpy(dtype=float), 1e-3, None)
        r_dot = df["R_DOT"].to_numpy(dtype=float)
        ay_dot = df["AY_DOT"].to_numpy(dtype=float)

        theta = np.column_stack(
            [
                r,
                ay,
                ay_dot,
                ay_dot / v,
                ay - v * r,
                ay * np.abs(ax),
                ay * (ax**2 + ay**2) / (eta_p**2),
                r * np.abs(r),
            ]
        )

        return FeatureMatrix(
            theta=theta,
            target=r_dot,
            feature_names=list(self.FEATURE_NAMES),
            latex_names=list(self.LATEX_NAMES),
            target_name="r_dot",
        )
