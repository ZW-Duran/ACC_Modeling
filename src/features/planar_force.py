"""Stage 2: Discrete-Time Whole-Vehicle Planar Force & Friction-Ellipse Yaw Dynamics library."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import CandidateLibrary, FeatureMatrix


class PlanarForceLibrary(CandidateLibrary):
    """Discrete-time planar force library for r_{k+1} = F_A(r_k, a_y_k, a_y_{k-1}, a_x_k, eta_P_k).

    Solves the continuous-time phase-orthogonality issue between r and dr/dt by modeling
    the 1-step-ahead yaw rate transition r_{k+1} on 20 Hz telemetry (k = 1 .. N-2).
    Uses AX_SPEED (dv/dt from SPEED) when available, falling back to G_LON.
    """

    name = "planar_force"
    REQUIRED_COLS: tuple[str, ...] = ("ROTY", "G_LAT", "SPEED", "ETA_P")

    FEATURE_NAMES: list[str] = [
        "r_k",
        "a_y_k",
        "a_y_k-1",
        "a_y_k*a_x_k",
        "a_y_k*(a_x_k^2+a_y_k^2)/eta_P_k^2",
        "r_k*|r_k|",
    ]

    LATEX_NAMES: list[str] = [
        r"r_k",
        r"a_{y,k}",
        r"a_{y,k-1}",
        r"a_{y,k} a_{x,k}",
        r"\frac{a_{y,k}(a_{x,k}^2 + a_{y,k}^2)}{\eta_{P,k}^2}",
        r"r_k |r_k|",
    ]

    def build(self, df: pd.DataFrame) -> FeatureMatrix:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for PlanarForceLibrary: {missing}")
        if len(df) < 3:
            raise ValueError(f"Discrete-time PlanarForceLibrary requires at least 3 rows, got {len(df)}.")

        ax_col = "AX_SPEED" if "AX_SPEED" in df.columns else "G_LON"
        if ax_col not in df.columns:
            raise ValueError("Missing longitudinal acceleration column ('AX_SPEED' or 'G_LON').")

        r = df["ROTY"].to_numpy(dtype=float)
        ax = df[ax_col].to_numpy(dtype=float)
        ay = df["G_LAT"].to_numpy(dtype=float)
        eta_p = np.clip(df["ETA_P"].to_numpy(dtype=float), 1e-3, None)

        # Valid interior indices k = 1 .. N-2 (safe bounds for k-1 and k+1)
        r_k = r[1:-1]
        ay_k = ay[1:-1]
        ay_prev = ay[:-2]
        ax_k = ax[1:-1]
        eta_p_k = eta_p[1:-1]
        r_next = r[2:]

        theta = np.column_stack(
            [
                r_k,
                ay_k,
                ay_prev,
                ay_k * ax_k,
                ay_k * (ax_k**2 + ay_k**2) / (eta_p_k**2),
                r_k * np.abs(r_k),
            ]
        )

        return FeatureMatrix(
            theta=theta,
            target=r_next,
            feature_names=list(self.FEATURE_NAMES),
            latex_names=list(self.LATEX_NAMES),
            target_name="r_{k+1}",
        )
