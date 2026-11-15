"""Stage 3 (Next Milestone): Single-Wheel & Suspension Contact-Patch Modeling (LF/RF/LR/RR)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from sindy_piml.config import CarConfig
from sindy_piml.features.base import CandidateLibrary, FeatureMatrix

WheelCorner = Literal["LF", "RF", "LR", "RR"]
GRAVITY = 9.80665


def estimate_corner_normal_load(
    df: pd.DataFrame,
    corner: WheelCorner,
    car: CarConfig,
    aero_cl_a: float = 0.0,
    air_density: float = 1.225,
) -> np.ndarray:
    """Estimate quasi-static + pitch/roll + aero normal force F_z (N) at a single wheel corner.

    Sign conventions (ACC MoTeC):
      - G_LON > 0: forward acceleration -> longitudinal load transfers rearward.
      - G_LAT > 0: cornering lateral acceleration -> lateral load transfers outward.
      - Aero downforce proxy scales with 0.5 * rho * Cl*A * v^2 distributed by weight balance.
    """
    is_front = corner in ("LF", "RF")
    is_left = corner in ("LF", "LR")

    axle_frac = car.front_weight_dist if is_front else (1.0 - car.front_weight_dist)
    track_w = car.track_width_front_m if is_front else car.track_width_rear_m

    fz_static = 0.5 * axle_frac * car.mass_kg * GRAVITY

    ax = df["G_LON"].to_numpy(dtype=float)
    ay = df["G_LAT"].to_numpy(dtype=float)
    v = df["SPEED"].to_numpy(dtype=float)

    # Pitch load transfer (front loses load under acceleration, gains under braking)
    pitch_sign = -1.0 if is_front else 1.0
    delta_fz_long = pitch_sign * 0.5 * car.mass_kg * ax * (car.cg_height_m / max(car.wheelbase_m, 1e-3))

    # Roll load transfer (proportional to axle weight share)
    roll_sign = -1.0 if is_left else 1.0
    delta_fz_lat = roll_sign * axle_frac * car.mass_kg * ay * (car.cg_height_m / max(track_w, 1e-3))

    # Aerodynamic downforce proxy
    fz_aero = 0.5 * axle_frac * (0.5 * air_density * aero_cl_a * (v**2))

    return np.clip(fz_static + delta_fz_long + delta_fz_lat + fz_aero, 50.0, None)


class WheelCornerLibrary(CandidateLibrary):
    """Stage 3 candidate library focusing on single-wheel grip, suspension travel, and load transfer.

    This class establishes the contract and channel schema for per-wheel modeling
    (`SUS_TRAVEL_{corner}`, `BUMPSTOP_FORCE_{corner}`, `WHEEL_SPEED_{corner}`,
    `TYRE_PRESS_{corner}`, `TYRE_TAIR_{corner}`), coupled with normal load F_z(t)
    and aero speed^2 pressure effects.
    """

    name = "wheel_corner"

    def __init__(self, corner: WheelCorner = "LF", car: CarConfig | None = None) -> None:
        self.corner = corner
        self.car = car or CarConfig()

    @property
    def required_channels(self) -> tuple[str, ...]:
        c = self.corner
        return (
            "ROTY",
            "G_LAT",
            "G_LON",
            "SPEED",
            "R_DOT",
            f"SUS_TRAVEL_{c}",
            f"BUMPSTOP_FORCE_{c}",
            f"WHEEL_SPEED_{c}",
            f"TYRE_PRESS_{c}",
            f"TYRE_TAIR_{c}",
        )

    def build(self, df: pd.DataFrame) -> FeatureMatrix:
        raise NotImplementedError(
            "WheelCornerLibrary (Stage 3) scaffolded; per-wheel SINDy equations are the next milestone."
        )
