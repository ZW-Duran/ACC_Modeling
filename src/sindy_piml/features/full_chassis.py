"""Stage 4 (Future Goal): Full-Chassis & Setup Parameterized Dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sindy_piml.features.base import CandidateLibrary, FeatureMatrix


@dataclass
class ChassisSetupSpec:
    """Physical setup parameters for full-chassis SINDy identification."""

    front_ride_height_mm: float = 55.0
    rear_ride_height_mm: float = 67.0
    camber_front_deg: float = -3.5
    camber_rear_deg: float = -2.2
    toe_front_deg: float = -0.10
    toe_rear_deg: float = 0.15
    spring_rate_front_n_mm: float = 160.0
    spring_rate_rear_n_mm: float = 140.0
    arb_front_nm_deg: float = 0.0
    arb_rear_nm_deg: float = 0.0
    bump_damping_n_s_m: float = 0.0
    rebound_damping_n_s_m: float = 0.0


class FullChassisLibrary(CandidateLibrary):
    """Stage 4 candidate library interface coupling 4-corner kinematics, aero rake, and suspension setup."""

    name = "full_chassis"

    def __init__(self, setup: ChassisSetupSpec | None = None) -> None:
        self.setup = setup or ChassisSetupSpec()

    def build(self, df: pd.DataFrame) -> FeatureMatrix:
        raise NotImplementedError(
            "FullChassisLibrary (Stage 4) is reserved for full 4-wheel chassis & setup modeling."
        )
