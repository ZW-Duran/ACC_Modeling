"""Base abstractions for physics-informed SINDy candidate libraries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureMatrix:
    """Container for an evaluated SINDy candidate library matrix."""

    theta: np.ndarray
    target: np.ndarray
    feature_names: list[str]
    latex_names: list[str]
    target_name: str = "r_dot"


class CandidateLibrary(ABC):
    """Abstract base class for physics-informed candidate libraries."""

    name: str

    @abstractmethod
    def build(self, df: pd.DataFrame) -> FeatureMatrix:
        """Construct the candidate matrix Theta and target vector r_dot from curated data."""
