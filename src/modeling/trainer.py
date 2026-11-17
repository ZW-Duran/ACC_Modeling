"""PySINDy sparse regression trainer for decoupled yaw-acceleration ODEs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pysindy as ps

from src.config import ModelingConfig
from src.features.base import FeatureMatrix


@dataclass
class FittedYawModel:
    """Container for a fitted sparse yaw-acceleration model."""

    coefficients: np.ndarray
    feature_names: list[str]
    latex_names: list[str]
    optimizer: ps.STLSQ

    def predict(self, theta: np.ndarray) -> np.ndarray:
        return theta @ self.coefficients

    def active_mask(self) -> np.ndarray:
        return np.abs(self.coefficients) > 0.0

    def format_equation(self, precision: int = 5) -> str:
        terms = []
        for coef, name in zip(self.coefficients, self.feature_names):
            if abs(coef) > 0.0:
                terms.append(f"{coef:+.{precision}f} * {name}")
        rhs = " ".join(terms) if terms else "0.0"
        return f"dr/dt = {rhs}"

    def to_dict(self) -> dict:
        return {
            "equation": self.format_equation(),
            "terms": {
                name: float(coef)
                for name, coef in zip(self.feature_names, self.coefficients)
            },
        }

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def fit_yaw_sindy(features: FeatureMatrix, cfg: ModelingConfig) -> FittedYawModel:
    """Fit sparse linear-in-parameters ODE r_dot = Theta @ xi using PySINDy STLSQ."""
    optimizer = ps.STLSQ(
        threshold=cfg.stlsq_threshold,
        alpha=cfg.stlsq_alpha,
        normalize_columns=cfg.normalize_columns,
    )
    optimizer.fit(features.theta, features.target.reshape(-1, 1))
    coef = np.asarray(optimizer.coef_, dtype=float).ravel()

    return FittedYawModel(
        coefficients=coef,
        feature_names=features.feature_names,
        latex_names=features.latex_names,
        optimizer=optimizer,
    )
