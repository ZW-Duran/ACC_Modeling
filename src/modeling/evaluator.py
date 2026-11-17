"""Model evaluation metrics and residual computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.base import FeatureMatrix
from src.modeling.trainer import FittedYawModel


@dataclass
class EvaluationResult:
    """Prediction, residuals, standardized Z-scores, and goodness-of-fit metrics."""

    y_true: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray
    z_scores: np.ndarray
    r2: float
    rmse: float
    mae: float
    residual_mean: float
    residual_std: float


def evaluate_model(
    model: FittedYawModel,
    features: FeatureMatrix,
    ref_mean: float | None = None,
    ref_std: float | None = None,
) -> EvaluationResult:
    """Evaluate a fitted yaw model on a FeatureMatrix and compute residuals / Z-scores.

    When evaluating out-of-distribution (e.g., race telemetry), pass `ref_mean` and
    `ref_std` from the clean practice baseline so Z(t) = (e(t) - mu_0) / sigma_0.
    """
    y_true = features.target
    y_pred = model.predict(features.theta)
    residuals = y_true - y_pred

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    mu = float(np.mean(residuals)) if ref_mean is None else ref_mean
    sigma = float(np.std(residuals, ddof=1)) if ref_std is None else ref_std
    z_scores = (residuals - mu) / max(sigma, 1e-12)

    return EvaluationResult(
        y_true=y_true,
        y_pred=y_pred,
        residuals=residuals,
        z_scores=z_scores,
        r2=r2,
        rmse=rmse,
        mae=mae,
        residual_mean=float(np.mean(residuals)),
        residual_std=float(np.std(residuals, ddof=1)),
    )
