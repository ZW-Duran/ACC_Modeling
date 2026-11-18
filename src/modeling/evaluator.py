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
    """Evaluate a fitted yaw model on a FeatureMatrix and compute residuals & standardized Z-scores.

    Parameters
    ----------
    model : FittedYawModel
        Trained SINDy model from :func:`fit_yaw_sindy`.
    features : FeatureMatrix
        Evaluation feature matrix and target array (units: ``rad/s`` for discrete-time
        ``r_{k+1}`` or ``rad/s^2`` for continuous-time ``r_dot``).
    ref_mean : float | None, optional
        Training baseline residual mean ``mu_0``. Pass when evaluating out-of-sample
        verification/race data so Z-scores are referenced to the training distribution.
    ref_std : float | None, optional
        Training baseline residual std ``sigma_0``.

    Returns
    -------
    EvaluationResult
        Contains ``y_true``, ``y_pred``, ``residuals`` (``y_true - y_pred``),
        ``z_scores``, ``r2``, ``rmse``, ``mae``, ``residual_mean``, and ``residual_std``.
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
