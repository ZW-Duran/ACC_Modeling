"""Residual Gaussianity diagnostics and Type-I Omission threshold sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class ThresholdOmissionRow:
    """Comparison of theoretical Gaussian Type-I error vs. observed tail omissions."""

    z_threshold: float
    expected_omissions: float
    actual_omissions: int
    expected_pct: float
    actual_pct: float


@dataclass
class ResidualDiagnostics:
    """Summary of residual distribution, normality tests, and Type-I omission sweep."""

    n_samples: int
    mean: float
    std: float
    skewness: float
    excess_kurtosis: float
    ks_statistic: float
    ks_pvalue: float
    shapiro_statistic: float
    shapiro_pvalue: float
    within_3sigma_pct: float
    beyond_5sigma_count: int
    beyond_5sigma_pct: float
    omission_sweep: list[ThresholdOmissionRow] = field(default_factory=list)


def analyze_residuals(
    residuals: np.ndarray,
    z_scores: np.ndarray,
    shapiro_max_n: int = 5000,
    seed: int = 42,
    z_sweep: tuple[float, ...] = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
) -> ResidualDiagnostics:
    """Compute normality tests, Type-I omission table, and tail-anomaly statistics."""
    n = len(residuals)
    if n < 3:
        raise ValueError("Need at least 3 samples for residual diagnostics.")

    mu = float(np.mean(residuals))
    sigma = float(np.std(residuals, ddof=1))
    standardized = (residuals - mu) / max(sigma, 1e-12)

    ks_stat, ks_p = stats.kstest(standardized, "norm")

    if n > shapiro_max_n:
        rng = np.random.default_rng(seed)
        sub = rng.choice(standardized, size=shapiro_max_n, replace=False)
        sw_stat, sw_p = stats.shapiro(sub)
    else:
        sw_stat, sw_p = stats.shapiro(standardized)

    abs_z = np.abs(z_scores)
    within_3 = float(np.mean(abs_z <= 3.0) * 100.0)
    beyond_5_count = int(np.sum(abs_z > 5.0))
    beyond_5_pct = float(beyond_5_count / n * 100.0)

    sweep: list[ThresholdOmissionRow] = []
    for z_t in z_sweep:
        p_omit = float(2.0 * stats.norm.sf(z_t))
        exp_cnt = n * p_omit
        act_cnt = int(np.sum(abs_z > z_t))
        sweep.append(
            ThresholdOmissionRow(
                z_threshold=z_t,
                expected_omissions=exp_cnt,
                actual_omissions=act_cnt,
                expected_pct=p_omit * 100.0,
                actual_pct=(act_cnt / n) * 100.0,
            )
        )

    return ResidualDiagnostics(
        n_samples=n,
        mean=mu,
        std=sigma,
        skewness=float(stats.skew(residuals)),
        excess_kurtosis=float(stats.kurtosis(residuals, fisher=True)),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        shapiro_statistic=float(sw_stat),
        shapiro_pvalue=float(sw_p),
        within_3sigma_pct=within_3,
        beyond_5sigma_count=beyond_5_count,
        beyond_5sigma_pct=beyond_5_pct,
        omission_sweep=sweep,
    )
