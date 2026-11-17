"""Moving-block bootstrap for time-series SINDy parameter confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pysindy as ps
from scipy import stats

from src.config import ModelingConfig
from src.features.base import FeatureMatrix


@dataclass
class TermConfidence:
    """Bootstrap confidence interval and t-test significance for a single candidate term."""

    name: str
    point_estimate: float
    boot_mean: float
    boot_std: float
    ci_95_low: float
    ci_95_high: float
    t_statistic: float
    p_value: float


@dataclass
class BootstrapSummary:
    """Block-bootstrap results across all candidate terms."""

    n_resamples: int
    block_size: int
    terms: list[TermConfidence]


def block_bootstrap_sindy(
    features: FeatureMatrix,
    point_coefficients: np.ndarray,
    modeling_cfg: ModelingConfig,
    block_size: int = 400,
    n_resamples: int = 500,
    seed: int = 42,
) -> BootstrapSummary:
    """Perform moving-block bootstrap to preserve temporal autocorrelation."""
    n_samples, n_features = features.theta.shape
    eff_block = max(1, min(block_size, n_samples))
    n_blocks = int(np.ceil(n_samples / eff_block))
    max_start = max(1, n_samples - eff_block + 1)

    rng = np.random.default_rng(seed)
    boot_coefs = np.zeros((n_resamples, n_features), dtype=float)

    for b in range(n_resamples):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + eff_block) for s in starts])[:n_samples]

        opt = ps.STLSQ(
            threshold=modeling_cfg.stlsq_threshold,
            alpha=modeling_cfg.stlsq_alpha,
            normalize_columns=modeling_cfg.normalize_columns,
        )
        opt.fit(features.theta[idx], features.target[idx].reshape(-1, 1))
        boot_coefs[b] = np.asarray(opt.coef_, dtype=float).ravel()

    df_dof = max(1, n_resamples - 1)
    term_results: list[TermConfidence] = []

    for j, name in enumerate(features.feature_names):
        samples_j = boot_coefs[:, j]
        b_mean = float(np.mean(samples_j))
        b_std = float(np.std(samples_j, ddof=1))
        ci_low, ci_high = np.percentile(samples_j, [2.5, 97.5])

        if b_std > 1e-12:
            t_stat = float(point_coefficients[j] / b_std)
            p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_dof))
        else:
            t_stat = 0.0 if abs(point_coefficients[j]) <= 1e-12 else float(np.sign(point_coefficients[j]) * 1e6)
            p_val = 1.0 if abs(point_coefficients[j]) <= 1e-12 else 0.0

        term_results.append(
            TermConfidence(
                name=name,
                point_estimate=float(point_coefficients[j]),
                boot_mean=b_mean,
                boot_std=b_std,
                ci_95_low=float(ci_low),
                ci_95_high=float(ci_high),
                t_statistic=t_stat,
                p_value=p_val,
            )
        )

    return BootstrapSummary(
        n_resamples=n_resamples,
        block_size=eff_block,
        terms=term_results,
    )
