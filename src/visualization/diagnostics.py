"""Publication-grade diagnostic plots for yaw-acceleration SINDy models."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.modeling.evaluator import EvaluationResult


def plot_yaw_acceleration_fit(
    df: pd.DataFrame,
    eval_res: EvaluationResult,
    title: str,
    save_path: str | Path,
    window_seconds: float = 30.0,
) -> Path:
    """Plot observed vs. SINDy-predicted yaw acceleration over a representative window."""
    t = df["Time"].to_numpy(dtype=float)
    t_rel = t - t[0]
    mask = t_rel <= window_seconds

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax1.plot(t_rel[mask], eval_res.y_true[mask], "k-", lw=1.4, label=r"Observed $\dot{r}_{\mathrm{obs}}$")
    ax1.plot(t_rel[mask], eval_res.y_pred[mask], "r--", lw=1.2, alpha=0.9, label=r"SINDy $\dot{r}_{\mathrm{pred}}$")
    ax1.set_ylabel(r"Yaw Accel $\dot{r}$ ($\mathrm{rad/s^2}$)")
    ax1.set_title(f"{title} | $R^2={eval_res.r2:.4f}$, RMSE={eval_res.rmse:.4f}")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(t_rel[mask], eval_res.residuals[mask], color="#1f77b4", lw=1.0, label="Residual $e(t)$")
    ax2.axhline(0.0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel(r"Residual ($\mathrm{rad/s^2}$)")
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="upper right")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_residual_distribution(
    eval_res: EvaluationResult,
    title: str,
    save_path: str | Path,
) -> Path:
    """Plot residual histogram with Gaussian overlay alongside a normal Q-Q plot."""
    res = eval_res.residuals
    mu, sigma = eval_res.residual_mean, max(eval_res.residual_std, 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(res, bins=100, density=True, alpha=0.65, color="#2b6cb0", label="Empirical Residuals")
    x_grid = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 400)
    ax1.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), "r-", lw=1.8, label=r"$\mathcal{N}(\mu_0, \sigma_0^2)$")
    ax1.axvline(mu + 5 * sigma, color="darkred", ls="--", lw=1.2, label=r"$\pm 5\sigma_0$")
    ax1.axvline(mu - 5 * sigma, color="darkred", ls="--", lw=1.2)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, None)
    ax1.set_title(f"{title} — Residual Distribution (Log Scale)")
    ax1.set_xlabel(r"Residual $e(t)$ ($\mathrm{rad/s^2}$)")
    ax1.set_ylabel("Density")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend()

    standardized = (res - mu) / sigma
    if len(standardized) > 10000:
        rng = np.random.default_rng(42)
        standardized = rng.choice(standardized, size=10000, replace=False)
    stats.probplot(standardized, dist="norm", plot=ax2)
    ax2.set_title("Normal Q-Q Plot")
    ax2.grid(True, ls=":", alpha=0.5)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_race_anomaly_trace(
    df: pd.DataFrame,
    z_scores: np.ndarray,
    z_thresh: float,
    title: str,
    save_path: str | Path,
) -> Path:
    """Plot standardized residual Z-score trace during race stress-testing."""
    t_rel = df["Time"].to_numpy(dtype=float) - float(df["Time"].iloc[0])

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t_rel, z_scores, lw=0.7, color="#4a5568", alpha=0.85, label=r"$Z(t)$")
    ax.axhline(3.0, color="orange", ls="--", lw=1.0, label=r"$\pm 3\sigma$")
    ax.axhline(-3.0, color="orange", ls="--", lw=1.0)
    ax.axhline(z_thresh, color="red", ls=":", lw=1.2, label=rf"$\pm {z_thresh:.0f}\sigma$ Anomaly")
    ax.axhline(-z_thresh, color="red", ls=":", lw=1.2)

    anom = np.abs(z_scores) > z_thresh
    if np.any(anom):
        ax.scatter(t_rel[anom], z_scores[anom], color="red", s=12, zorder=5, label=f"Outliers (n={int(np.sum(anom))})")

    ax.set_title(f"{title} — Race Stress-Test Standardized Residuals")
    ax.set_xlabel("Race Elapsed Time (s)")
    ax.set_ylabel("Standardized Residual $Z(t)$")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
