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
    window_seconds: float | None = None,
) -> Path:
    """Plot observed vs. SINDy-predicted yaw dynamics over the full session or a specified window."""
    if len(df) == len(eval_res.y_true) + 2:
        # Target r_{k+1} = ROTY[2:] lives at timestamp Time[2:]
        df = df.iloc[2:].reset_index(drop=True)
    t = df["Time"].to_numpy(dtype=float)
    t_rel = t - t[0]
    mask = np.ones(len(t_rel), dtype=bool) if window_seconds is None else (t_rel <= window_seconds)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax1.plot(t_rel[mask], eval_res.y_true[mask], "k-", lw=1.4, label=r"Observed $r_{k+1,\mathrm{obs}}$")
    ax1.plot(t_rel[mask], eval_res.y_pred[mask], "r--", lw=1.2, alpha=0.9, label=r"SINDy $r_{k+1,\mathrm{pred}}$")
    ax1.set_ylabel(r"Next-Step Yaw Rate $r_{k+1}$ ($\mathrm{rad/s}$)")
    ax1.set_title(f"{title} | $R^2={eval_res.r2:.4f}$, RMSE={eval_res.rmse:.4f}")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(t_rel[mask], eval_res.residuals[mask], color="#1f77b4", lw=1.0, label="Residual $e(t)$")
    ax2.axhline(0.0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel(r"Residual ($\mathrm{rad/s}$)")
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
    """Plot empirical residual count histogram with non-parametric KDE, Laplace, and Student-t fits."""
    res = eval_res.residuals
    n = len(res)
    mu, sigma = eval_res.residual_mean, max(eval_res.residual_std, 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    counts, bin_edges, _ = ax1.hist(
        res, bins=80, alpha=0.65, color="#2b6cb0", edgecolor="white", linewidth=0.4, label="Empirical Count"
    )
    bin_width = float(bin_edges[1] - bin_edges[0])
    scale_to_count = n * bin_width

    x_grid = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 400)

    # Fit non-Gaussian heavy-tailed distributions & non-parametric KDE directly
    loc_l, scale_l = stats.laplace.fit(res)
    df_t, loc_t, scale_t = stats.t.fit(res)
    kde = stats.gaussian_kde(res)

    ax1.plot(x_grid, kde(x_grid) * scale_to_count, "k-", lw=1.8, label="Empirical KDE")
    ax1.plot(
        x_grid,
        stats.laplace.pdf(x_grid, loc=loc_l, scale=scale_l) * scale_to_count,
        "r--",
        lw=1.8,
        label=rf"Laplace Fit ($b={scale_l:.4f}$)",
    )
    ax1.plot(
        x_grid,
        stats.t.pdf(x_grid, df=df_t, loc=loc_t, scale=scale_t) * scale_to_count,
        color="#d97706",
        ls="-.",
        lw=1.6,
        label=rf"Student-$t$ Fit ($\nu={df_t:.2f}$)",
    )
    ax1.axvline(mu + 5 * sigma, color="darkred", ls=":", lw=1.1, label=r"$\pm 5\sigma$")
    ax1.axvline(mu - 5 * sigma, color="darkred", ls=":", lw=1.1)
    ax1.set_title(f"{title} — Residual Count & Fitted Distributions")
    ax1.set_xlabel(r"Residual $e(t)$ ($\mathrm{rad/s}$)")
    ax1.set_ylabel("Count (Samples)")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend()

    sample = res
    if len(sample) > 10000:
        rng = np.random.default_rng(42)
        sample = rng.choice(sample, size=10000, replace=False)
    stats.probplot(sample, dist=stats.laplace, sparams=(loc_l, scale_l), plot=ax2)
    ax2.set_title(rf"Laplace Q-Q Plot ($b={scale_l:.4f}$)")
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
    """Plot standardized residual Z-score trace during verification / stress-testing."""
    if len(df) == len(z_scores) + 2:
        df = df.iloc[2:].reset_index(drop=True)
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


def plot_finite_diff_yaw_accel(
    df: pd.DataFrame,
    eval_res: EvaluationResult,
    title: str,
    save_path: str | Path,
    dt: float | None = None,
) -> dict[str, float | int | Path]:
    """Compare finite-difference yaw acceleration ``(r_{k+1} - r_k)/dt`` against raw ``ROTY`` derivative.

    Strips out the trivial ``1 * r_k`` persistence in 1-step-ahead ``r_{k+1}`` models
    to expose true dynamical accuracy and phase lag (in frames and milliseconds).
    """
    t = df["Time"].to_numpy(dtype=float)
    step_dt = float(np.median(np.diff(t))) if dt is None else float(dt)
    r_all = df["ROTY"].to_numpy(dtype=float)

    n_pred = len(eval_res.y_true)
    if len(df) == n_pred + 2:
        # Stage 2 discrete library: k = 1 .. N-2 -> r_k = ROTY[1:-1], t_{k+1} = Time[2:]
        r_k = r_all[1:-1]
        t_rel = t[2:] - t[0]
    elif len(df) == n_pred + 1:
        # Stage 3 Euler forward: k = 0 .. N-2 -> r_k = ROTY[:-1], t_{k+1} = Time[1:]
        r_k = r_all[:-1]
        t_rel = t[1:] - t[0]
    else:
        raise ValueError(f"Length mismatch: len(df)={len(df)} vs len(y_true)={n_pred}")

    rd_obs = (eval_res.y_true - r_k) / step_dt
    rd_pred = (eval_res.y_pred - r_k) / step_dt
    res_rd = rd_obs - rd_pred

    ss_res = float(np.sum(res_rd**2))
    ss_tot = float(np.sum((rd_obs - np.mean(rd_obs)) ** 2))
    r2_rd = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse_rd = float(np.sqrt(np.mean(res_rd**2)))

    lags = np.arange(-10, 11)
    pn = (rd_pred - np.mean(rd_pred)) / (np.std(rd_pred) + 1e-12)
    tn = (rd_obs - np.mean(rd_obs)) / (np.std(rd_obs) + 1e-12)
    ccs = [float(np.mean(np.roll(pn, -lag) * tn)) for lag in lags]
    best_lag = int(lags[int(np.argmax(ccs))])
    lag_ms = best_lag * step_dt * 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.plot(t_rel, rd_obs, "k-", lw=1.3, label=r"Raw $\mathrm{ROTY}$ Diff $(r_{k+1}-r_k)/\Delta t$")
    ax1.plot(t_rel, rd_pred, "r--", lw=1.1, alpha=0.9, label=r"Model Diff $(\hat{r}_{k+1}-r_k)/\Delta t$")
    ax1.set_ylabel(r"Finite-Diff $\dot{r}$ ($\mathrm{rad/s^2}$)")
    ax1.set_title(f"{title} | $\\dot{{r}}$ $R^2={r2_rd:.4f}$, RMSE={rmse_rd:.4f}, lag={best_lag:+d} frames ({lag_ms:+.0f} ms)")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(t_rel, res_rd, color="#1f77b4", lw=1.0, label=r"$\dot{r}$ Residual")
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

    return {
        "r2_rd": r2_rd,
        "rmse_rd": rmse_rd,
        "lag_frames": best_lag,
        "lag_ms": lag_ms,
        "save_path": out,
    }
