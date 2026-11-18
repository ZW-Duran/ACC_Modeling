"""Command-line entry point for the ACC_Modeling vehicle dynamics pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

from src.config import load_config
from src.data.curation import curate_telemetry
from src.data.motec_loader import load_motec_csv
from src.features.planar_force import PlanarForceLibrary
from src.modeling.evaluator import evaluate_model
from src.modeling.trainer import fit_yaw_sindy
from src.stats.anomaly import map_anomalies_to_corners
from src.stats.bootstrap import block_bootstrap_sindy
from src.stats.diagnostics import analyze_residuals
from src.visualization.diagnostics import (
    plot_race_anomaly_trace,
    plot_residual_distribution,
    plot_yaw_acceleration_fit,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str | Path,
    car_path: str | Path | None = None,
    track_path: str | Path | None = None,
    curation_path: str | Path | None = None,
    modeling_path: str | Path | None = None,
    setup_path: str | Path | None = None,
) -> dict:
    """Execute end-to-end curation, SINDy identification, statistical validation, and verification.

    Parameters
    ----------
    config_path : str | Path
        Path to the experiment YAML manifest (e.g., ``configs/experiments/stage2_planar_force/...``).
    car_path, track_path, curation_path, modeling_path, setup_path : str | Path | None, optional
        Optional YAML paths overriding the corresponding entries in ``config_path``.

    Returns
    -------
    dict
        Structured experiment report (also saved to ``outputs/<experiment_name>/report.json``)
        containing ``equation``, ``clean_metrics``, ``bootstrap``, and (if ``race_csv`` is set)
        ``race_metrics`` with out-of-sample verification performance and corner anomalies.

    Notes
    -----
    - Writes diagnostic plots (``clean_yaw_fit.png``, ``clean_residuals_qq.png``) to
      ``outputs/<experiment_name>/`` and the fitted model JSON to ``models/<experiment_name>.json``.
    """
    cfg = load_config(
        config_path,
        car_path=car_path,
        track_path=track_path,
        curation_path=curation_path,
        modeling_path=modeling_path,
        setup_path=setup_path,
    )
    out_dir = Path("outputs") / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running experiment: %s (%s @ %s)", cfg.experiment_name, cfg.vehicle, cfg.circuit)

    # Phase 1: Clean practice data curation
    raw_clean, meta_clean = load_motec_csv(cfg.data.clean_csv)
    df_clean = curate_telemetry(
        raw_clean,
        meta_clean,
        cfg.curation,
        trim_mode=cfg.data.clean_trim_mode,
        speed_threshold_kmh=cfg.data.clean_speed_threshold_kmh,
    )

    # Phase 2: Candidate library + PySINDy fit
    if cfg.formulation != "planar_force":
        raise NotImplementedError(f"Formulation '{cfg.formulation}' is scheduled for a future stage.")

    library = PlanarForceLibrary()
    clean_feats = library.build(df_clean)
    model = fit_yaw_sindy(clean_feats, cfg.modeling)
    clean_eval = evaluate_model(model, clean_feats)

    logger.info("Identified ODE: %s", model.format_equation())
    logger.info("Clean Practice R^2=%.4f | RMSE=%.4f rad/s^2", clean_eval.r2, clean_eval.rmse)

    model.save_json(Path("models") / f"{cfg.experiment_name}.json")

    # Phase 3A: Residual diagnostics & Block Bootstrap on clean practice
    clean_diag = analyze_residuals(clean_eval.residuals, clean_eval.z_scores, seed=cfg.stats.random_seed)
    block_frames = max(1, int(round(cfg.stats.bootstrap_block_seconds * meta_clean.sample_rate_hz)))
    boot = block_bootstrap_sindy(
        clean_feats,
        model.coefficients,
        cfg.modeling,
        block_size=block_frames,
        n_resamples=cfg.stats.bootstrap_n_resamples,
        seed=cfg.stats.random_seed,
    )

    plot_yaw_acceleration_fit(
        df_clean,
        clean_eval,
        title=f"{cfg.vehicle} (Clean Practice)",
        save_path=out_dir / "clean_yaw_fit.png",
    )
    plot_residual_distribution(
        clean_eval,
        title=f"{cfg.vehicle} (Clean Practice)",
        save_path=out_dir / "clean_residuals_qq.png",
    )

    report: dict = {
        "experiment": cfg.experiment_name,
        "vehicle": cfg.vehicle,
        "circuit": cfg.circuit,
        "formulation": cfg.formulation,
        "equation": model.format_equation(),
        "clean_metrics": {
            "n_samples": len(df_clean),
            "r2": clean_eval.r2,
            "rmse": clean_eval.rmse,
            "mae": clean_eval.mae,
            "diagnostics": asdict(clean_diag),
        },
        "bootstrap": {
            "n_resamples": boot.n_resamples,
            "block_size_frames": boot.block_size,
            "terms": [asdict(t) for t in boot.terms],
        },
    }

    # Phase 3B: Race stress-test (if configured)
    if cfg.data.race_csv:
        raw_race, meta_race = load_motec_csv(cfg.data.race_csv)
        df_race = curate_telemetry(
            raw_race,
            meta_race,
            cfg.curation,
            trim_mode=cfg.data.race_trim_mode,
            speed_threshold_kmh=cfg.data.clean_speed_threshold_kmh,
        )
        race_feats = library.build(df_race)
        race_eval = evaluate_model(
            model,
            race_feats,
            ref_mean=clean_eval.residual_mean,
            ref_std=clean_eval.residual_std,
        )
        race_diag = analyze_residuals(race_eval.residuals, race_eval.z_scores, seed=cfg.stats.random_seed)
        anomalies = map_anomalies_to_corners(
            df_race,
            race_eval.z_scores,
            cfg.circuit_corners,
            threshold_z=cfg.stats.z_anomaly_threshold,
        )

        logger.info(
            "Race Stress-Test R^2=%.4f | Within ±3σ=%.2f%% | >%.0fσ=%d",
            race_eval.r2,
            race_diag.within_3sigma_pct,
            cfg.stats.z_anomaly_threshold,
            anomalies.total_anomalies,
        )

        plot_race_anomaly_trace(
            df_race,
            race_eval.z_scores,
            cfg.stats.z_anomaly_threshold,
            title=f"{cfg.vehicle} (Race)",
            save_path=out_dir / "race_anomaly_trace.png",
        )
        anomalies.anomalies_df.to_csv(out_dir / "race_anomalies.csv", index=False)

        report["race_metrics"] = {
            "n_samples": len(df_race),
            "r2": race_eval.r2,
            "rmse": race_eval.rmse,
            "mae": race_eval.mae,
            "diagnostics": asdict(race_diag),
            "corner_anomalies": anomalies.corner_counts,
        }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Artifacts saved to %s", out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACC_Modeling vehicle dynamics identification.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/stage2_planar_force/barcelona__audi_r8_gt3_evo2__r8_A.yaml"),
        help="Path to experiment YAML configuration.",
    )
    parser.add_argument("--car", type=Path, default=None, help="Optional override vehicle.yaml.")
    parser.add_argument("--setup", type=Path, default=None, help="Optional override per-car setup YAML.")
    parser.add_argument("--track", type=Path, default=None, help="Optional override track YAML.")
    parser.add_argument("--curation", type=Path, default=None, help="Optional override curation YAML.")
    parser.add_argument("--modeling", type=Path, default=None, help="Optional override modeling YAML.")
    args = parser.parse_args()
    run_pipeline(
        args.config,
        car_path=args.car,
        track_path=args.track,
        curation_path=args.curation,
        modeling_path=args.modeling,
        setup_path=args.setup,
    )


if __name__ == "__main__":
    main()
