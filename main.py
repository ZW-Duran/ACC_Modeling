"""ACC_Modeling CLI: Deterministic Physics Diagnostics (diagnose) & Counterfactual Setup Simulation (simulate)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CurationConfig, load_car_config, load_setup_file, load_track_config
from src.data.curation import curate_telemetry
from src.data.motec_loader import load_motec_csv
from src.modeling.stage4_diagnostic import (
    STAGE4_CHANNELS,
    build_ai_handoff_payload,
    evaluate_7dof_lap_state,
    run_counterfactual_simulation,
    save_diagnose_plots,
    save_simulate_plots,
)


def _load_and_curate(csv_path: str | Path, car_p_opt: float, car_sigma_p: float):
    raw, meta = load_motec_csv(csv_path, channels=STAGE4_CHANNELS)
    # Keep ~0.25s Savitzky-Golay smoothing window across 20 Hz (window=5) and 200 Hz (window=51) exports
    sg_win = max(5, int(round(0.25 * meta.sample_rate_hz)) | 1) if meta.sample_rate_hz > 30.0 else 5
    curation_cfg = CurationConfig(savgol_window=sg_win, p_opt_psi=car_p_opt, sigma_p_psi=car_sigma_p)
    df = curate_telemetry(raw, meta, curation_cfg, trim_mode="none")
    return df


def _cmd_diagnose(args: argparse.Namespace) -> None:
    car = load_car_config(args.vehicle)
    setup = load_setup_file(args.setup)
    track = load_track_config(args.track)
    df = _load_and_curate(args.csv, car.tire_p_opt_psi, car.tire_sigma_p_psi)

    ui_tyres = None
    if args.ui_tyres:
        with open(args.ui_tyres, "r", encoding="utf-8") as f:
            ui_tyres = json.load(f)

    state = evaluate_7dof_lap_state(df, car, setup, mu_0=None, use_predicted_travel=False)
    payload = build_ai_handoff_payload(args.csv, df, car, setup, track, state, ui_tyres=ui_tyres)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    plots = save_diagnose_plots(state, payload["corners"], payload["vehicle_and_setup_baseline"], out_path)
    print(f"[diagnose] Exported {out_path} ({len(payload['corners'])} corners)")
    for p in plots:
        print(f"[diagnose] Saved plot: {p}")


def _cmd_simulate(args: argparse.Namespace) -> None:
    car = load_car_config(args.vehicle)
    base_setup = load_setup_file(args.base_setup)
    cand_setup = load_setup_file(args.candidate_setup)
    track = load_track_config(args.track)
    df = _load_and_curate(args.baseline_csv, car.tire_p_opt_psi, car.tire_sigma_p_psi)

    calib_dir = Path("data/calibration") / car.car_id
    report, state_v1, state_v2 = run_counterfactual_simulation(
        df, car, base_setup, cand_setup, track, calib_dir=calib_dir if calib_dir.exists() else None
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_path = save_simulate_plots(state_v1, state_v2, out_path)
    print(f"[simulate] Exported {out_path} (valid_bounds={report['bounds_validation']['valid']})")
    print(f"[simulate] Saved plot: {plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACC_Modeling Stage 4 Closed-Loop Diagnostic & Simulation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="Extract 7-DOF telemetry diagnostics and AI handoff JSON")
    p_diag.add_argument("--csv", required=True, help="MoTeC telemetry CSV file path")
    p_diag.add_argument("--vehicle", required=True, help="Vehicle calibration YAML path")
    p_diag.add_argument("--setup", required=True, help="Baseline garage setup YAML/JSON path")
    p_diag.add_argument("--track", required=True, help="Track metadata YAML path")
    p_diag.add_argument("--out", default="outputs/stage4_diagnosis/ai_handoff.json", help="Output JSON path")
    p_diag.add_argument("--ui-tyres", default=None, help="Optional UI tyre IMO temperature/wear JSON path")
    p_diag.set_defaults(func=_cmd_diagnose)

    p_sim = sub.add_parser("simulate", help="Counterfactual 7-DOF simulation comparing v1 baseline vs v2 candidate")
    p_sim.add_argument("--baseline-csv", required=True, help="Baseline MoTeC telemetry CSV path")
    p_sim.add_argument("--vehicle", required=True, help="Vehicle calibration YAML path")
    p_sim.add_argument("--base-setup", required=True, help="Baseline setup YAML/JSON path")
    p_sim.add_argument("--candidate-setup", required=True, help="Candidate setup JSON/YAML path from external AI")
    p_sim.add_argument("--track", required=True, help="Track metadata YAML path")
    p_sim.add_argument("--out", default="outputs/stage4_diagnosis/simulation_diff_report.json", help="Output diff JSON path")
    p_sim.set_defaults(func=_cmd_simulate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
