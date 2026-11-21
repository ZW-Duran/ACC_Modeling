import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.config import load_car_config, load_setup_file, load_track_config
from src.data.curation import curate_telemetry
from src.data.motec_loader import load_motec_csv
from src.modeling.stage4_diagnostic import (
    STAGE4_CHANNELS,
    build_ai_handoff_payload,
    compute_vehicle_coupled_baseline,
    evaluate_7dof_lap_state,
    run_counterfactual_simulation,
    save_diagnose_plots,
    save_simulate_plots,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stage4_track_tags_and_vehicle_bounds():
    track = load_track_config(ROOT / "configs/tracks/barcelona_catalunya.yaml")
    assert len(track.corners) == 8
    by_name = {c.name: c for c in track.corners}
    assert by_name["T3 Renault"].speed_class == "high"
    assert by_name["T3 Renault"].aero_sensitivity == "high"
    assert by_name["T1-T2 Elf"].kerb_usage == "high"
    assert by_name["T5 Seat"].speed_class == "low"

    car = load_car_config(ROOT / "configs/cars/audi_r8_gt3_evo2/vehicle.yaml")
    assert "suspension" in car.setup_bounds
    assert "bumpstop_range_clicks" in car.setup_bounds["suspension"]
    assert "tyres_and_alignment" in car.setup_bounds


@pytest.mark.skipif(
    not (ROOT / "data/training_Barcelona_GT3.csv").exists(),
    reason="Local GT3 track telemetry not present.",
)
def test_stage4_diagnose_and_simulate_pipeline(tmp_path):
    car = load_car_config(ROOT / "configs/cars/audi_r8_gt3_evo2/vehicle.yaml")
    setup_v1 = load_setup_file(ROOT / "configs/cars/audi_r8_gt3_evo2/setups/Barcelona/r8_A.yaml")
    track = load_track_config(ROOT / "configs/tracks/barcelona_catalunya.yaml")

    baseline = compute_vehicle_coupled_baseline(car, setup_v1)
    assert 40.0 < baseline["static_lltd_front_pct"] < 65.0
    assert 0.1 < baseline["zeta_slow_front"] < 2.0
    assert 0.1 < baseline["zeta_slow_rear"] < 2.0

    # Test end-to-end `diagnose` CLI with --ui-tyres
    ui_tyres_path = tmp_path / "ui_tyres.json"
    ui_tyres_path.write_text(json.dumps({"LF": {"IMO": [88, 92, 95], "wear_pct": 98.5}}), encoding="utf-8")
    handoff_path = tmp_path / "ai_handoff.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "diagnose",
            "--csv",
            str(ROOT / "data/training_Barcelona_GT3.csv"),
            "--vehicle",
            str(ROOT / "configs/cars/audi_r8_gt3_evo2/vehicle.yaml"),
            "--setup",
            str(ROOT / "configs/cars/audi_r8_gt3_evo2/setups/Barcelona/r8_A.yaml"),
            "--track",
            str(ROOT / "configs/tracks/barcelona_catalunya.yaml"),
            "--ui-tyres",
            str(ui_tyres_path),
            "--out",
            str(handoff_path),
        ],
        cwd=ROOT,
        check=True,
    )

    assert handoff_path.exists()
    assert (tmp_path / "ai_handoff_quadrant_utilization.png").exists()
    assert (tmp_path / "ai_handoff_lltd_bumpstop.png").exists()

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert handoff["ui_tyre_diagnostics"]["LF"]["wear_pct"] == 98.5
    assert "four_quadrant_summary" in handoff
    assert len(handoff["four_quadrant_summary"]["corners"]) == 8
    assert "zeta_slow_front" in handoff["vehicle_and_setup_baseline"]
    assert "physics_model_spec" in handoff
    assert "constitutive_equations" in handoff["physics_model_spec"]
    assert "valid_setup_bounds" in handoff["physics_model_spec"]
    assert "expected_ai_response_schema" in handoff
    assert len(handoff["corners"]) == 8

    c0 = handoff["corners"][0]
    assert set(c0["grip_utilization"]["phases"].keys()) == {"entry", "apex", "exit"}
    assert c0["grip_utilization"]["state_label"] in {
        "FRONT_SATURATED",
        "REAR_SATURATED",
        "BALANCED_LIMIT",
        "DRIVER_MARGIN",
    }
    assert "bumpstop_min_margin_mm" in c0["nonlinear_margins"]
    assert "front_toe_turn_in_gain" in c0["alignment_proxies"]
    assert "rear_toe_high_speed_yaw_osc_rms_rad_s" in c0["alignment_proxies"]

    # Test end-to-end `simulate` CLI with multi-parameter candidate JSON
    cand_path = tmp_path / "setup_v2.json"
    cand_setup = {
        "setup_name": "barcelona_R8_A_v2_candidate",
        "tyres_and_alignment": {
            "cold_pressure_psi": [25.0, 25.5, 24.3, 25.4],
            "camber_deg": [-3.8, -3.8, -3.3, -3.3],
        },
        "suspension": {
            "bumpstop_range_clicks": [7, 7, 14, 14],
            "arb_clicks": [3, 2],
        },
        "aero_clicks": {"rear_wing": 4, "brake_ducts": [2, 3]},
        "ride_height_mm": [67, 54],
    }
    cand_path.write_text(json.dumps(cand_setup), encoding="utf-8")
    sim_out = tmp_path / "simulation_diff_report.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "simulate",
            "--baseline-csv",
            str(ROOT / "data/training_Barcelona_GT3.csv"),
            "--vehicle",
            str(ROOT / "configs/cars/audi_r8_gt3_evo2/vehicle.yaml"),
            "--base-setup",
            str(ROOT / "configs/cars/audi_r8_gt3_evo2/setups/Barcelona/r8_A.yaml"),
            "--candidate-setup",
            str(cand_path),
            "--track",
            str(ROOT / "configs/tracks/barcelona_catalunya.yaml"),
            "--out",
            str(sim_out),
        ],
        cwd=ROOT,
        check=True,
    )

    assert sim_out.exists()
    assert (tmp_path / "simulation_diff_report_comparison.png").exists()
    sim = json.loads(sim_out.read_text(encoding="utf-8"))
    assert sim["bounds_validation"]["valid"] is True
    assert sim["baseline_diff"]["static_lltd_front_pct"]["delta"] < 0.0
    assert sim["baseline_diff"]["main_straight_end_speed_kmh"]["delta"] > 0.0

    t1_diff = sim["corner_diffs"][0]
    assert t1_diff["front_bumpstop_min_margin_mm"]["delta"] > 0.5
    assert t1_diff["lltd_instant_peak_pct"]["delta"] < 0.0
