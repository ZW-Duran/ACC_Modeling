from pathlib import Path

import pytest

from src.calibration.blackbox import (
    _load_calib_csv,
    calibrate_aero,
    calibrate_arb_stiffness,
    calibrate_bumpstop_range,
    calibrate_dampers,
    run_gt3_blackbox_calibration,
    verify_setup_consistency,
)

CALIB_DIR = Path(__file__).resolve().parents[1] / "data/calibration"


@pytest.mark.skipif(not CALIB_DIR.exists(), reason="Local telemetry calibration directory not present.")
def test_setup_consistency_and_test2_match():
    res = verify_setup_consistency(CALIB_DIR)
    assert res["test2_matches_baseline_chassis"] is True
    assert res["sessions"]["test1"]["vmax_kmh"] > res["sessions"]["test2"]["vmax_kmh"]
    assert res["sessions"]["test2"]["vmax_kmh"] > res["sessions"]["baseline"]["vmax_kmh"]

    rep = run_gt3_blackbox_calibration(CALIB_DIR)
    assert rep["calibration_mass_kg"] == 1292.5
    assert rep["hotlap_fuel_liters"] == 10.0
    assert rep["stationary_datum_mm"] == {"front": 15.0, "rear": 17.0}


@pytest.mark.skipif(not CALIB_DIR.exists(), reason="Local telemetry calibration directory not present.")
def test_bumpstop_range_calibration():
    base_df = _load_calib_csv(CALIB_DIR / "baseline.csv")
    t5_df = _load_calib_csv(CALIB_DIR / "test5.csv")
    bs = calibrate_bumpstop_range(base_df, t5_df)
    assert bs["front_base_mm"] == 17.0
    assert bs["front_per_click_mm"] == 1.0
    assert bs["rear_base_mm"] == 24.0
    assert bs["rear_per_click_mm"] == 1.0


@pytest.mark.skipif(not CALIB_DIR.exists(), reason="Local telemetry calibration directory not present.")
def test_aero_calibration():
    aero = calibrate_aero(CALIB_DIR, mass_kg=1292.5)
    assert aero["aero_baseline"]["base_cl"] > 1.0
    assert 0.40 < aero["aero_baseline"]["base_aero_front_dist"] < 0.48
    clicks = aero["aero_clicks"]
    assert clicks["rear_wing_dcl_per_click"] > 0.0
    assert clicks["rear_wing_dcd_per_click"] > 0.0
    assert clicks["front_brake_duct_dcd_per_click"] == 0.0035
    assert clicks["rear_brake_duct_dcd_per_click"] == 0.0018
    assert clicks["front_brake_duct_dcd_per_click"] > clicks["rear_brake_duct_dcd_per_click"] > 0.0


@pytest.mark.skipif(not CALIB_DIR.exists(), reason="Local telemetry calibration directory not present.")
def test_arb_and_damper_calibration():
    arb = calibrate_arb_stiffness(CALIB_DIR)
    assert arb["front_base"] > 0 and arb["front_per_click"] > 0
    assert arb["rear_base"] > 0 and arb["rear_per_click"] > 0

    damp = calibrate_dampers(CALIB_DIR)
    for key in (
        "bump_slow_base",
        "bump_slow_per_click",
        "bump_fast_base",
        "bump_fast_per_click",
        "rebound_slow_base",
        "rebound_slow_per_click",
        "rebound_fast_base",
        "rebound_fast_per_click",
    ):
        assert damp[key] > 0, f"Expected positive physical value for {key}, got {damp[key]}"
