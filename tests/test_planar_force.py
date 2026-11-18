from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CarConfig, ModelingConfig, load_config
from src.features.planar_force import PlanarForceLibrary
from src.features.wheel_corner import estimate_corner_normal_load
from src.modeling.evaluator import evaluate_model
from src.modeling.trainer import fit_yaw_sindy


def test_planar_force_library_recovers_synthetic_ode():
    rng = np.random.default_rng(123)
    n = 3000
    delta = rng.uniform(-0.3, 0.3, size=n)
    delta_dot = rng.uniform(-1.0, 1.0, size=n)
    r = rng.uniform(-0.8, 0.8, size=n)
    ay = rng.uniform(-15.0, 15.0, size=n)
    ax = rng.uniform(-12.0, 12.0, size=n)
    v = rng.uniform(20.0, 70.0, size=n)
    ay_dot = rng.uniform(-30.0, 30.0, size=n)
    eta_p = rng.uniform(0.7, 1.0, size=n)

    r_dot = 1.4 * delta - 0.8 * r + 0.015 * ay_dot - 0.02 * ay * np.abs(ax) / (eta_p**2)

    df = pd.DataFrame(
        {
            "STEERANGLE": delta,
            "STEER_DOT": delta_dot,
            "ROTY": r,
            "G_LAT": ay,
            "G_LON": ax,
            "SPEED": v,
            "AY_DOT": ay_dot,
            "ETA_P": eta_p,
            "R_DOT": r_dot,
        }
    )

    lib = PlanarForceLibrary()
    feats = lib.build(df)
    assert feats.theta.shape == (n, 8)

    model = fit_yaw_sindy(feats, ModelingConfig(stlsq_threshold=0.005, stlsq_alpha=0.0, normalize_columns=True))
    res = evaluate_model(model, feats)

    assert res.r2 > 0.9999
    assert np.isclose(model.coefficients[0], 1.4, atol=1e-3)
    assert np.isclose(model.coefficients[2], -0.8, atol=1e-3)
    assert np.isclose(model.coefficients[5], 0.015, atol=1e-3)
    assert np.isclose(model.coefficients[6], -0.02, atol=1e-3)


def test_modular_config_loading():
    repo_root = Path(__file__).resolve().parents[1]
    gt3 = load_config(repo_root / "configs/experiments/stage2_planar_force/barcelona__audi_r8_gt3_evo2__r8_A.yaml")
    assert gt3.experiment_name == "stage2_barcelona_gt3_r8_A"
    assert gt3.car.car_id == "audi_r8_gt3_evo2"
    assert gt3.car.vehicle_id == "audi_r8_gt3_evo2"
    assert np.isclose(gt3.car.wheelbase_m, 2.70)
    assert gt3.car.aero_baseline.reference_area_m2 == 2.08
    assert np.isclose(gt3.car.aero_baseline.base_cl, 1.177, atol=1e-3)
    assert np.isclose(gt3.car.aero_baseline.base_aero_front_dist, 0.437, atol=1e-3)
    assert gt3.car.calibration_slots["bumpstop_range_mm"]["front_base_mm"] == 17.0
    assert gt3.car.calibration_slots["bumpstop_range_mm"]["front_per_click_mm"] == 1.0
    assert gt3.car.calibration_slots["dampers_n_s_m"]["knee_velocity_mps"] == 0.05
    assert gt3.car.calibration_slots["dampers_n_s_m"]["bump_slow_base"] > 0
    assert gt3.car.calibration_slots["arb_stiffness_n_m"]["front_per_click"] > 0
    assert gt3.car.calibration_slots["aero_clicks"]["rear_wing_dcl_per_click"] > 0
    assert (
        gt3.car.calibration_slots["aero_clicks"]["front_brake_duct_dcd_per_click"]
        > gt3.car.calibration_slots["aero_clicks"]["rear_brake_duct_dcd_per_click"]
        > 0
    )
    assert gt3.setup["setup_name"] == "barcelona_R8_A"
    assert gt3.setup["suspension"]["arb_clicks"] == [4, 2]
    assert gt3.track.track_id == "barcelona_catalunya"
    assert len(gt3.track.corners) == 8
    assert gt3.curation.savgol_window == 41
    assert gt3.modeling.formulation == "planar_force"

    gt4 = load_config(repo_root / "configs/experiments/stage2_planar_force/barcelona__audi_r8_gt4__a1.yaml")
    assert gt4.experiment_name == "stage2_barcelona_gt4_a1"
    assert gt4.car.car_id == "audi_r8_gt4"
    assert np.isclose(gt4.car.wheelbase_m, 2.65)
    assert np.isclose(gt4.car.mass_kg, 1460.0)
    assert gt4.car.aero_baseline.base_cl == 0.32
    assert gt4.setup["setup_name"] == "barcelona_a1"
    assert gt4.setup["aero_clicks"]["rear_wing"] == 1


def test_wheel_corner_load_transfer_conservation():
    car = CarConfig(mass_kg=1300.0, front_weight_dist=0.43, wheelbase_m=2.65, cg_height_m=0.42)
    df = pd.DataFrame({"G_LON": [-10.0, 0.0, 5.0], "G_LAT": [0.0, 12.0, -8.0], "SPEED": [50.0, 40.0, 60.0]})
    total_fz = sum(estimate_corner_normal_load(df, c, car) for c in ("LF", "RF", "LR", "RR"))
    expected = np.full(3, car.mass_kg * 9.80665)
    assert np.allclose(total_fz, expected, rtol=1e-6)
