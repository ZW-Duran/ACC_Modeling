from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration.verification import verify_calibrated_coefficients
from src.config import CarConfig, load_config
from src.data.curation import curate_telemetry
from src.data.motec_loader import load_motec_csv
from src.modeling.stage3_verification import (
    compute_corner_normal_loads,
    predict_suspension_state,
    verify_stage3_scenarios,
    verify_straight_aero_drag,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stage3_progressive_bumpstop_and_normal_loads_synthetic():
    car = CarConfig(
        mass_kg=1161.2,
        front_weight_dist=0.43,
        wheelbase_m=2.70,
        track_width_front_m=1.67,
        track_width_rear_m=1.65,
        cg_height_m=0.42,
    )
    setup = {
        "suspension": {
            "wheel_rate_n_m": [153000, 153000, 163000, 163000],
            "bumpstop_rate_n_mm": [1000, 1000, 300, 300],
            "bumpstop_range_clicks": [5, 5, 14, 14],
            "arb_clicks": [4, 2],
        },
        "aero_clicks": {"rear_wing": 5, "brake_ducts": [2, 3]},
    }

    n = 200
    t = np.arange(n) * 0.05
    df = pd.DataFrame(
        {
            "Time": t,
            "LAP_DIST_M": np.linspace(0.0, 4500.0, n),
            "SPEED": np.full(n, 60.0),
            "G_LAT": 10.0 * np.sin(0.5 * t),
            "AX_SPEED": -15.0 * np.cos(0.5 * t),
        }
    )

    fz, _, fz_tot = compute_corner_normal_loads(df, car, setup, fuel_liters=2.0)
    assert np.allclose(fz["LF"] + fz["RF"] + fz["LR"] + fz["RR"], fz_tot)

    state = predict_suspension_state(df, car, setup, fuel_liters=2.0)
    assert len(state.pitch_deg_pred) == n
    assert np.max(state.bs_force_pred_n["LF"]) > 0.0


@pytest.mark.skipif(
    not (ROOT / "data/training_Barcelona_GT3.csv").exists(),
    reason="Local GT3 track telemetry not present.",
)
def test_stage3_verification_on_real_barcelona_telemetry():
    cfg = load_config(ROOT / "configs/experiments/stage3_verification/barcelona__audi_r8_gt3_evo2__r8_A.yaml")
    assert cfg.experiment_name == "stage3_barcelona_gt3_verification"

    cols = [
        "Time", "Distance", "ROTY", "STEERANGLE", "SPEED", "G_LAT", "G_LON", "THROTTLE", "BRAKE",
        "SUS_TRAVEL_LF", "SUS_TRAVEL_RF", "SUS_TRAVEL_LR", "SUS_TRAVEL_RR",
        "BUMPSTOPUP_RIDE_LF", "BUMPSTOPUP_RIDE_RF", "BUMPSTOPUP_RIDE_LR", "BUMPSTOPUP_RIDE_RR",
        "BUMPSTOP_FORCE_LF", "BUMPSTOP_FORCE_RF", "BUMPSTOP_FORCE_LR", "BUMPSTOP_FORCE_RR",
        "TYRE_PRESS_LF", "TYRE_PRESS_RF", "TYRE_PRESS_LR", "TYRE_PRESS_RR",
    ]
    raw_tr, m_tr = load_motec_csv(ROOT / cfg.data.clean_csv, channels=cols)
    df_tr = curate_telemetry(raw_tr, m_tr, cfg.curation, trim_mode="none")

    cal_res = verify_calibrated_coefficients(raw_tr, df_tr, cfg.car, cfg.setup, fuel_liters=2.0)
    assert cal_res.bumpstop["LF"].classification_accuracy_pct > 95.0
    assert cal_res.axle.front_roll_r2 > 0.85

    aero_drag = verify_straight_aero_drag(df_tr, cfg.car, cfg.setup, ROOT / "data/calibration/audi_r8_gt3_evo2", fuel_liters=2.0)
    assert aero_drag.rmse_kmh < 2.0

    state, scenarios = verify_stage3_scenarios(df_tr, cfg.car, cfg.setup, fuel_liters=2.0)
    assert len(scenarios) == 5
    bs_lf = (raw_tr["BUMPSTOPUP_RIDE_LF"] == 1).to_numpy()
    in_bs_rmse = float(np.sqrt(np.mean((state.z_obs_mm["LF"][bs_lf] - state.z_pred_mm["LF"][bs_lf]) ** 2)))
    assert in_bs_rmse < 2.5
