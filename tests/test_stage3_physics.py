from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CarConfig, load_config
from src.modeling.stage3_physics import (
    Stage3PhysicsParams,
    Stage3WheelCornerModel,
    compute_corner_normal_loads,
)


def test_stage3_normal_load_conservation_and_fit():
    car = CarConfig(
        mass_kg=1161.2,
        front_weight_dist=0.43,
        wheelbase_m=2.70,
        track_width_front_m=1.67,
        track_width_rear_m=1.65,
        cg_height_m=0.42,
    )
    setup = {
        "steering_ratio": 16.0,
        "suspension": {"wheel_rate_n_m": [153000, 153000, 163000, 163000], "arb_clicks": [4, 2]},
        "aero_clicks": {"rear_wing": 5},
    }

    rng = np.random.default_rng(7)
    n = 400
    t = np.arange(n) * 0.05
    v = np.full(n, 45.0)
    delta_w = 0.04 * np.sin(1.2 * t)
    r = 0.35 * np.sin(1.2 * t - 0.1)
    ay = v * r
    ax = np.zeros(n)

    df = pd.DataFrame(
        {
            "Time": t,
            "SPEED": v,
            "ROTY": r,
            "STEERANGLE": delta_w * 16.0,
            "G_LAT": ay,
            "AX_SPEED": ax,
            "R_DOT": np.gradient(r, t),
            "TYRE_PRESS_LF": np.full(n, 27.0),
            "TYRE_PRESS_RF": np.full(n, 27.0),
            "TYRE_PRESS_LR": np.full(n, 27.0),
            "TYRE_PRESS_RR": np.full(n, 27.0),
        }
    )

    fz, fz_susp, fz_tot = compute_corner_normal_loads(df, car, setup, fuel_liters=2.0)
    assert len(fz) == 4
    assert np.allclose(fz["LF"] + fz["RF"] + fz["LR"] + fz["RR"], fz_tot)
    assert np.all(fz_tot > 1162.7 * 9.80665)

    # Plant synthetic Pacejka observations
    true_params = Stage3PhysicsParams(mu0=1.5, pacejka_b=9.0, pacejka_c=1.4, pacejka_e=-0.3, load_sensitivity=0.12, iz_kg_m2=1900.0, vy_compliance_s=-0.05)
    plant = Stage3WheelCornerModel(car=car, setup=setup, fuel_liters=2.0, params=true_params)
    sim_true = plant.evaluate(df)

    df["G_LAT"] = sim_true.ay_pred_mps2
    df["R_DOT"] = sim_true.r_dot_pred_rad_s2

    fitted = Stage3WheelCornerModel(car=car, setup=setup, fuel_liters=2.0)
    res = fitted.fit(df, max_nfev=120)
    assert res.eval_ay.r2 > 0.99
    assert res.eval_r_dot.r2 > 0.95


def test_stage3_experiment_config_loads():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs/experiments/stage3_wheel_corner/barcelona__audi_r8_gt3_evo2__r8_A.yaml")
    assert cfg.experiment_name == "stage3_barcelona_gt3_r8_A"
    assert cfg.car.mass_kg == 1161.2
