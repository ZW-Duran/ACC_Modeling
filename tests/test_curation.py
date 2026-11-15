from pathlib import Path

import numpy as np
import pandas as pd

from sindy_piml.config import CurationConfig
from sindy_piml.data.curation import compute_tire_grip_factor, curate_telemetry, trim_session
from sindy_piml.data.motec_loader import TelemetryMetadata


def _make_synthetic_df(n: int = 1000, dt: float = 0.005) -> tuple[pd.DataFrame, TelemetryMetadata]:
    t = np.arange(n) * dt
    # Smooth yaw rate r(t) = 0.5 * sin(2 * pi * 0.5 * t)
    r = 0.5 * np.sin(np.pi * t)
    speed = np.linspace(0.0, 50.0, n)  # m/s (0 -> 180 km/h)
    df = pd.DataFrame(
        {
            "Time": t,
            "ROTY": r,
            "STEERANGLE": 0.1 * np.sin(np.pi * t),
            "SPEED": speed,
            "G_LAT": 8.0 * np.sin(np.pi * t),
            "G_LON": np.zeros(n),
            "TYRE_PRESS_LF": np.full(n, 27.0),
            "TYRE_PRESS_RF": np.full(n, 27.0),
            "TYRE_PRESS_LR": np.full(n, 27.0),
            "TYRE_PRESS_RR": np.full(n, 27.0),
        }
    )
    meta = TelemetryMetadata(filepath=Path("dummy.csv"), sample_rate_hz=1.0 / dt, beacon_markers_s=[2.0])
    return df, meta


def test_trim_session_speed_and_beacon():
    df, meta = _make_synthetic_df()
    trimmed_sp = trim_session(df, meta, mode="speed_threshold", speed_threshold_kmh=100.0)
    assert trimmed_sp["SPEED"].iloc[0] >= 100.0 / 3.6 - 1e-9

    trimmed_bc = trim_session(df, meta, mode="first_beacon")
    assert trimmed_bc["Time"].iloc[0] >= 2.0


def test_tire_grip_factor_peak_and_decay():
    df = pd.DataFrame(
        {
            "TYRE_PRESS_LF": [27.0, 24.0],
            "TYRE_PRESS_RF": [27.0, 24.0],
            "TYRE_PRESS_LR": [27.0, 24.0],
            "TYRE_PRESS_RR": [27.0, 24.0],
        }
    )
    p_mean, eta = compute_tire_grip_factor(df, p_opt_psi=27.0, sigma_p_psi=1.5)
    assert np.isclose(p_mean[0], 27.0)
    assert np.isclose(eta[0], 1.0)
    assert eta[1] < 0.2


def test_curate_telemetry_analytical_derivative():
    df, meta = _make_synthetic_df(n=1000, dt=0.005)
    cfg = CurationConfig(savgol_window=21, savgol_polyorder=3)
    curated = curate_telemetry(df, meta, cfg, trim_mode="none")

    # True derivative of 0.5*sin(pi*t) is 0.5*pi*cos(pi*t)
    t = curated["Time"].to_numpy()
    expected_r_dot = 0.5 * np.pi * np.cos(np.pi * t)
    # Ignore edge boundary samples
    mae = np.mean(np.abs(curated["R_DOT"].to_numpy()[25:-25] - expected_r_dot[25:-25]))
    assert mae < 0.01
