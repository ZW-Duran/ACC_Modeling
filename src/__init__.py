"""ACC_Modeling: Physics-Informed Vehicle Dynamics Modeling & Telemetry Calibration.

Public package entry point re-exporting the primary telemetry loading, curation,
feature-library, SINDy identification, statistical validation, visualization,
and black-box setup calibration APIs for reuse across projects.

Unit & Sampling Conventions
---------------------------
- **Kinematics & Dynamics (SI)**:
  - ``Time``: seconds (``s``), strictly sorted ascending.
  - ``SPEED``: meters per second (``m/s``).
  - ``ROTY`` ($r$): radians per second (``rad/s``).
  - ``R_DOT`` ($\\dot{r}$): radians per second squared (``rad/s^2``).
  - ``G_LAT`` ($a_y$), ``G_LON``, ``AX_SPEED`` ($a_x \\approx \\dot{v}$): meters per second squared (``m/s^2``).
  - ``STEERANGLE`` ($\\delta$): radians (``rad``).
  - ``TYRE_PRESS_*``: pounds per square inch (``psi``).
- **Sampling Rates**:
  - Whole-vehicle Stage 2 SINDy modeling: **20 Hz** (``dt = 0.05 s``, ``savgol_window = 5``),
    matching native ACC ``ROTY`` / ``G_LAT`` sensor updates.
  - Black-box suspension/damper calibration: **200 Hz** (``dt = 0.005 s``), required to
    resolve ``> 1.2 Hz`` damper ringdown and bumpstop impact velocities.
"""

from src.calibration import (
    calibrate_aero,
    calibrate_arb_stiffness,
    calibrate_bumpstop_range,
    calibrate_dampers,
    calibrate_mass,
    run_gt3_blackbox_calibration,
    verify_setup_consistency,
)
from src.cli import run_pipeline
from src.config import ExperimentConfig, load_config
from src.data import (
    TelemetryMetadata,
    compute_tire_grip_factor,
    curate_telemetry,
    load_motec_csv,
    trim_session,
)
from src.features import (
    CandidateLibrary,
    ChassisSetupSpec,
    FeatureMatrix,
    FullChassisLibrary,
    PlanarForceLibrary,
    WheelCornerLibrary,
    estimate_corner_normal_load,
)
from src.modeling import EvaluationResult, FittedYawModel, evaluate_model, fit_yaw_sindy
from src.stats import (
    AnomalySummary,
    BootstrapSummary,
    ResidualDiagnostics,
    TermConfidence,
    analyze_residuals,
    block_bootstrap_sindy,
    map_anomalies_to_corners,
)
from src.visualization import (
    plot_race_anomaly_trace,
    plot_residual_distribution,
    plot_yaw_acceleration_fit,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Config & Pipeline
    "ExperimentConfig",
    "load_config",
    "run_pipeline",
    # Data Loading & Curation
    "TelemetryMetadata",
    "load_motec_csv",
    "trim_session",
    "compute_tire_grip_factor",
    "curate_telemetry",
    # Feature Libraries
    "CandidateLibrary",
    "FeatureMatrix",
    "PlanarForceLibrary",
    "WheelCornerLibrary",
    "estimate_corner_normal_load",
    "ChassisSetupSpec",
    "FullChassisLibrary",
    # Modeling & Evaluation
    "FittedYawModel",
    "fit_yaw_sindy",
    "EvaluationResult",
    "evaluate_model",
    # Calibration
    "verify_setup_consistency",
    "calibrate_mass",
    "calibrate_bumpstop_range",
    "calibrate_aero",
    "calibrate_arb_stiffness",
    "calibrate_dampers",
    "run_gt3_blackbox_calibration",
    # Statistics & Diagnostics
    "ResidualDiagnostics",
    "analyze_residuals",
    "TermConfidence",
    "BootstrapSummary",
    "block_bootstrap_sindy",
    "AnomalySummary",
    "map_anomalies_to_corners",
    # Visualization
    "plot_yaw_acceleration_fit",
    "plot_residual_distribution",
    "plot_race_anomaly_trace",
]
