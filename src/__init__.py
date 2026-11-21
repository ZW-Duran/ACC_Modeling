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
    AxleVerificationMetrics,
    BumpstopVerificationMetrics,
    CalibrationVerificationResult,
    CornerTravelVerificationMetrics,
    calibrate_aero,
    calibrate_arb_stiffness,
    calibrate_bumpstop_range,
    calibrate_dampers,
    calibrate_mass,
    resolve_setup_physics,
    run_gt3_blackbox_calibration,
    verify_calibrated_coefficients,
    verify_setup_consistency,
)
from src.cli import run_pipeline
from src.config import ExperimentConfig, load_car_config, load_config, load_setup_file, load_track_config
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
from src.modeling import (
    SCENARIO_WINDOWS_M,
    STAGE4_CHANNELS,
    AeroDragVerificationResult,
    EvaluationResult,
    FittedYawModel,
    LapDiagnosticState,
    ScenarioMetrics,
    SuspensionVerificationState,
    build_ai_handoff_payload,
    build_physics_model_spec,
    classify_grip_state,
    compute_corner_normal_loads,
    compute_vehicle_coupled_baseline,
    evaluate_7dof_lap_state,
    evaluate_model,
    extract_corner_diagnostics,
    fit_yaw_sindy,
    predict_suspension_state,
    run_counterfactual_simulation,
    save_diagnose_plots,
    save_simulate_plots,
    verify_stage3_scenarios,
    verify_straight_aero_drag,
)
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
    plot_finite_diff_yaw_accel,
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
    "load_car_config",
    "load_track_config",
    "load_setup_file",
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
    # Modeling, Stage 3 Verification & Stage 4 Closed-Loop Diagnostics
    "FittedYawModel",
    "fit_yaw_sindy",
    "EvaluationResult",
    "evaluate_model",
    "SCENARIO_WINDOWS_M",
    "AeroDragVerificationResult",
    "SuspensionVerificationState",
    "ScenarioMetrics",
    "compute_corner_normal_loads",
    "predict_suspension_state",
    "verify_straight_aero_drag",
    "verify_stage3_scenarios",
    "STAGE4_CHANNELS",
    "LapDiagnosticState",
    "compute_vehicle_coupled_baseline",
    "evaluate_7dof_lap_state",
    "classify_grip_state",
    "extract_corner_diagnostics",
    "build_physics_model_spec",
    "build_ai_handoff_payload",
    "run_counterfactual_simulation",
    "save_diagnose_plots",
    "save_simulate_plots",
    # Calibration & Stage 3 Verification
    "verify_setup_consistency",
    "calibrate_mass",
    "calibrate_bumpstop_range",
    "calibrate_aero",
    "calibrate_arb_stiffness",
    "calibrate_dampers",
    "run_gt3_blackbox_calibration",
    "BumpstopVerificationMetrics",
    "AxleVerificationMetrics",
    "CornerTravelVerificationMetrics",
    "CalibrationVerificationResult",
    "resolve_setup_physics",
    "verify_calibrated_coefficients",
    # Statistics & Diagnostics
    "ResidualDiagnostics",
    "analyze_residuals",
    "TermConfidence",
    "BootstrapSummary",
    "block_bootstrap_sindy",
    "AnomalySummary",
    "map_anomalies_to_corners",
    # Visualization
    "plot_finite_diff_yaw_accel",
    "plot_yaw_acceleration_fit",
    "plot_residual_distribution",
    "plot_race_anomaly_trace",
]
