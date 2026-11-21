"""Model fitting, Stage 3 verification, and Stage 4 closed-loop 7-DOF diagnostic modules."""

from src.modeling.evaluator import EvaluationResult, evaluate_model
from src.modeling.stage3_verification import (
    SCENARIO_WINDOWS_M,
    AeroDragVerificationResult,
    ScenarioMetrics,
    SuspensionVerificationState,
    compute_corner_normal_loads,
    predict_suspension_state,
    verify_stage3_scenarios,
    verify_straight_aero_drag,
)
from src.modeling.stage4_diagnostic import (
    STAGE4_CHANNELS,
    LapDiagnosticState,
    build_ai_handoff_payload,
    build_physics_model_spec,
    classify_grip_state,
    compute_vehicle_coupled_baseline,
    evaluate_7dof_lap_state,
    extract_corner_diagnostics,
    run_counterfactual_simulation,
    save_diagnose_plots,
    save_simulate_plots,
)
from src.modeling.trainer import FittedYawModel, fit_yaw_sindy

__all__ = [
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
]
