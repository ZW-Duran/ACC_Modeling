"""Model fitting and verification modules (Stage 2 SINDy & Stage 3 Verification)."""

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
]
