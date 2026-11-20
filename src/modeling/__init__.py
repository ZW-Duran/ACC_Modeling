"""Model fitting and evaluation modules (Stage 2 SINDy & Stage 3 4-wheel physics)."""

from src.modeling.evaluator import EvaluationResult, evaluate_model
from src.modeling.stage3_physics import (
    Stage3PhysicsParams,
    Stage3SimulationResult,
    Stage3WheelCornerModel,
    compute_corner_normal_loads,
    run_stage3_pipeline,
)
from src.modeling.trainer import FittedYawModel, fit_yaw_sindy

__all__ = [
    "FittedYawModel",
    "fit_yaw_sindy",
    "EvaluationResult",
    "evaluate_model",
    "Stage3PhysicsParams",
    "Stage3SimulationResult",
    "Stage3WheelCornerModel",
    "compute_corner_normal_loads",
    "run_stage3_pipeline",
]
