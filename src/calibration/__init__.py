"""Black-box setup click-to-physical parameter calibration from controlled MoTeC telemetry."""

from src.calibration.blackbox import (
    calibrate_aero,
    calibrate_arb_stiffness,
    calibrate_bumpstop_range,
    calibrate_dampers,
    calibrate_mass,
    run_gt3_blackbox_calibration,
    verify_setup_consistency,
)

from src.calibration.verification import (
    AxleVerificationMetrics,
    BumpstopVerificationMetrics,
    CalibrationVerificationResult,
    CornerTravelVerificationMetrics,
    resolve_setup_physics,
    verify_calibrated_coefficients,
)

__all__ = [
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
]
