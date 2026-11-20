"""Stage 3: Zero-shot verification of calibrated black-box coefficients on independent telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import CarConfig

GRAVITY = 9.80665
AIR_DENSITY = 1.225
CORNERS: tuple[str, ...] = ("LF", "RF", "LR", "RR")


@dataclass
class BumpstopVerificationMetrics:
    """Bumpstop engagement threshold and binary classification accuracy for a single corner."""

    corner: str
    predicted_threshold_mm: float
    observed_min_engaged_mm: float | None
    threshold_error_mm: float | None
    classification_accuracy_pct: float
    engaged_frames: int


@dataclass
class AxleVerificationMetrics:
    """Zero-shot roll gradient and aero/pitch heave accuracy across front and rear axles."""

    front_roll_r2: float
    front_roll_rmse_mm: float
    rear_roll_r2: float
    rear_roll_rmse_mm: float
    front_heave_rmse_mm: float
    rear_heave_rmse_mm: float


@dataclass
class CornerTravelVerificationMetrics:
    """Individual corner ``SUS_TRAVEL`` reconstruction accuracy on bumpstop-free frames."""

    corner: str
    r2: float
    rmse_mm: float
    mae_mm: float


@dataclass
class CalibrationVerificationResult:
    """Complete Stage 3 zero-shot verification report and time-series arrays."""

    bumpstop: dict[str, BumpstopVerificationMetrics]
    axle: AxleVerificationMetrics
    corners: dict[str, CornerTravelVerificationMetrics]
    dz_front_obs_mm: np.ndarray
    dz_front_pred_mm: np.ndarray
    dz_rear_obs_mm: np.ndarray
    dz_rear_pred_mm: np.ndarray
    z_front_mean_obs_mm: np.ndarray
    z_front_mean_pred_mm: np.ndarray
    z_rear_mean_obs_mm: np.ndarray
    z_rear_mean_pred_mm: np.ndarray
    z_corner_obs_mm: dict[str, np.ndarray]
    z_corner_pred_mm: dict[str, np.ndarray]
    no_bumpstop_mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar verification metrics into a JSON-compatible dictionary."""
        return {
            "bumpstop": {k: asdict(v) for k, v in self.bumpstop.items()},
            "axle": asdict(self.axle),
            "corners": {k: asdict(v) for k, v in self.corners.items()},
            "n_total_frames": int(len(self.no_bumpstop_mask)),
            "n_no_bumpstop_frames": int(np.sum(self.no_bumpstop_mask)),
        }


def resolve_setup_physics(
    car: CarConfig,
    setup: dict[str, Any],
    fuel_liters: float = 2.0,
) -> dict[str, float]:
    """Resolve SI chassis, aero, spring, ARB, bumpstop, and steering constants from configs.

    Parameters
    ----------
    car : CarConfig
        Calibrated vehicle physical constants and click-to-SI black-box slots.
    setup : dict[str, Any]
        Per-track garage setup dictionary.
    fuel_liters : float, default=2.0
        Fuel load [L] added to ``car.mass_kg`` using ``0.75 kg/L`` density.

    Returns
    -------
    dict[str, float]
        SI values for mass, wheel rates, ARB stiffnesses, roll share, aero ``Cl``,
        bumpstop thresholds, reference area, and steering ratio.
    """
    mass_kg = float(car.mass_kg + fuel_liters * 0.75)
    susp = setup.get("suspension", {})
    wr = susp.get("wheel_rate_n_m", [153000.0, 153000.0, 163000.0, 163000.0])
    kw_f = float(wr[0])
    kw_r = float(wr[2])
    arb_clicks = susp.get("arb_clicks", [4, 2])
    bs_clicks = susp.get("bumpstop_range_clicks", [5, 5, 14, 14])

    arb_slots = car.calibration_slots.get("arb_stiffness_n_m", {})
    k_arb_f = float(arb_slots.get("front_base", 25258.0)) + float(arb_slots.get("front_per_click", 4339.1)) * float(
        arb_clicks[0]
    )
    k_arb_r = float(arb_slots.get("rear_base", 75718.4)) + float(arb_slots.get("rear_per_click", 4959.3)) * float(
        arb_clicks[1]
    )

    k_roll_f = (kw_f + k_arb_f) * (car.track_width_front_m**2)
    k_roll_r = (kw_r + k_arb_r) * (car.track_width_rear_m**2)
    roll_front_share = k_roll_f / max(k_roll_f + k_roll_r, 1e-6)

    aero_clicks = setup.get("aero_clicks", {})
    wing_click = float(aero_clicks.get("rear_wing", 5))
    aero_slot = car.calibration_slots.get("aero_clicks", {})
    base_cl = float(car.aero_baseline.base_cl or 1.177)
    base_f_dist = float(car.aero_baseline.base_aero_front_dist or 0.424)
    dcl_wing = float(aero_slot.get("rear_wing_dcl_per_click", 0.0352))
    cl_tot = base_cl + wing_click * dcl_wing
    # Rear wing shifts aero balance rearward as wing increases
    cl_f = base_cl * base_f_dist + wing_click * (-0.0154)
    cl_r = max(cl_tot - cl_f, 0.1)

    bs_slot = car.calibration_slots.get("bumpstop_range_mm", {})
    bs_thresh_f = float(bs_slot.get("front_base_mm", 17.0)) + float(bs_slot.get("front_per_click_mm", 1.0)) * float(
        bs_clicks[0]
    )
    bs_thresh_r = float(bs_slot.get("rear_base_mm", 24.0)) + float(bs_slot.get("rear_per_click_mm", 1.0)) * float(
        bs_clicks[2]
    )

    return {
        "mass_kg": mass_kg,
        "kw_f": kw_f,
        "kw_r": kw_r,
        "k_arb_f": k_arb_f,
        "k_arb_r": k_arb_r,
        "k_eff_f": kw_f + k_arb_f,
        "k_eff_r": kw_r + k_arb_r,
        "roll_front_share": roll_front_share,
        "cl_tot": cl_tot,
        "cl_f": cl_f,
        "cl_r": cl_r,
        "ref_area_m2": float(car.aero_baseline.reference_area_m2 or 2.08),
        "steering_ratio": float(setup.get("steering_ratio", 16.0)),
        "bs_thresh_f_mm": bs_thresh_f,
        "bs_thresh_r_mm": bs_thresh_r,
    }


def _r2_rmse(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    diff = obs - pred
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1.0 - float(np.sum(diff**2)) / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return r2, rmse, mae


def _corr_sign(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 1.0
    c = float(np.corrcoef(x, y)[0, 1])
    return 1.0 if c >= 0.0 else -1.0


def verify_calibrated_coefficients(
    df_raw: pd.DataFrame,
    df_curated: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    fuel_liters: float = 2.0,
) -> CalibrationVerificationResult:
    """Zero-shot verification of calibrated black-box coefficients on independent telemetry.

    Evaluates bumpstop thresholds against raw ``BUMPSTOPUP_RIDE_*``, axle roll gradients and
    aero/pitch heave on bumpstop-free frames, and individual 4-corner ``SUS_TRAVEL_*`` tracking
    without fitting any additional regression weights.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw SI telemetry (containing ``SUS_TRAVEL_{LF,RF,LR,RR}`` [mm] and
        ``BUMPSTOPUP_RIDE_{LF,RF,LR,RR}`` ``{0, 1}``).
    df_curated : pd.DataFrame
        Curated SI telemetry aligned with ``df_raw`` (containing ``SPEED`` [m/s],
        ``G_LAT`` [m/s^2], ``AX_SPEED`` or ``G_LON`` [m/s^2], and ``SUS_TRAVEL_*`` [mm]).
    car : CarConfig
        Vehicle configuration carrying calibrated constants and black-box click slots.
    setup : dict[str, Any]
        Setup dictionary for the session under verification.
    fuel_liters : float, default=2.0
        Fuel load [L] during the verified session.

    Returns
    -------
    CalibrationVerificationResult
        Per-corner bumpstop accuracy, front/rear roll and heave error metrics, corner travel
        reconstruction metrics, and observed vs. zero-shot predicted arrays.
    """
    required_raw = [f"SUS_TRAVEL_{c}" for c in CORNERS] + [f"BUMPSTOPUP_RIDE_{c}" for c in CORNERS]
    missing_raw = [c for c in required_raw if c not in df_raw.columns]
    if missing_raw:
        raise ValueError(f"Missing raw calibration-verification columns: {missing_raw}")

    required_cur = ["SPEED", "G_LAT"] + [f"SUS_TRAVEL_{c}" for c in CORNERS]
    missing_cur = [c for c in required_cur if c not in df_curated.columns]
    if missing_cur:
        raise ValueError(f"Missing curated calibration-verification columns: {missing_cur}")

    phys = resolve_setup_physics(car, setup, fuel_liters=fuel_liters)
    m = phys["mass_kg"]
    fw = car.front_weight_dist
    h_cg = car.cg_height_m

    # 1. Bumpstop engagement threshold verification (raw unsmoothed telemetry)
    bumpstop_metrics: dict[str, BumpstopVerificationMetrics] = {}
    no_bs = np.ones(len(df_raw), dtype=bool)
    for c in CORNERS:
        pred_thr = phys["bs_thresh_f_mm"] if c in ("LF", "RF") else phys["bs_thresh_r_mm"]
        bu = df_raw[f"BUMPSTOPUP_RIDE_{c}"].to_numpy(dtype=float) == 1.0
        st_raw = df_raw[f"SUS_TRAVEL_{c}"].to_numpy(dtype=float)
        no_bs &= ~bu
        engaged_n = int(np.sum(bu))
        if engaged_n > 0:
            obs_thr = float(np.min(st_raw[bu]))
            thr_err = float(obs_thr - pred_thr)
        else:
            obs_thr = None
            thr_err = None
        acc_pct = float(np.mean((st_raw >= pred_thr) == bu) * 100.0)
        bumpstop_metrics[c] = BumpstopVerificationMetrics(
            corner=c,
            predicted_threshold_mm=round(pred_thr, 3),
            observed_min_engaged_mm=round(obs_thr, 3) if obs_thr is not None else None,
            threshold_error_mm=round(thr_err, 3) if thr_err is not None else None,
            classification_accuracy_pct=round(acc_pct, 2),
            engaged_frames=engaged_n,
        )

    if len(no_bs) != len(df_curated):
        # Trim alignment if df_curated dropped pit-exit prefix rows
        no_bs = no_bs[-len(df_curated) :]

    eval_mask = no_bs if int(np.sum(no_bs)) >= 10 else np.ones(len(df_curated), dtype=bool)

    v = df_curated["SPEED"].to_numpy(dtype=float)
    ay = df_curated["G_LAT"].to_numpy(dtype=float)
    ax_col = "AX_SPEED" if "AX_SPEED" in df_curated.columns else "G_LON"
    ax = df_curated[ax_col].to_numpy(dtype=float)

    # 2. ARB & Roll Gradient verification
    dz_f_obs = (df_curated["SUS_TRAVEL_LF"] - df_curated["SUS_TRAVEL_RF"]).to_numpy(dtype=float)
    dz_r_obs = (df_curated["SUS_TRAVEL_LR"] - df_curated["SUS_TRAVEL_RR"]).to_numpy(dtype=float)

    s_f = _corr_sign(dz_f_obs[eval_mask], ay[eval_mask])
    s_r = _corr_sign(dz_r_obs[eval_mask], ay[eval_mask])
    dz_f_pred = s_f * (2.0 * (m * fw) * h_cg / (max(car.track_width_front_m, 1e-3) * phys["k_eff_f"])) * ay * 1e3
    dz_r_pred = (
        s_r * (2.0 * (m * (1.0 - fw)) * h_cg / (max(car.track_width_rear_m, 1e-3) * phys["k_eff_r"])) * ay * 1e3
    )

    f_roll_r2, f_roll_rmse, _ = _r2_rmse(dz_f_obs[eval_mask], dz_f_pred[eval_mask])
    r_roll_r2, r_roll_rmse, _ = _r2_rmse(dz_r_obs[eval_mask], dz_r_pred[eval_mask])

    # 3. Aero downforce & pitch heave verification (reference static datum: 15 mm front, 17 mm rear)
    q = 0.5 * AIR_DENSITY * phys["ref_area_m2"] * (v**2)
    z0_f_mm, z0_r_mm = 15.0, 17.0
    zf_mean_obs = 0.5 * (df_curated["SUS_TRAVEL_LF"] + df_curated["SUS_TRAVEL_RF"]).to_numpy(dtype=float)
    zr_mean_obs = 0.5 * (df_curated["SUS_TRAVEL_LR"] + df_curated["SUS_TRAVEL_RR"]).to_numpy(dtype=float)

    dfz_pitch = 0.5 * m * ax * (h_cg / max(car.wheelbase_m, 1e-3))
    zf_mean_pred = z0_f_mm + (0.5 * q * phys["cl_f"] - dfz_pitch) / (phys["kw_f"] * 1e-3)
    zr_mean_pred = z0_r_mm + (0.5 * q * phys["cl_r"] + dfz_pitch) / (phys["kw_r"] * 1e-3)

    _, f_heave_rmse, _ = _r2_rmse(zf_mean_obs[eval_mask], zf_mean_pred[eval_mask])
    _, r_heave_rmse, _ = _r2_rmse(zr_mean_obs[eval_mask], zr_mean_pred[eval_mask])

    axle_metrics = AxleVerificationMetrics(
        front_roll_r2=round(f_roll_r2, 4),
        front_roll_rmse_mm=round(f_roll_rmse, 3),
        rear_roll_r2=round(r_roll_r2, 4),
        rear_roll_rmse_mm=round(r_roll_rmse, 3),
        front_heave_rmse_mm=round(f_heave_rmse, 3),
        rear_heave_rmse_mm=round(r_heave_rmse, 3),
    )

    # 4. Individual 4-corner SUS_TRAVEL reconstruction
    z_corner_pred = {
        "LF": zf_mean_pred + 0.5 * dz_f_pred,
        "RF": zf_mean_pred - 0.5 * dz_f_pred,
        "LR": zr_mean_pred + 0.5 * dz_r_pred,
        "RR": zr_mean_pred - 0.5 * dz_r_pred,
    }
    z_corner_obs: dict[str, np.ndarray] = {}
    corner_metrics: dict[str, CornerTravelVerificationMetrics] = {}
    for c in CORNERS:
        obs_c = df_curated[f"SUS_TRAVEL_{c}"].to_numpy(dtype=float)
        z_corner_obs[c] = obs_c
        r2_c, rmse_c, mae_c = _r2_rmse(obs_c[eval_mask], z_corner_pred[c][eval_mask])
        corner_metrics[c] = CornerTravelVerificationMetrics(
            corner=c,
            r2=round(r2_c, 4),
            rmse_mm=round(rmse_c, 3),
            mae_mm=round(mae_c, 3),
        )

    return CalibrationVerificationResult(
        bumpstop=bumpstop_metrics,
        axle=axle_metrics,
        corners=corner_metrics,
        dz_front_obs_mm=dz_f_obs,
        dz_front_pred_mm=dz_f_pred,
        dz_rear_obs_mm=dz_r_obs,
        dz_rear_pred_mm=dz_r_pred,
        z_front_mean_obs_mm=zf_mean_obs,
        z_front_mean_pred_mm=zf_mean_pred,
        z_rear_mean_obs_mm=zr_mean_obs,
        z_rear_mean_pred_mm=zr_mean_pred,
        z_corner_obs_mm=z_corner_obs,
        z_corner_pred_mm=z_corner_pred,
        no_bumpstop_mask=eval_mask,
    )
