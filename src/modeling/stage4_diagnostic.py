"""Stage 4: 7-DOF Inverse-Dynamics Grip Utilization & Closed-Loop AI Diagnostic Engine.

Deterministic physics feature extraction and counterfactual multi-variable simulation
for closed-loop setup engineering (with zero hardcoded setup recommendation rules).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.verification import AIR_DENSITY, CORNERS, GRAVITY, resolve_setup_physics
from src.config import CarConfig, CornerRegion, TrackConfig
from src.modeling.stage3_verification import _solve_progressive_bumpstop, verify_straight_aero_drag

STAGE4_CHANNELS: tuple[str, ...] = (
    "Time",
    "Distance",
    "ROTY",
    "STEERANGLE",
    "SPEED",
    "G_LAT",
    "G_LON",
    "THROTTLE",
    "BRAKE",
    "TC",
    "ABS",
    "SUS_TRAVEL_LF",
    "SUS_TRAVEL_RF",
    "SUS_TRAVEL_LR",
    "SUS_TRAVEL_RR",
    "BUMPSTOPUP_RIDE_LF",
    "BUMPSTOPUP_RIDE_RF",
    "BUMPSTOPUP_RIDE_LR",
    "BUMPSTOPUP_RIDE_RR",
    "BUMPSTOP_FORCE_LF",
    "BUMPSTOP_FORCE_RF",
    "BUMPSTOP_FORCE_LR",
    "BUMPSTOP_FORCE_RR",
    "TYRE_PRESS_LF",
    "TYRE_PRESS_RF",
    "TYRE_PRESS_LR",
    "TYRE_PRESS_RR",
)

M_CALIB_REF_KG = 1168.7
BASELINE_RH_MM = (66.0, 54.0)  # [Rear, Front]
STATIC_TRAVEL_MM = (15.0, 17.0)  # [Front, Rear]
BUMPSTOP_P_EXP = 2.73
BUMPSTOP_K_PROG_REF = 1.65


@dataclass
class LapDiagnosticState:
    """Full-lap 7-DOF inverse-dynamics state arrays."""

    dist_m: np.ndarray
    speed_ms: np.ndarray
    ay_mps2: np.ndarray
    ax_mps2: np.ndarray
    roty_rad_s: np.ndarray
    r_dot_rad_s2: np.ndarray
    steer_rad: np.ndarray
    steer_dot_rad_s: np.ndarray
    understeer_deg: np.ndarray
    z_mm: dict[str, np.ndarray]
    bs_force_n: dict[str, np.ndarray]
    bs_margin_mm: dict[str, np.ndarray]
    fz_corners_n: dict[str, np.ndarray]
    lltd_inst_pct: np.ndarray
    rh_dyn_front_mm: np.ndarray
    rh_dyn_rear_mm: np.ndarray
    dyn_camber_outer_front_deg: np.ndarray
    dyn_camber_outer_rear_deg: np.ndarray
    fy_req_front_n: np.ndarray
    fy_req_rear_n: np.ndarray
    f_req_front_n: np.ndarray
    f_req_rear_n: np.ndarray
    f_cap_front_n: np.ndarray
    f_cap_rear_n: np.ndarray
    rho_front: np.ndarray
    rho_rear: np.ndarray
    eta_p_mean: np.ndarray
    hot_press_psi: dict[str, np.ndarray]
    mu_0: float


def _corr_sign(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return 1.0
    c = float(np.corrcoef(x, y)[0, 1])
    return 1.0 if c >= 0.0 else -1.0


def _resolve_coupled_aero_and_rh(car: CarConfig, setup: dict[str, Any]) -> dict[str, float]:
    """Resolve aero Cl/Cd with ride-height & rake coupling relative to baseline [66, 54] mm."""
    phys = resolve_setup_physics(car, setup, fuel_liters=float(setup.get("fuel", 2.0)))
    rh = setup.get("ride_height_mm", [BASELINE_RH_MM[0], BASELINE_RH_MM[1]])
    rh_r = float(rh[0])
    rh_f = float(rh[1])
    rake_mm = rh_r - rh_f
    base_rake_mm = BASELINE_RH_MM[0] - BASELINE_RH_MM[1]
    base_mean_rh_mm = 0.5 * (BASELINE_RH_MM[0] + BASELINE_RH_MM[1])

    d_rake_mm = rake_mm - base_rake_mm
    d_mean_rh_mm = 0.5 * (rh_f + rh_r) - base_mean_rh_mm

    cl_tot_base = phys["cl_tot"]
    cl_tot = max(0.2, cl_tot_base - 0.003 * d_mean_rh_mm)
    aero_front_share_base = phys["cl_f"] / max(cl_tot_base, 1e-6)
    aero_front_share = float(np.clip(aero_front_share_base + 0.0015 * d_rake_mm, 0.25, 0.65))
    cl_f = cl_tot * aero_front_share
    cl_r = cl_tot * (1.0 - aero_front_share)

    aero_slot = car.calibration_slots.get("aero_clicks", {})
    aero_cfg = setup.get("aero_clicks", {})
    wing_c = float(aero_cfg.get("rear_wing", 5))
    ducts = aero_cfg.get("brake_ducts", [2, 3])
    dcd_wing = wing_c * float(aero_slot.get("rear_wing_dcd_per_click", 0.0073))
    dcd_ducts = float(ducts[1]) * float(aero_slot.get("front_brake_duct_dcd_per_click", 0.0030)) + float(
        ducts[0]
    ) * float(aero_slot.get("rear_brake_duct_dcd_per_click", 0.0017))
    cd_base = float(car.aero_baseline.base_cd or 0.38)
    cd_total = cd_base + dcd_wing + dcd_ducts + 0.0005 * abs(d_rake_mm)

    return {
        "rh_front_mm": rh_f,
        "rh_rear_mm": rh_r,
        "rake_mm": rake_mm,
        "rake_deg": float(np.rad2deg(np.arctan((rake_mm * 1e-3) / max(car.wheelbase_m, 1e-3)))),
        "cl_total": cl_tot,
        "cl_front": cl_f,
        "cl_rear": cl_r,
        "aero_front_balance_pct": aero_front_share * 100.0,
        "cd_total": cd_total,
        "dcd_wing": dcd_wing,
        "dcd_ducts": dcd_ducts,
    }


def compute_vehicle_coupled_baseline(car: CarConfig, setup: dict[str, Any]) -> dict[str, Any]:
    """Compute coupled chassis, aero, static LLTD, rake, and equivalent damping ratios."""
    fuel_l = float(setup.get("fuel", 2.0))
    phys = resolve_setup_physics(car, setup, fuel_liters=fuel_l)
    m = phys["mass_kg"]
    l_f = car.wheelbase_m * (1.0 - car.front_weight_dist)
    l_r = car.wheelbase_m * car.front_weight_dist
    i_z = 0.92 * m * l_f * l_r

    # Mass-scaled ARB stiffness & static LLTD
    m_ratio = m / M_CALIB_REF_KG
    k_arb_f = phys["k_arb_f"] * m_ratio
    k_arb_r = phys["k_arb_r"] * m_ratio
    k_roll_f = (phys["kw_f"] + k_arb_f) * (car.track_width_front_m**2)
    k_roll_r = (phys["kw_r"] + k_arb_r) * (car.track_width_rear_m**2)
    static_lltd_f_pct = 100.0 * k_roll_f / max(k_roll_f + k_roll_r, 1e-6)

    aero_rh = _resolve_coupled_aero_and_rh(car, setup)

    # Equivalent critical damping ratios per axle
    damp_slot = car.calibration_slots.get("dampers_n_s_m", {})
    damp_clicks = setup.get("dampers_clicks", {})
    sqrt_m_ratio = float(np.sqrt(m_ratio))

    def _mean_axle(key: str, default: float) -> tuple[float, float]:
        arr = damp_clicks.get(key, [default, default, default, default])
        return 0.5 * (float(arr[0]) + float(arr[1])), 0.5 * (float(arr[2]) + float(arr[3]))

    bs_f, bs_r = _mean_axle("bump_slow", 22.0)
    bf_f, bf_r = _mean_axle("bump_fast", 20.0)
    rs_f, rs_r = _mean_axle("rebound_slow", 25.0)
    rf_f, rf_r = _mean_axle("rebound_fast", 30.0)

    c_bs_f = (float(damp_slot.get("bump_slow_base", 2725.0)) + float(damp_slot.get("bump_slow_per_click", 19.88)) * bs_f) * sqrt_m_ratio
    c_bs_r = (float(damp_slot.get("bump_slow_base", 2725.0)) + float(damp_slot.get("bump_slow_per_click", 19.88)) * bs_r) * sqrt_m_ratio
    c_bf_f = (float(damp_slot.get("bump_fast_base", 1818.6)) + float(damp_slot.get("bump_fast_per_click", 20.73)) * bf_f) * sqrt_m_ratio
    c_bf_r = (float(damp_slot.get("bump_fast_base", 1818.6)) + float(damp_slot.get("bump_fast_per_click", 20.73)) * bf_r) * sqrt_m_ratio
    c_rs_f = (float(damp_slot.get("rebound_slow_base", 3286.8)) + float(damp_slot.get("rebound_slow_per_click", 15.54)) * rs_f) * sqrt_m_ratio
    c_rs_r = (float(damp_slot.get("rebound_slow_base", 3286.8)) + float(damp_slot.get("rebound_slow_per_click", 15.54)) * rs_r) * sqrt_m_ratio
    c_rf_f = (float(damp_slot.get("rebound_fast_base", 1953.9)) + float(damp_slot.get("rebound_fast_per_click", 5.99)) * rf_f) * sqrt_m_ratio
    c_rf_r = (float(damp_slot.get("rebound_fast_base", 1953.9)) + float(damp_slot.get("rebound_fast_per_click", 5.99)) * rf_r) * sqrt_m_ratio

    m_corner_f = 0.5 * car.front_weight_dist * m
    m_corner_r = 0.5 * (1.0 - car.front_weight_dist) * m
    c_crit_f = 2.0 * float(np.sqrt(phys["kw_f"] * m_corner_f))
    c_crit_r = 2.0 * float(np.sqrt(phys["kw_r"] * m_corner_r))

    return {
        "vehicle_name": car.name,
        "vehicle_id": car.car_id,
        "setup_name": str(setup.get("setup_name", "custom")),
        "fuel_liters": round(fuel_l, 2),
        "total_mass_kg": round(m, 2),
        "dry_mass_kg": round(car.mass_kg, 2),
        "wheelbase_m": round(car.wheelbase_m, 3),
        "track_width_front_m": round(car.track_width_front_m, 3),
        "track_width_rear_m": round(car.track_width_rear_m, 3),
        "cg_height_m": round(car.cg_height_m, 3),
        "yaw_inertia_kg_m2": round(i_z, 1),
        "static_front_weight_dist_pct": round(car.front_weight_dist * 100.0, 2),
        "static_lltd_front_pct": round(static_lltd_f_pct, 2),
        "static_ride_height_front_mm": round(aero_rh["rh_front_mm"], 1),
        "static_ride_height_rear_mm": round(aero_rh["rh_rear_mm"], 1),
        "static_rake_mm": round(aero_rh["rake_mm"], 1),
        "static_rake_deg": round(aero_rh["rake_deg"], 3),
        "aero_front_balance_pct": round(aero_rh["aero_front_balance_pct"], 2),
        "cl_total": round(aero_rh["cl_total"], 4),
        "cd_total": round(aero_rh["cd_total"], 4),
        "wheel_rate_front_n_m": round(phys["kw_f"], 1),
        "wheel_rate_rear_n_m": round(phys["kw_r"], 1),
        "arb_stiffness_front_n_m": round(k_arb_f, 1),
        "arb_stiffness_rear_n_m": round(k_arb_r, 1),
        "bumpstop_thresh_front_mm": round(phys["bs_thresh_f_mm"], 2),
        "bumpstop_thresh_rear_mm": round(phys["bs_thresh_r_mm"], 2),
        "c_crit_front_n_s_m": round(c_crit_f, 1),
        "c_crit_rear_n_s_m": round(c_crit_r, 1),
        "zeta_slow_front": round(0.5 * (c_bs_f + c_rs_f) / c_crit_f, 4),
        "zeta_fast_front": round(0.5 * (c_bf_f + c_rf_f) / c_crit_f, 4),
        "zeta_slow_rear": round(0.5 * (c_bs_r + c_rs_r) / c_crit_r, 4),
        "zeta_fast_rear": round(0.5 * (c_bf_r + c_rf_r) / c_crit_r, 4),
        "damping_detail": {
            "front": {
                "bump_slow_n_s_m": round(c_bs_f, 1),
                "rebound_slow_n_s_m": round(c_rs_f, 1),
                "bump_fast_n_s_m": round(c_bf_f, 1),
                "rebound_fast_n_s_m": round(c_rf_f, 1),
                "zeta_bump_slow": round(c_bs_f / c_crit_f, 4),
                "zeta_rebound_slow": round(c_rs_f / c_crit_f, 4),
            },
            "rear": {
                "bump_slow_n_s_m": round(c_bs_r, 1),
                "rebound_slow_n_s_m": round(c_rs_r, 1),
                "bump_fast_n_s_m": round(c_bf_r, 1),
                "rebound_fast_n_s_m": round(c_rf_r, 1),
                "zeta_bump_slow": round(c_bs_r / c_crit_r, 4),
                "zeta_rebound_slow": round(c_rs_r / c_crit_r, 4),
            },
        },
    }


def evaluate_7dof_lap_state(
    df: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    mu_0: float | None = None,
    use_predicted_travel: bool = False,
    cold_pressure_delta_psi: list[float] | None = None,
) -> LapDiagnosticState:
    """Evaluate 7-DOF inverse-dynamics normal loads, instantaneous LLTD, dynamic camber, and grip utilization."""
    fuel_l = float(setup.get("fuel", 2.0))
    phys = resolve_setup_physics(car, setup, fuel_liters=fuel_l)
    aero_rh = _resolve_coupled_aero_and_rh(car, setup)

    m = phys["mass_kg"]
    m_ratio = m / M_CALIB_REF_KG
    k_arb_f = phys["k_arb_f"] * m_ratio
    k_arb_r = phys["k_arb_r"] * m_ratio
    k_eff_f = phys["kw_f"] + k_arb_f
    k_eff_r = phys["kw_r"] + k_arb_r
    fw = car.front_weight_dist
    h_cg = car.cg_height_m
    l_f = car.wheelbase_m * (1.0 - fw)
    l_r = car.wheelbase_m * fw
    i_z = 0.92 * m * l_f * l_r

    dist = df["LAP_DIST_M"].to_numpy(dtype=float) if "LAP_DIST_M" in df.columns else np.arange(len(df), dtype=float)
    v = df["SPEED"].to_numpy(dtype=float)
    ay = df["G_LAT"].to_numpy(dtype=float)
    ax = df["AX_SPEED" if "AX_SPEED" in df.columns else "G_LON"].to_numpy(dtype=float)
    roty = df["ROTY"].to_numpy(dtype=float)
    r_dot = df["R_DOT"].to_numpy(dtype=float) if "R_DOT" in df.columns else np.gradient(roty, df["Time"].to_numpy(dtype=float))
    steer = df["STEERANGLE"].to_numpy(dtype=float) if "STEERANGLE" in df.columns else np.zeros(len(df), dtype=float)
    steer_dot = df["STEER_DOT"].to_numpy(dtype=float) if "STEER_DOT" in df.columns else np.zeros(len(df), dtype=float)

    # Coordinate sign invariance between STEERANGLE and ROTY
    steer_sign = _corr_sign(steer, roty)
    road_steer = (steer_sign * steer) / max(phys["steering_ratio"], 1e-3)
    neutral_steer = (car.wheelbase_m * roty) / np.maximum(v, 8.0)
    cornering_sign = np.where(np.abs(roty) > 0.03, np.sign(roty), 0.0)
    understeer_deg = np.rad2deg((road_steer - neutral_steer) * cornering_sign)
    understeer_deg = np.where(v > 10.0, understeer_deg, 0.0)

    # Free travel & progressive bumpstop equilibrium (Stage 3 core)
    q = 0.5 * AIR_DENSITY * phys["ref_area_m2"] * (v**2)
    z0_f_mm, z0_r_mm = STATIC_TRAVEL_MM
    dfz_pitch = 0.5 * m * ax * (h_cg / max(car.wheelbase_m, 1e-3))
    zf_mean_free = z0_f_mm + (0.5 * q * aero_rh["cl_front"] - dfz_pitch) / (phys["kw_f"] * 1e-3)
    zr_mean_free = z0_r_mm + (0.5 * q * aero_rh["cl_rear"] + dfz_pitch) / (phys["kw_r"] * 1e-3)

    s_f, s_r = 1.0, 1.0
    has_obs_sus = all(f"SUS_TRAVEL_{c}" in df.columns for c in CORNERS)
    if has_obs_sus:
        s_f = _corr_sign((df["SUS_TRAVEL_LF"] - df["SUS_TRAVEL_RF"]).to_numpy(dtype=float), ay)
        s_r = _corr_sign((df["SUS_TRAVEL_LR"] - df["SUS_TRAVEL_RR"]).to_numpy(dtype=float), ay)

    dz_f = s_f * (2.0 * (m * fw) * h_cg / (max(car.track_width_front_m, 1e-3) * k_eff_f)) * ay * 1e3
    dz_r = s_r * (2.0 * (m * (1.0 - fw)) * h_cg / (max(car.track_width_rear_m, 1e-3) * k_eff_r)) * ay * 1e3
    z_free_mm = {
        "LF": zf_mean_free + 0.5 * dz_f,
        "RF": zf_mean_free - 0.5 * dz_f,
        "LR": zr_mean_free + 0.5 * dz_r,
        "RR": zr_mean_free - 0.5 * dz_r,
    }

    bs_rates = setup.get("suspension", {}).get("bumpstop_rate_n_mm", [1000.0, 1000.0, 300.0, 300.0])
    z_mm: dict[str, np.ndarray] = {}
    bs_force_n: dict[str, np.ndarray] = {}
    bs_margin_mm: dict[str, np.ndarray] = {}
    k_tan_n_m: dict[str, np.ndarray] = {}

    for idx, c in enumerate(CORNERS):
        is_f = c in ("LF", "RF")
        kw_n_m = phys["kw_f"] if is_f else phys["kw_r"]
        bs_thr = phys["bs_thresh_f_mm"] if is_f else phys["bs_thresh_r_mm"]
        k_prog = BUMPSTOP_K_PROG_REF * (float(bs_rates[idx]) / 1000.0)
        z_pred, f_pred = _solve_progressive_bumpstop(z_free_mm[c], bs_thr, kw_n_m * 1e-3, k_prog_coef=k_prog, p_exp=BUMPSTOP_P_EXP)

        if has_obs_sus and not use_predicted_travel:
            z_use = df[f"SUS_TRAVEL_{c}"].to_numpy(dtype=float)
            over = np.maximum(0.0, z_use - bs_thr)
            bs_col = f"BUMPSTOP_FORCE_{c}"
            f_use = (
                np.maximum(0.0, -df[bs_col].to_numpy(dtype=float))
                if bs_col in df.columns
                else k_prog * (over**BUMPSTOP_P_EXP)
            )
        else:
            z_use = z_pred
            over = np.maximum(0.0, z_use - bs_thr)
            f_use = f_pred

        z_mm[c] = z_use
        bs_force_n[c] = f_use
        bs_margin_mm[c] = bs_thr - z_use
        # Tangent wheel rate in [N/m] including progressive bumpstop slope dF/dz [N/mm] * 1000
        k_tan_n_m[c] = kw_n_m + 1000.0 * BUMPSTOP_P_EXP * k_prog * (over ** (BUMPSTOP_P_EXP - 1.0))

    # Instantaneous Roll Stiffness & Instantaneous LLTD
    k_roll_f_inst = (0.5 * (k_tan_n_m["LF"] + k_tan_n_m["RF"]) + k_arb_f) * (car.track_width_front_m**2)
    k_roll_r_inst = (0.5 * (k_tan_n_m["LR"] + k_tan_n_m["RR"]) + k_arb_r) * (car.track_width_rear_m**2)
    lltd_share_f = k_roll_f_inst / np.maximum(k_roll_f_inst + k_roll_r_inst, 1e-6)
    lltd_inst_pct = lltd_share_f * 100.0

    # 4-Corner Normal Loads using instantaneous LLTD
    fz_aero_f = 0.5 * q * aero_rh["cl_front"]
    fz_aero_r = 0.5 * q * aero_rh["cl_rear"]
    fz0_f = 0.5 * fw * m * GRAVITY
    fz0_r = 0.5 * (1.0 - fw) * m * GRAVITY
    dfz_lat_f = lltd_share_f * m * ay * (h_cg / max(car.track_width_front_m, 1e-3))
    dfz_lat_r = (1.0 - lltd_share_f) * m * ay * (h_cg / max(car.track_width_rear_m, 1e-3))
    fz_corners_n = {
        "LF": np.clip(fz0_f - dfz_pitch - dfz_lat_f + fz_aero_f, 150.0, None),
        "RF": np.clip(fz0_f - dfz_pitch + dfz_lat_f + fz_aero_f, 150.0, None),
        "LR": np.clip(fz0_r + dfz_pitch - dfz_lat_r + fz_aero_r, 150.0, None),
        "RR": np.clip(fz0_r + dfz_pitch + dfz_lat_r + fz_aero_r, 150.0, None),
    }

    # Dynamic ride heights & Dynamic camber
    zf_mean = 0.5 * (z_mm["LF"] + z_mm["RF"])
    zr_mean = 0.5 * (z_mm["LR"] + z_mm["RR"])
    rh_dyn_front_mm = aero_rh["rh_front_mm"] - (zf_mean - z0_f_mm)
    rh_dyn_rear_mm = aero_rh["rh_rear_mm"] - (zr_mean - z0_r_mm)

    align = setup.get("tyres_and_alignment", {})
    cam_arr = align.get("camber_deg", [-4.0, -4.0, -3.5, -3.5])
    cast_arr = align.get("caster_deg", [14.5, 14.5])
    cam_f0 = 0.5 * (float(cam_arr[0]) + float(cam_arr[1]))
    cam_r0 = 0.5 * (float(cam_arr[2]) + float(cam_arr[3]))
    caster_mean = 0.5 * (float(cast_arr[0]) + float(cast_arr[1]))

    roll_f_deg = np.rad2deg(np.arcsin(np.clip((z_mm["LF"] - z_mm["RF"]) * 1e-3 / max(car.track_width_front_m, 1e-3), -0.5, 0.5)))
    roll_r_deg = np.rad2deg(np.arcsin(np.clip((z_mm["LR"] - z_mm["RR"]) * 1e-3 / max(car.track_width_rear_m, 1e-3), -0.5, 0.5)))
    dyn_camber_outer_front_deg = cam_f0 + 0.65 * np.abs(roll_f_deg) - caster_mean * np.sin(np.abs(road_steer))
    dyn_camber_outer_rear_deg = cam_r0 + 0.65 * np.abs(roll_r_deg)

    # Hot tyre pressure & grip multipliers
    p_deltas = cold_pressure_delta_psi if cold_pressure_delta_psi is not None else [0.0, 0.0, 0.0, 0.0]
    hot_press_psi: dict[str, np.ndarray] = {}
    eta_p_corner: dict[str, np.ndarray] = {}
    for idx, c in enumerate(CORNERS):
        col = f"TYRE_PRESS_{c}"
        p_c = (
            df[col].to_numpy(dtype=float) + float(p_deltas[idx])
            if col in df.columns
            else np.full(len(df), car.tire_p_opt_psi, dtype=float) + float(p_deltas[idx])
        )
        hot_press_psi[c] = p_c
        eta_p_corner[c] = np.exp(-((p_c - car.tire_p_opt_psi) ** 2) / (2.0 * (car.tire_sigma_p_psi**2)))

    eta_p_f = 0.5 * (eta_p_corner["LF"] + eta_p_corner["RF"])
    eta_p_r = 0.5 * (eta_p_corner["LR"] + eta_p_corner["RR"])
    eta_p_mean = 0.5 * (eta_p_f + eta_p_r)

    eta_cam_f = np.clip(1.0 - 0.008 * ((dyn_camber_outer_front_deg - (-2.3)) ** 2), 0.90, 1.02)
    eta_cam_r = np.clip(1.0 - 0.008 * ((dyn_camber_outer_rear_deg - (-2.0)) ** 2), 0.90, 1.02)

    # Inverse-dynamics 3-DOF planar axle force demand (Fx, Fy, Mz equilibrium)
    fy_req_front = (m * l_r * ay + i_z * r_dot) / max(car.wheelbase_m, 1e-3)
    fy_req_rear = (m * l_f * ay - i_z * r_dot) / max(car.wheelbase_m, 1e-3)

    f_drag = q * aero_rh["cd_total"]
    fx_tot = m * ax + f_drag
    brake_bias = float(setup.get("drivetrain_and_brakes", {}).get("brake_bias_pct", 54.6)) / 100.0
    fz_f_tot = fz_corners_n["LF"] + fz_corners_n["RF"]
    fz_r_tot = fz_corners_n["LR"] + fz_corners_n["RR"]
    fz_f_share = fz_f_tot / np.maximum(fz_f_tot + fz_r_tot, 100.0)
    abs_col = df["ABS"].to_numpy(dtype=float) if "ABS" in df.columns else np.zeros(len(df), dtype=float)
    abs_blend = np.maximum(np.clip(abs_col, 0.0, 1.0), np.clip((-ax - 4.0) / 8.0, 0.0, 1.0))
    eff_brake_bias = np.where(
        fz_f_share > brake_bias,
        brake_bias + 0.85 * abs_blend * (fz_f_share - brake_bias),
        brake_bias,
    )
    is_braking = fx_tot < 0.0
    fx_req_front = np.where(is_braking, eff_brake_bias * fx_tot, 0.0)
    fx_req_rear = np.where(is_braking, (1.0 - eff_brake_bias) * fx_tot, fx_tot)

    f_req_front = np.sqrt(fy_req_front**2 + fx_req_front**2)
    f_req_rear = np.sqrt(fy_req_rear**2 + fx_req_rear**2)

    # Load-sensitive tire friction capacity (mu_0 calibrated if None)
    def _cap_unit_mu(fz_l: np.ndarray, fz_r: np.ndarray, eta_p: np.ndarray, eta_cam: np.ndarray) -> np.ndarray:
        s_l = fz_l * np.clip(1.0 - 0.12 * ((fz_l - 3500.0) / 3500.0), 0.72, 1.15)
        s_r = fz_r * np.clip(1.0 - 0.12 * ((fz_r - 3500.0) / 3500.0), 0.72, 1.15)
        return (s_l + s_r) * eta_p * eta_cam

    cap1_f = np.maximum(_cap_unit_mu(fz_corners_n["LF"], fz_corners_n["RF"], eta_p_f, eta_cam_f), 500.0)
    cap1_r = np.maximum(_cap_unit_mu(fz_corners_n["LR"], fz_corners_n["RR"], eta_p_r, eta_cam_r), 500.0)

    if mu_0 is None:
        ratio_unit = np.maximum(f_req_front / cap1_f, f_req_rear / cap1_r)
        active = (np.abs(ay) > 4.0) | (ax < -4.0)
        ref_q = float(np.percentile(ratio_unit[active] if np.any(active) else ratio_unit, 98.0))
        mu_0_eff = float(np.clip(ref_q / 0.98, 1.2, 2.5))
    else:
        mu_0_eff = float(mu_0)

    f_cap_front = mu_0_eff * cap1_f
    f_cap_rear = mu_0_eff * cap1_r
    rho_front = f_req_front / np.maximum(f_cap_front, 100.0)
    rho_rear = f_req_rear / np.maximum(f_cap_rear, 100.0)

    return LapDiagnosticState(
        dist_m=dist,
        speed_ms=v,
        ay_mps2=ay,
        ax_mps2=ax,
        roty_rad_s=roty,
        r_dot_rad_s2=r_dot,
        steer_rad=steer,
        steer_dot_rad_s=steer_dot,
        understeer_deg=understeer_deg,
        z_mm=z_mm,
        bs_force_n=bs_force_n,
        bs_margin_mm=bs_margin_mm,
        fz_corners_n=fz_corners_n,
        lltd_inst_pct=lltd_inst_pct,
        rh_dyn_front_mm=rh_dyn_front_mm,
        rh_dyn_rear_mm=rh_dyn_rear_mm,
        dyn_camber_outer_front_deg=dyn_camber_outer_front_deg,
        dyn_camber_outer_rear_deg=dyn_camber_outer_rear_deg,
        fy_req_front_n=fy_req_front,
        fy_req_rear_n=fy_req_rear,
        f_req_front_n=f_req_front,
        f_req_rear_n=f_req_rear,
        f_cap_front_n=f_cap_front,
        f_cap_rear_n=f_cap_rear,
        rho_front=rho_front,
        rho_rear=rho_rear,
        eta_p_mean=eta_p_mean,
        hot_press_psi=hot_press_psi,
        mu_0=mu_0_eff,
    )


def classify_grip_state(rho_front_peak: float, rho_rear_peak: float) -> str:
    """Classify axle saturation quadrant from peak front/rear friction-circle utilization."""
    if rho_front_peak < 0.85 and rho_rear_peak < 0.85:
        return "DRIVER_MARGIN"
    if rho_front_peak - rho_rear_peak > 0.05:
        return "FRONT_SATURATED"
    if rho_rear_peak - rho_front_peak > 0.05:
        return "REAR_SATURATED"
    return "BALANCED_LIMIT"


def _corner_phase_masks(dist: np.ndarray, corner: CornerRegion) -> dict[str, np.ndarray]:
    m_all = (dist >= corner.start_m) & (dist <= corner.end_m)
    return m_all


def _split_entry_apex_exit(dist: np.ndarray, speed: np.ndarray, start_m: float, end_m: float) -> dict[str, np.ndarray]:
    m_all = (dist >= start_m) & (dist <= end_m)
    span = end_m - start_m
    mid_lo, mid_hi = start_m + 0.2 * span, start_m + 0.8 * span
    m_mid = m_all & (dist >= mid_lo) & (dist <= mid_hi)
    sub_mask = m_mid if np.any(m_mid) else m_all
    if not np.any(sub_mask):
        return {"entry": m_all, "apex": m_all, "exit": m_all}

    idx_candidates = np.where(sub_mask)[0]
    apex_idx = idx_candidates[int(np.argmin(speed[idx_candidates]))]
    apex_d = float(dist[apex_idx])

    m_entry = m_all & (dist < apex_d - 20.0)
    m_apex = m_all & (dist >= apex_d - 35.0) & (dist <= apex_d + 35.0)
    m_exit = m_all & (dist > apex_d + 20.0)
    if not np.any(m_entry):
        m_entry = m_all & (dist <= apex_d)
    if not np.any(m_exit):
        m_exit = m_all & (dist >= apex_d)
    return {"entry": m_entry, "apex": m_apex, "exit": m_exit}


def _turn_in_gain_and_lag_ms(
    time_s: np.ndarray,
    steer_dot: np.ndarray,
    r_dot: np.ndarray,
    steering_ratio: float,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Compute entry steering-rate-to-yaw-accel gain [1/s] and response lag [ms]."""
    if np.sum(mask) < 6:
        return 0.0, 0.0
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.05
    u = steer_dot[mask] / max(steering_ratio, 1e-3)
    y = r_dot[mask]
    if float(np.std(u)) < 1e-4 or float(np.std(y)) < 1e-4:
        return 0.0, 0.0

    max_k = min(4, len(u) // 3)
    best_lag_k = 0
    best_corr = -1.0
    best_gain = 0.0
    for k in range(0, max_k + 1):
        u_s = u[: len(u) - k] if k > 0 else u
        y_s = y[k:]
        if float(np.std(u_s)) < 1e-5 or float(np.std(y_s)) < 1e-5:
            continue
        c = abs(float(np.corrcoef(u_s, y_s)[0, 1]))
        if c > best_corr:
            best_corr = c
            best_lag_k = k
            var_u = float(np.var(u_s))
            best_gain = abs(float(np.cov(u_s, y_s)[0, 1]) / max(var_u, 1e-8))
    return best_gain, best_lag_k * dt * 1000.0


def extract_corner_diagnostics(
    df: pd.DataFrame,
    state: LapDiagnosticState,
    track: TrackConfig,
    baseline: dict[str, Any],
    setup: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract per-corner Entry/Apex/Exit diagnostics, utilization, margins, and alignment proxies."""
    time_s = df["Time"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.05
    throttle = df["THROTTLE"].to_numpy(dtype=float) if "THROTTLE" in df.columns else np.zeros(len(df))
    brake = df["BRAKE"].to_numpy(dtype=float) if "BRAKE" in df.columns else np.zeros(len(df))
    tc_col = df["TC"].to_numpy(dtype=float) if "TC" in df.columns else np.zeros(len(df))

    steer_jerk = np.gradient(state.steer_dot_rad_s, dt)
    brake_jerk = np.gradient(brake, dt)
    throttle_jerk = np.gradient(throttle, dt)
    # High-frequency yaw oscillation metric (deviations of 2nd difference of roty)
    yaw_hf = np.zeros_like(state.roty_rad_s)
    if len(state.roty_rad_s) >= 3:
        yaw_hf[1:-1] = state.roty_rad_s[1:-1] - 0.5 * (state.roty_rad_s[:-2] + state.roty_rad_s[2:])

    steering_ratio = float(setup.get("steering_ratio", 16.0))
    static_lltd = float(baseline["static_lltd_front_pct"])

    corners_out: list[dict[str, Any]] = []
    for corner in track.corners:
        m_c = (state.dist_m >= corner.start_m) & (state.dist_m <= corner.end_m)
        if not np.any(m_c):
            continue

        phases = _split_entry_apex_exit(state.dist_m, state.speed_ms, corner.start_m, corner.end_m)
        phase_summary: dict[str, dict[str, Any]] = {}
        for p_name, m_p in phases.items():
            if not np.any(m_p):
                continue
            rf_p = float(np.percentile(state.rho_front[m_p], 95.0))
            rr_p = float(np.percentile(state.rho_rear[m_p], 95.0))
            phase_summary[p_name] = {
                "rho_front_mean": round(float(np.mean(state.rho_front[m_p])), 4),
                "rho_front_peak": round(rf_p, 4),
                "rho_rear_mean": round(float(np.mean(state.rho_rear[m_p])), 4),
                "rho_rear_peak": round(rr_p, 4),
                "understeer_angle_deg_mean": round(float(np.mean(state.understeer_deg[m_p])), 2),
                "state_label": classify_grip_state(rf_p, rr_p),
            }

        rho_f_peak = float(np.percentile(state.rho_front[m_c], 95.0))
        rho_r_peak = float(np.percentile(state.rho_rear[m_c], 95.0))
        lltd_peak = float(np.max(state.lltd_inst_pct[m_c]))
        lltd_mean = float(np.mean(state.lltd_inst_pct[m_c]))

        max_lat_g = float(np.max(np.abs(state.ay_mps2[m_c])) / GRAVITY)
        max_brk_g = float(np.max(np.maximum(0.0, -state.ax_mps2[m_c])) / GRAVITY)
        max_acc_g = float(np.max(np.maximum(0.0, state.ax_mps2[m_c])) / GRAVITY)

        gain, lag_ms = _turn_in_gain_and_lag_ms(time_s, state.steer_dot_rad_s, state.r_dot_rad_s2, steering_ratio, phases["entry"])
        m_exit = phases["exit"]
        low_steer_exit = m_exit & (np.abs(np.rad2deg(state.steer_rad / max(steering_ratio, 1e-3))) < 12.0) & (throttle > 30.0)
        tc_freq_pct = float(100.0 * np.mean(tc_col[low_steer_exit] > 0.5)) if np.any(low_steer_exit) else 0.0

        corners_out.append(
            {
                "name": corner.name,
                "start_m": corner.start_m,
                "end_m": corner.end_m,
                "speed_class": corner.speed_class,
                "aero_sensitivity": corner.aero_sensitivity,
                "kerb_usage": corner.kerb_usage,
                "n_samples": int(np.sum(m_c)),
                "performance": {
                    "min_speed_kmh": round(float(np.min(state.speed_ms[m_c]) * 3.6), 1),
                    "max_speed_kmh": round(float(np.max(state.speed_ms[m_c]) * 3.6), 1),
                    "max_lat_g": round(max_lat_g, 3),
                    "max_lon_braking_g": round(max_brk_g, 3),
                    "max_lon_accel_g": round(max_acc_g, 3),
                    "understeer_angle_deg_mean": round(float(np.mean(state.understeer_deg[m_c])), 2),
                    "understeer_angle_deg_peak": round(float(np.max(np.abs(state.understeer_deg[m_c]))), 2),
                    "steer_jerk_rms_rad_s2": round(float(np.sqrt(np.mean(steer_jerk[m_c] ** 2))), 2),
                    "brake_jerk_rms_pct_s": round(float(np.sqrt(np.mean(brake_jerk[m_c] ** 2))), 1),
                    "throttle_jerk_rms_pct_s": round(float(np.sqrt(np.mean(throttle_jerk[m_c] ** 2))), 1),
                },
                "grip_utilization": {
                    "rho_front_mean": round(float(np.mean(state.rho_front[m_c])), 4),
                    "rho_front_peak": round(rho_f_peak, 4),
                    "rho_rear_mean": round(float(np.mean(state.rho_rear[m_c])), 4),
                    "rho_rear_peak": round(rho_r_peak, 4),
                    "state_label": classify_grip_state(rho_f_peak, rho_r_peak),
                    "phases": phase_summary,
                },
                "nonlinear_margins": {
                    "lltd_instant_peak_pct": round(lltd_peak, 2),
                    "lltd_instant_mean_pct": round(lltd_mean, 2),
                    "lltd_jump_from_static_pct": round(lltd_peak - static_lltd, 2),
                    "bumpstop_min_margin_mm": {c: round(float(np.min(state.bs_margin_mm[c][m_c])), 2) for c in CORNERS},
                    "bumpstop_peak_force_n": {c: round(float(np.max(state.bs_force_n[c][m_c])), 1) for c in CORNERS},
                    "tyre_pressure_grip_loss_pct": round(float((1.0 - np.mean(state.eta_p_mean[m_c])) * 100.0), 3),
                    "hot_pressure_mean_psi": {c: round(float(np.mean(state.hot_press_psi[c][m_c])), 2) for c in CORNERS},
                    "min_ground_clearance_mm": {
                        "front": round(float(np.min(state.rh_dyn_front_mm[m_c])), 1),
                        "rear": round(float(np.min(state.rh_dyn_rear_mm[m_c])), 1),
                    },
                },
                "alignment_proxies": {
                    "dynamic_camber_outer_front_deg_mean": round(float(np.mean(state.dyn_camber_outer_front_deg[m_c])), 2),
                    "dynamic_camber_outer_front_deg_peak": round(float(np.min(state.dyn_camber_outer_front_deg[m_c])), 2),
                    "braking_to_cornering_grip_ratio": round(max_brk_g / max(max_lat_g, 0.2), 3),
                    "front_toe_turn_in_gain": round(gain, 3),
                    "front_toe_turn_in_lag_ms": round(lag_ms, 1),
                    "rear_toe_high_speed_yaw_osc_rms_rad_s": round(float(np.sqrt(np.mean(yaw_hf[m_c] ** 2))), 4),
                    "exit_low_steer_tc_freq_pct": round(tc_freq_pct, 1),
                },
            }
        )
    return corners_out


def build_physics_model_spec(car: CarConfig) -> dict[str, Any]:
    """Export Stage 3/4 constitutive formulas, calibrated click-to-SI slopes, and ACC bounds."""
    return {
        "constitutive_equations": {
            "progressive_bumpstop_force_n": "F_bs = k_prog * max(0, z_mm - bs_thresh_mm)^2.73, where k_prog = 1.65 * (bumpstop_rate_n_mm / 1000.0)",
            "tangent_wheel_rate_n_m": "k_tan = k_w + 1000 * 2.73 * k_prog * max(0, z_mm - bs_thresh_mm)^1.73",
            "instantaneous_lltd_front": "LLTD_F(t) = K_phi_F_inst / (K_phi_F_inst + K_phi_R_inst), K_phi_axle = (mean(k_tan) + k_arb * (m / 1168.7)) * track_width^2",
            "two_speed_damper_n_s_m": "c(v_susp, click) = (base_slow + slope_slow * click) if |v_susp| <= 0.05 m/s else (base_fast + slope_fast * click), scaled by sqrt(m / 1168.7)",
            "critical_damping_ratio": "zeta = c_eff / (2 * sqrt(k_w * m_corner))",
            "aero_downforce_and_drag_n": "F_z_aero = 0.5 * 1.225 * A_ref * Cl(rear_wing, rake, rh) * v^2; F_d = 0.5 * 1.225 * A_ref * Cd(rear_wing, ducts, rake) * v^2",
            "dynamic_camber_deg": "gamma_dyn_outer_F = camber_static_F + 0.65 * |roll_F_deg| - caster_deg * sin(|steer_rad / steer_ratio|)",
            "inverse_dynamics_grip_utilization": "Fy_req_F = (m * l_r * ay + I_z * r_dot) / L; Fy_req_R = (m * l_f * ay - I_z * r_dot) / L; rho = F_req / F_cap(Fz, eta_P, eta_cam)",
        },
        "click_to_physics_slopes": {
            "aero_baseline": {
                "reference_area_m2": car.aero_baseline.reference_area_m2,
                "base_cl": car.aero_baseline.base_cl,
                "base_cd": car.aero_baseline.base_cd,
                "base_aero_front_dist": car.aero_baseline.base_aero_front_dist,
            },
            "calibration_slots": car.calibration_slots,
            "calibration_reference_mass_kg": M_CALIB_REF_KG,
        },
        "valid_setup_bounds": car.setup_bounds,
    }


def build_expected_ai_response_schema(setup: dict[str, Any]) -> dict[str, Any]:
    """Provide strict JSON schema and prompt guidance for the external AI setup candidate."""
    return {
        "instructions": (
            "Analyze vehicle_and_setup_baseline, corners, nonlinear_margins, alignment_proxies, and optional "
            "ui_tyre_diagnostics. Estimate coordinated multi-parameter setup adjustments using the formulas "
            "and slopes in physics_model_spec strictly within physics_model_spec.valid_setup_bounds. "
            "All exact nonlinear 7-DOF coupled effects will be replayed by `python main.py simulate`."
        ),
        "required_top_level_keys": [
            "setup_name",
            "tyres_and_alignment",
            "suspension",
            "drivetrain_and_brakes",
            "dampers_clicks",
            "aero_clicks",
            "ride_height_mm",
        ],
        "candidate_template": {
            "setup_name": f"{setup.get('setup_name', 'setup')}_v2_candidate",
            "reasoning_summary": "<concise physical rationale linking bottlenecks to coordinated parameter changes>",
            "tyres_and_alignment": setup.get("tyres_and_alignment", {}),
            "suspension": setup.get("suspension", {}),
            "steering_ratio": setup.get("steering_ratio", 16.0),
            "drivetrain_and_brakes": setup.get("drivetrain_and_brakes", {}),
            "dampers_clicks": setup.get("dampers_clicks", {}),
            "aero_clicks": setup.get("aero_clicks", {}),
            "ride_height_mm": setup.get("ride_height_mm", [66, 54]),
            "fuel": setup.get("fuel", 2),
        },
    }


def build_ai_handoff_payload(
    csv_path: str | Path,
    df: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    track: TrackConfig,
    state: LapDiagnosticState,
    ui_tyres: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete `ai_handoff.json` payload."""
    baseline = compute_vehicle_coupled_baseline(car, setup)
    corners = extract_corner_diagnostics(df, state, track, baseline, setup)

    quadrant_counts = {"DRIVER_MARGIN": 0, "FRONT_SATURATED": 0, "REAR_SATURATED": 0, "BALANCED_LIMIT": 0}
    quadrant_points = []
    for c in corners:
        gu = c["grip_utilization"]
        nm = c["nonlinear_margins"]
        lbl = gu["state_label"]
        quadrant_counts[lbl] = quadrant_counts.get(lbl, 0) + 1
        quadrant_points.append(
            {
                "corner": c["name"],
                "speed_class": c["speed_class"],
                "aero_sensitivity": c["aero_sensitivity"],
                "state_label": lbl,
                "rho_front_peak": gu["rho_front_peak"],
                "rho_rear_peak": gu["rho_rear_peak"],
                "rho_front_mean": gu["rho_front_mean"],
                "rho_rear_mean": gu["rho_rear_mean"],
                "axle_balance_delta_rho_f_minus_r": round(gu["rho_front_peak"] - gu["rho_rear_peak"], 4),
                "lltd_instant_peak_pct": nm["lltd_instant_peak_pct"],
                "lltd_jump_from_static_pct": nm["lltd_jump_from_static_pct"],
                "front_bumpstop_min_margin_mm": min(nm["bumpstop_min_margin_mm"]["LF"], nm["bumpstop_min_margin_mm"]["RF"]),
                "rear_bumpstop_min_margin_mm": min(nm["bumpstop_min_margin_mm"]["LR"], nm["bumpstop_min_margin_mm"]["RR"]),
            }
        )

    return {
        "metadata": {
            "csv_path": str(csv_path),
            "vehicle_id": car.car_id,
            "track_id": track.track_id,
            "track_name": track.name,
            "n_samples": int(len(df)),
            "lap_distance_min_m": round(float(np.min(state.dist_m)), 1),
            "lap_distance_max_m": round(float(np.max(state.dist_m)), 1),
            "calibrated_tire_mu_0": round(state.mu_0, 4),
        },
        "vehicle_and_setup_baseline": baseline,
        "current_setup": setup,
        "ui_tyre_diagnostics": ui_tyres,
        "four_quadrant_summary": {
            "quadrant_counts": quadrant_counts,
            "lap_peak_lltd_front_pct": round(float(np.max(state.lltd_inst_pct)), 2),
            "lap_min_front_bumpstop_margin_mm": round(float(min(np.min(state.bs_margin_mm["LF"]), np.min(state.bs_margin_mm["RF"]))), 2),
            "lap_min_rear_bumpstop_margin_mm": round(float(min(np.min(state.bs_margin_mm["LR"]), np.min(state.bs_margin_mm["RR"]))), 2),
            "corners": quadrant_points,
        },
        "corners": corners,
        "physics_model_spec": build_physics_model_spec(car),
        "expected_ai_response_schema": build_expected_ai_response_schema(setup),
    }


def _deep_merge_setup(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    payload = candidate.get("candidate_setup", candidate) if isinstance(candidate.get("candidate_setup"), dict) else candidate
    for k, v in payload.items():
        if k == "reasoning_summary":
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            sub = dict(merged[k])
            sub.update(v)
            merged[k] = sub
        else:
            merged[k] = v
    return merged


def _validate_setup_bounds(setup: dict[str, Any], bounds: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate setup values against ``vehicle.yaml`` setup_bounds."""
    violations: list[dict[str, Any]] = []
    if not bounds:
        return {"valid": True, "violations": []}

    def _check_val(path: str, val: Any, b: dict[str, Any], idx: int | None = None) -> None:
        v = float(val)
        opts = b.get("front_options" if idx is not None and idx < 2 and "front_options" in b else "rear_options" if idx is not None and idx >= 2 and "rear_options" in b else "options")
        if opts is not None:
            if not any(abs(v - float(o)) <= 1e-6 for o in opts):
                violations.append({"param": f"{path}[{idx}]" if idx is not None else path, "value": v, "allowed_options": opts})
            return
        lo = b.get("front_min" if idx is not None and idx < 2 and "front_min" in b else "rear_min" if idx is not None and idx >= 2 and "rear_min" in b else "min")
        hi = b.get("front_max" if idx is not None and idx < 2 and "front_max" in b else "rear_max" if idx is not None and idx >= 2 and "rear_max" in b else "max")
        if path == "ride_height_mm" and idx is not None:
            lo = b.get("rear_min" if idx == 0 else "front_min", lo)
            hi = b.get("rear_max" if idx == 0 else "front_max", hi)
        elif path.endswith("arb_clicks") and idx is not None:
            lo = b.get("front_min" if idx == 0 else "rear_min", lo)
            hi = b.get("front_max" if idx == 0 else "rear_max", hi)
        if lo is not None and v < float(lo) - 1e-6:
            violations.append({"param": f"{path}[{idx}]" if idx is not None else path, "value": v, "min": lo, "max": hi})
        if hi is not None and v > float(hi) + 1e-6:
            violations.append({"param": f"{path}[{idx}]" if idx is not None else path, "value": v, "min": lo, "max": hi})

    def _walk(s_node: dict[str, Any], b_node: dict[str, Any], prefix: str = "") -> None:
        for key, b_spec in b_node.items():
            if key not in s_node or not isinstance(b_spec, dict):
                continue
            val = s_node[key]
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                _walk(val, b_spec, full_key)
            elif isinstance(val, list):
                for i, elem in enumerate(val):
                    _check_val(full_key, elem, b_spec, idx=i)
            else:
                _check_val(full_key, val, b_spec)

    _walk(setup, bounds)
    return {"valid": len(violations) == 0, "violations": violations}


def _diff_dict_params(base: dict[str, Any], cand: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for k in sorted(set(base.keys()) | set(cand.keys())):
        if k in ("setup_name", "vehicle_id", "reasoning_summary"):
            continue
        b_v = base.get(k)
        c_v = cand.get(k)
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(b_v, dict) and isinstance(c_v, dict):
            changes.update(_diff_dict_params(b_v, c_v, full))
        elif b_v != c_v:
            changes[full] = {"v1": b_v, "v2": c_v}
    return changes


def run_counterfactual_simulation(
    df: pd.DataFrame,
    car: CarConfig,
    base_setup: dict[str, Any],
    candidate_setup_raw: dict[str, Any],
    track: TrackConfig,
    calib_dir: str | Path | None = None,
) -> tuple[dict[str, Any], LapDiagnosticState, LapDiagnosticState]:
    """Run counterfactual 7-DOF physics replay comparing ``base_setup`` (v1) vs ``candidate_setup`` (v2)."""
    cand_setup = _deep_merge_setup(base_setup, candidate_setup_raw)
    bounds_check = _validate_setup_bounds(cand_setup, car.setup_bounds)

    # Calibrate baseline mu_0 from observed state once so v1 and v2 share identical tire limit baseline
    obs_state = evaluate_7dof_lap_state(df, car, base_setup, mu_0=None, use_predicted_travel=False)
    state_v1 = evaluate_7dof_lap_state(df, car, base_setup, mu_0=obs_state.mu_0, use_predicted_travel=True)

    p1 = base_setup.get("tyres_and_alignment", {}).get("cold_pressure_psi", [25.0, 25.0, 25.0, 25.0])
    p2 = cand_setup.get("tyres_and_alignment", {}).get("cold_pressure_psi", p1)
    p_delta = [float(p2[i]) - float(p1[i]) for i in range(4)]
    state_v2 = evaluate_7dof_lap_state(
        df, car, cand_setup, mu_0=obs_state.mu_0, use_predicted_travel=True, cold_pressure_delta_psi=p_delta
    )

    base_v1 = compute_vehicle_coupled_baseline(car, base_setup)
    base_v2 = compute_vehicle_coupled_baseline(car, cand_setup)
    corners_v1 = extract_corner_diagnostics(df, state_v1, track, base_v1, base_setup)
    corners_v2 = extract_corner_diagnostics(df, state_v2, track, base_v2, cand_setup)

    # Main straight terminal speed comparison
    v1_end_kmh = float(np.max(state_v1.speed_ms[state_v1.dist_m <= 710.0]) * 3.6) if np.any(state_v1.dist_m <= 710.0) else float(np.max(state_v1.speed_ms) * 3.6)
    v2_end_kmh = v1_end_kmh
    if calib_dir is not None and (Path(calib_dir) / "test1.csv").exists() and np.any((df["LAP_DIST_M"] <= 700.0) & (df["THROTTLE"] == 100)):
        a1 = verify_straight_aero_drag(df, car, base_setup, calib_dir, fuel_liters=float(base_setup.get("fuel", 2.0)))
        a2 = verify_straight_aero_drag(df, car, cand_setup, calib_dir, fuel_liters=float(cand_setup.get("fuel", 2.0)))
        v1_end_kmh = a1.end_v_pred_kmh
        v2_end_kmh = a2.end_v_pred_kmh
    else:
        # Analytical straight-line aero-drag delta estimate over 700m
        d_cd = float(base_v2["cd_total"]) - float(base_v1["cd_total"])
        m_avg = 0.5 * (base_v1["total_mass_kg"] + base_v2["total_mass_kg"])
        q_avg = 0.5 * AIR_DENSITY * (car.aero_baseline.reference_area_m2 or 2.08) * (66.0**2)
        d_v_ms = -(q_avg * d_cd / m_avg) * (700.0 / 60.0)
        v2_end_kmh = v1_end_kmh + d_v_ms * 3.6

    corner_diffs: list[dict[str, Any]] = []
    for c1, c2 in zip(corners_v1, corners_v2):
        nm1, nm2 = c1["nonlinear_margins"], c2["nonlinear_margins"]
        gu1, gu2 = c1["grip_utilization"], c2["grip_utilization"]
        al1, al2 = c1["alignment_proxies"], c2["alignment_proxies"]
        min_bs1 = min(nm1["bumpstop_min_margin_mm"]["LF"], nm1["bumpstop_min_margin_mm"]["RF"])
        min_bs2 = min(nm2["bumpstop_min_margin_mm"]["LF"], nm2["bumpstop_min_margin_mm"]["RF"])
        corner_diffs.append(
            {
                "name": c1["name"],
                "lltd_instant_peak_pct": {"v1": nm1["lltd_instant_peak_pct"], "v2": nm2["lltd_instant_peak_pct"], "delta": round(nm2["lltd_instant_peak_pct"] - nm1["lltd_instant_peak_pct"], 2)},
                "front_bumpstop_min_margin_mm": {"v1": round(min_bs1, 2), "v2": round(min_bs2, 2), "delta": round(min_bs2 - min_bs1, 2)},
                "front_bumpstop_peak_force_n": {
                    "v1": max(nm1["bumpstop_peak_force_n"]["LF"], nm1["bumpstop_peak_force_n"]["RF"]),
                    "v2": max(nm2["bumpstop_peak_force_n"]["LF"], nm2["bumpstop_peak_force_n"]["RF"]),
                },
                "rho_front_peak": {"v1": gu1["rho_front_peak"], "v2": gu2["rho_front_peak"], "delta": round(gu2["rho_front_peak"] - gu1["rho_front_peak"], 4)},
                "rho_rear_peak": {"v1": gu1["rho_rear_peak"], "v2": gu2["rho_rear_peak"], "delta": round(gu2["rho_rear_peak"] - gu1["rho_rear_peak"], 4)},
                "state_label": {"v1": gu1["state_label"], "v2": gu2["state_label"]},
                "dynamic_camber_outer_front_deg_mean": {"v1": al1["dynamic_camber_outer_front_deg_mean"], "v2": al2["dynamic_camber_outer_front_deg_mean"]},
                "min_ground_clearance_front_mm": {"v1": nm1["min_ground_clearance_mm"]["front"], "v2": nm2["min_ground_clearance_mm"]["front"]},
                "min_ground_clearance_rear_mm": {"v1": nm1["min_ground_clearance_mm"]["rear"], "v2": nm2["min_ground_clearance_mm"]["rear"]},
            }
        )

    min_rh_f_v2 = float(np.min(state_v2.rh_dyn_front_mm))
    min_rh_r_v2 = float(np.min(state_v2.rh_dyn_rear_mm))
    report = {
        "parameter_changes": _diff_dict_params(base_setup, cand_setup),
        "bounds_validation": bounds_check,
        "baseline_diff": {
            "static_lltd_front_pct": {"v1": base_v1["static_lltd_front_pct"], "v2": base_v2["static_lltd_front_pct"], "delta": round(base_v2["static_lltd_front_pct"] - base_v1["static_lltd_front_pct"], 2)},
            "static_rake_mm": {"v1": base_v1["static_rake_mm"], "v2": base_v2["static_rake_mm"], "delta": round(base_v2["static_rake_mm"] - base_v1["static_rake_mm"], 1)},
            "aero_front_balance_pct": {"v1": base_v1["aero_front_balance_pct"], "v2": base_v2["aero_front_balance_pct"], "delta": round(base_v2["aero_front_balance_pct"] - base_v1["aero_front_balance_pct"], 2)},
            "zeta_slow_front": {"v1": base_v1["zeta_slow_front"], "v2": base_v2["zeta_slow_front"], "delta": round(base_v2["zeta_slow_front"] - base_v1["zeta_slow_front"], 4)},
            "zeta_slow_rear": {"v1": base_v1["zeta_slow_rear"], "v2": base_v2["zeta_slow_rear"], "delta": round(base_v2["zeta_slow_rear"] - base_v1["zeta_slow_rear"], 4)},
            "main_straight_end_speed_kmh": {"v1": round(v1_end_kmh, 2), "v2": round(v2_end_kmh, 2), "delta": round(v2_end_kmh - v1_end_kmh, 2)},
            "min_dynamic_ground_clearance_mm": {
                "front_v2": round(min_rh_f_v2, 1),
                "rear_v2": round(min_rh_r_v2, 1),
                "bottoming_risk": bool(min_rh_f_v2 < 2.0 or min_rh_r_v2 < 2.0),
            },
        },
        "corner_diffs": corner_diffs,
    }
    return report, state_v1, state_v2


def save_diagnose_plots(
    state: LapDiagnosticState,
    corners: list[dict[str, Any]],
    baseline: dict[str, Any],
    out_json_path: str | Path,
) -> list[Path]:
    """Save Stage 4 quadrant utilization map and instantaneous LLTD + bumpstop trace."""
    p = Path(out_json_path)
    out_dir = p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = p.stem

    # 1. Track Four-Quadrant Utilization Plot
    fig1, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = {"DRIVER_MARGIN": "tab:green", "FRONT_SATURATED": "tab:red", "REAR_SATURATED": "tab:orange", "BALANCED_LIMIT": "tab:blue"}
    for c in corners:
        rf = c["grip_utilization"]["rho_front_peak"]
        rr = c["grip_utilization"]["rho_rear_peak"]
        lbl = c["grip_utilization"]["state_label"]
        ax.scatter(rf, rr, s=110, color=colors.get(lbl, "gray"), edgecolors="black", zorder=3)
        ax.annotate(c["name"], (rf, rr), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    ax.axvline(0.85, color="gray", ls="--", lw=1.0)
    ax.axhline(0.85, color="gray", ls="--", lw=1.0)
    ax.plot([0.5, 1.15], [0.5, 1.15], "k:", lw=1.0, label="Balanced Limit (rho_F = rho_R)")
    ax.set_xlabel("Front Axle Peak Utilization rho_F")
    ax.set_ylabel("Rear Axle Peak Utilization rho_R")
    ax.set_title("Stage 4 Track Corner Four-Quadrant Grip Utilization")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)
    p1 = out_dir / f"{stem}_quadrant_utilization.png"
    fig1.tight_layout()
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)

    # 2. Instantaneous LLTD & Front Bumpstop Margin Trace
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
    ax1.plot(state.dist_m, state.lltd_inst_pct, color="tab:purple", lw=1.4, label="Instantaneous LLTD Front (%)")
    ax1.axhline(baseline["static_lltd_front_pct"], color="k", ls="--", lw=1.1, label=f"Static LLTD ({baseline['static_lltd_front_pct']:.1f}%)")
    ax1.set_ylabel("Front LLTD (%)")
    ax1.set_title("Instantaneous LLTD Jump & Front Progressive Bumpstop Margin")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.plot(state.dist_m, state.bs_margin_mm["LF"], color="tab:blue", lw=1.3, label="LF Bumpstop Margin (mm)")
    ax2.plot(state.dist_m, state.bs_margin_mm["RF"], color="tab:orange", lw=1.3, label="RF Bumpstop Margin (mm)")
    ax2.axhline(0.0, color="tab:red", ls="--", lw=1.1, label="Bumpstop Contact (0 mm; <0 = Penetration)")
    ax2.set_xlabel("Track Distance (m)")
    ax2.set_ylabel("Bumpstop Margin (mm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)
    p2 = out_dir / f"{stem}_lltd_bumpstop.png"
    fig2.tight_layout()
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    return [p1, p2]


def save_simulate_plots(
    state_v1: LapDiagnosticState,
    state_v2: LapDiagnosticState,
    out_json_path: str | Path,
) -> Path:
    """Save counterfactual v1 vs v2 LLTD, front bumpstop margin, and grip utilization comparison plot."""
    p = Path(out_json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plot_path = p.parent / f"{p.stem}_comparison.png"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True)
    ax1.plot(state_v1.dist_m, state_v1.lltd_inst_pct, "k-", lw=1.4, label="v1 Baseline LLTD_F (%)")
    ax1.plot(state_v2.dist_m, state_v2.lltd_inst_pct, "tab:blue", ls="--", lw=1.4, label="v2 Candidate LLTD_F (%)")
    ax1.set_ylabel("Instant LLTD (%)")
    ax1.set_title("Counterfactual Setup Simulation: v1 Baseline vs. v2 Candidate")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    min_bs_v1 = np.minimum(state_v1.bs_margin_mm["LF"], state_v1.bs_margin_mm["RF"])
    min_bs_v2 = np.minimum(state_v2.bs_margin_mm["LF"], state_v2.bs_margin_mm["RF"])
    ax2.plot(state_v1.dist_m, min_bs_v1, "k-", lw=1.4, label="v1 Min Front BS Margin (mm)")
    ax2.plot(state_v2.dist_m, min_bs_v2, "tab:green", ls="--", lw=1.4, label="v2 Min Front BS Margin (mm)")
    ax2.axhline(0.0, color="tab:red", ls=":", lw=1.0)
    ax2.set_ylabel("Front BS Margin (mm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    ax3.plot(state_v1.dist_m, state_v1.rho_front, "k-", lw=1.3, alpha=0.8, label="v1 Front rho_F")
    ax3.plot(state_v2.dist_m, state_v2.rho_front, "tab:orange", ls="--", lw=1.3, label="v2 Front rho_F")
    ax3.set_xlabel("Track Distance (m)")
    ax3.set_ylabel("Grip Utilization rho_F")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path
