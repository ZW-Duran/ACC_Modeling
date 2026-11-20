"""Stage 3 Verification: Full-Chassis Suspension, Aero Drag & Bumpstop Physics Verification.

Merges zero-shot calibration coefficient verification with full-chassis suspension
and longitudinal aero-drag verification across five continuous Barcelona track segments:
  1. Main Straight (0-710 m): Rear wing + brake duct drag speed curve & aero downforce
  2. T1 Heavy Braking (700-960 m): Longitudinal pitch transfer & progressive front bumpstop
  3. T3 High-Speed Long Corner (960-1550 m): Aero + ARB lateral load transfer
  4. T4-T5 Trail Braking & Turn-In (1550-2350 m): Combined diagonal load transfer
  5. T10-T16 Continuous Complex to Finish (3150-4586 m): T10 braking + T11 curb + T12-T16
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.calibration.blackbox import _load_calib_csv, _wot_ax_curve
from src.calibration.verification import AIR_DENSITY, CORNERS, GRAVITY, resolve_setup_physics
from src.config import CarConfig

SCENARIO_WINDOWS_M: tuple[tuple[str, str, float, float], ...] = (
    ("1_main_straight", "1. Main Straight (Aero Drag & Pitch Attitude)", 0.0, 710.0),
    ("2_t1_braking", "2. T1 Heavy Braking (Pitch & Progressive Bumpstop)", 700.0, 960.0),
    ("3_t3_long_corner", "3. T3 High-Speed Long Corner (Aero + Lateral)", 960.0, 1550.0),
    ("4_t4_t5_trail_braking", "4. T4–T5 Trail Braking & Turn-In (Combined)", 1550.0, 2350.0),
    ("5_t10_t16_sector3", "5. T10–T16 Continuous Complex to Finish (Braking + T11 Curb + Chicane)", 3150.0, 4586.0),
)


@dataclass
class AeroDragVerificationResult:
    """Straight-line aerodynamic drag speed-curve verification (wing + brake ducts)."""

    dist_m: np.ndarray
    v_obs_kmh: np.ndarray
    v_pred_kmh: np.ndarray
    v_no_ducts_kmh: np.ndarray
    rmse_kmh: float
    end_v_obs_kmh: float
    end_v_pred_kmh: float
    end_v_no_ducts_kmh: float
    dcd_wing: float
    dcd_ducts: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse_kmh": round(self.rmse_kmh, 3),
            "end_v_obs_kmh": round(self.end_v_obs_kmh, 2),
            "end_v_pred_kmh": round(self.end_v_pred_kmh, 2),
            "end_v_no_ducts_kmh": round(self.end_v_no_ducts_kmh, 2),
            "dcd_wing": round(self.dcd_wing, 4),
            "dcd_ducts": round(self.dcd_ducts, 4),
        }


@dataclass
class SuspensionVerificationState:
    """Full-lap physical suspension, progressive bumpstop force, and normal load predictions."""

    fz_corners_n: dict[str, np.ndarray]
    fz_susp_total_n: np.ndarray
    fz_aero_static_total_n: np.ndarray
    z_pred_mm: dict[str, np.ndarray]
    z_obs_mm: dict[str, np.ndarray]
    bs_force_pred_n: dict[str, np.ndarray]
    bs_force_obs_n: dict[str, np.ndarray]
    pitch_deg_pred: np.ndarray
    pitch_deg_obs: np.ndarray


@dataclass
class ScenarioMetrics:
    """Verification summary metrics for one continuous track segment."""

    key: str
    label: str
    start_m: float
    end_m: float
    n_samples: int
    speed_range_kmh: tuple[float, float]
    mean_corner_rmse_mm: float
    corner_rmse_mm: dict[str, float]
    max_bs_force_obs_n: float
    max_bs_force_pred_n: float
    bs_force_rmse_n: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _r2_rmse(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    diff = obs - pred
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1.0 - float(np.sum(diff**2)) / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean(diff**2)))
    return r2, rmse


def _solve_progressive_bumpstop(
    z_free_mm: np.ndarray,
    bs_thr_mm: float,
    kw_n_mm: float,
    k_prog_coef: float = 1.65,
    p_exp: float = 2.73,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve progressive polymer bumpstop equilibrium ``kw*(z - z_free) + k_prog*(z - bs_thr)^p = 0``."""
    z_eff = z_free_mm.copy()
    bs_force = np.zeros_like(z_free_mm)
    over = z_free_mm > bs_thr_mm
    if not np.any(over):
        return z_eff, bs_force

    dz = np.maximum(0.05, (z_free_mm[over] - bs_thr_mm) * 0.5)
    for _ in range(15):
        f = kw_n_mm * (bs_thr_mm + dz - z_free_mm[over]) + k_prog_coef * (dz**p_exp)
        df_dz = kw_n_mm + k_prog_coef * p_exp * (dz ** (p_exp - 1.0))
        dz = np.maximum(1e-4, dz - f / df_dz)

    z_eff[over] = bs_thr_mm + dz
    bs_force[over] = k_prog_coef * (dz**p_exp)
    return z_eff, bs_force


def compute_corner_normal_loads(
    df: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    fuel_liters: float = 2.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute 4-corner normal loads ``F_z(t)`` [N] and cross-check against suspension telemetry."""
    phys = resolve_setup_physics(car, setup, fuel_liters=fuel_liters)
    m = phys["mass_kg"]
    v = df["SPEED"].to_numpy(dtype=float)
    ay = df["G_LAT"].to_numpy(dtype=float)
    ax_col = "AX_SPEED" if "AX_SPEED" in df.columns else "G_LON"
    ax = df[ax_col].to_numpy(dtype=float)

    q = 0.5 * AIR_DENSITY * phys["ref_area_m2"] * (v**2)
    fz_aero_f = 0.5 * q * phys["cl_f"]
    fz_aero_r = 0.5 * q * phys["cl_r"]

    fz0_f = 0.5 * car.front_weight_dist * m * GRAVITY
    fz0_r = 0.5 * (1.0 - car.front_weight_dist) * m * GRAVITY

    dfz_long = 0.5 * m * ax * (car.cg_height_m / max(car.wheelbase_m, 1e-3))
    roll_f = phys["roll_front_share"]
    dfz_lat_f = roll_f * m * ay * (car.cg_height_m / max(car.track_width_front_m, 1e-3))
    dfz_lat_r = (1.0 - roll_f) * m * ay * (car.cg_height_m / max(car.track_width_rear_m, 1e-3))

    fz_corners = {
        "LF": np.clip(fz0_f - dfz_long - dfz_lat_f + fz_aero_f, 100.0, None),
        "RF": np.clip(fz0_f - dfz_long + dfz_lat_f + fz_aero_f, 100.0, None),
        "LR": np.clip(fz0_r + dfz_long - dfz_lat_r + fz_aero_r, 100.0, None),
        "RR": np.clip(fz0_r + dfz_long + dfz_lat_r + fz_aero_r, 100.0, None),
    }
    fz_aero_static_total = fz_corners["LF"] + fz_corners["RF"] + fz_corners["LR"] + fz_corners["RR"]

    if all(f"SUS_TRAVEL_{c}" in df.columns for c in CORNERS):
        z_lf = df["SUS_TRAVEL_LF"].to_numpy(dtype=float) * 1e-3
        z_rf = df["SUS_TRAVEL_RF"].to_numpy(dtype=float) * 1e-3
        z_lr = df["SUS_TRAVEL_LR"].to_numpy(dtype=float) * 1e-3
        z_rr = df["SUS_TRAVEL_RR"].to_numpy(dtype=float) * 1e-3
        z0_f, z0_r = 0.015, 0.017
        fz_susp = (
            m * GRAVITY
            + phys["kw_f"] * ((z_lf - z0_f) + (z_rf - z0_f))
            + phys["kw_r"] * ((z_lr - z0_r) + (z_rr - z0_r))
        )
        for c in CORNERS:
            bs_col = f"BUMPSTOP_FORCE_{c}"
            if bs_col in df.columns:
                # ACC MoTeC exports compression bumpstop force as negative values
                fz_susp = fz_susp + np.maximum(0.0, -df[bs_col].to_numpy(dtype=float))
        fz_susp_total = fz_susp
    else:
        fz_susp_total = fz_aero_static_total.copy()

    return fz_corners, fz_susp_total, fz_aero_static_total


def verify_straight_aero_drag(
    df_curated: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    calib_dir: str | Path,
    fuel_liters: float = 2.0,
) -> AeroDragVerificationResult:
    """Verify calibrated rear-wing and brake-duct drag against the main-straight speed curve."""
    cdir = Path(calib_dir)
    t1_df = _load_calib_csv(cdir / "test1.csv")
    bins = np.linspace(58.0, 75.0, 18)
    v1, a1 = _wot_ax_curve(t1_df, bins)
    m_t1 = float(car.mass_kg + 10.0 * 0.75)
    m_tr = float(car.mass_kg + fuel_liters * 0.75)
    f_net_t1 = m_t1 * a1  # F_drive(v) - F_drag(wing=0, ducts=[2,3])

    aero_slot = car.calibration_slots.get("aero_clicks", {})
    aero_cfg = setup.get("aero_clicks", {})
    wing_c = float(aero_cfg.get("rear_wing", 5))
    ducts = aero_cfg.get("brake_ducts", [2, 3])
    dcd_wing = wing_c * float(aero_slot.get("rear_wing_dcd_per_click", 0.0073))
    dcd_ducts = float(ducts[1]) * float(aero_slot.get("front_brake_duct_dcd_per_click", 0.0030)) + float(
        ducts[0]
    ) * float(aero_slot.get("rear_brake_duct_dcd_per_click", 0.0017))

    ref_a = float(car.aero_baseline.reference_area_m2 or 2.08)
    m_str = (df_curated["LAP_DIST_M"] <= 700.0) & (df_curated["THROTTLE"] == 100) & (df_curated["BRAKE"] == 0)
    t_str = df_curated.loc[m_str, "Time"].to_numpy(dtype=float)
    d_str = df_curated.loc[m_str, "LAP_DIST_M"].to_numpy(dtype=float)
    v_str = df_curated.loc[m_str, "SPEED"].to_numpy(dtype=float)

    def _ode_speed(_t: float, y: list[float], extra_dcd: float) -> list[float]:
        v_cur = float(y[0])
        f_t1 = float(np.interp(v_cur, v1, f_net_t1))
        f_extra = 0.5 * AIR_DENSITY * ref_a * extra_dcd * (v_cur**2)
        return [(f_t1 - f_extra) / m_tr]

    sol_cal = solve_ivp(
        lambda t, y: _ode_speed(t, y, dcd_wing), [float(t_str[0]), float(t_str[-1])], [float(v_str[0])], t_eval=t_str, max_step=0.05
    )
    sol_noducts = solve_ivp(
        lambda t, y: _ode_speed(t, y, -dcd_ducts), [float(t_str[0]), float(t_str[-1])], [float(v_str[0])], t_eval=t_str, max_step=0.05
    )

    v_obs_kmh = v_str * 3.6
    v_pred_kmh = sol_cal.y[0] * 3.6
    v_noducts_kmh = sol_noducts.y[0] * 3.6
    _, rmse_kmh = _r2_rmse(v_obs_kmh, v_pred_kmh)

    return AeroDragVerificationResult(
        dist_m=d_str,
        v_obs_kmh=v_obs_kmh,
        v_pred_kmh=v_pred_kmh,
        v_no_ducts_kmh=v_noducts_kmh,
        rmse_kmh=rmse_kmh,
        end_v_obs_kmh=float(v_obs_kmh[-1]),
        end_v_pred_kmh=float(v_pred_kmh[-1]),
        end_v_no_ducts_kmh=float(v_noducts_kmh[-1]),
        dcd_wing=dcd_wing,
        dcd_ducts=dcd_ducts,
    )


def predict_suspension_state(
    df: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    fuel_liters: float = 2.0,
) -> SuspensionVerificationState:
    """Predict 4-corner suspension travel & progressive bumpstop force without fitting."""
    phys = resolve_setup_physics(car, setup, fuel_liters=fuel_liters)
    m = phys["mass_kg"]
    fw = car.front_weight_dist
    h_cg = car.cg_height_m
    v = df["SPEED"].to_numpy(dtype=float)
    ay = df["G_LAT"].to_numpy(dtype=float)
    ax_col = "AX_SPEED" if "AX_SPEED" in df.columns else "G_LON"
    ax = df[ax_col].to_numpy(dtype=float)

    fz_corners, fz_susp_total, fz_aero_static_total = compute_corner_normal_loads(
        df, car, setup, fuel_liters=fuel_liters
    )

    q = 0.5 * AIR_DENSITY * phys["ref_area_m2"] * (v**2)
    z0_f_mm, z0_r_mm = 15.0, 17.0
    dfz_pitch = 0.5 * m * ax * (h_cg / max(car.wheelbase_m, 1e-3))
    zf_mean_free = z0_f_mm + (0.5 * q * phys["cl_f"] - dfz_pitch) / (phys["kw_f"] * 1e-3)
    zr_mean_free = z0_r_mm + (0.5 * q * phys["cl_r"] + dfz_pitch) / (phys["kw_r"] * 1e-3)

    s_f, s_r = 1.0, 1.0
    if all(f"SUS_TRAVEL_{c}" in df.columns for c in CORNERS):
        dz_f_obs = (df["SUS_TRAVEL_LF"] - df["SUS_TRAVEL_RF"]).to_numpy(dtype=float)
        dz_r_obs = (df["SUS_TRAVEL_LR"] - df["SUS_TRAVEL_RR"]).to_numpy(dtype=float)
        if np.std(dz_f_obs) > 1e-9 and np.std(ay) > 1e-9:
            s_f = 1.0 if float(np.corrcoef(dz_f_obs, ay)[0, 1]) >= 0.0 else -1.0
        if np.std(dz_r_obs) > 1e-9 and np.std(ay) > 1e-9:
            s_r = 1.0 if float(np.corrcoef(dz_r_obs, ay)[0, 1]) >= 0.0 else -1.0

    dz_f = s_f * (2.0 * (m * fw) * h_cg / (max(car.track_width_front_m, 1e-3) * phys["k_eff_f"])) * ay * 1e3
    dz_r = s_r * (2.0 * (m * (1.0 - fw)) * h_cg / (max(car.track_width_rear_m, 1e-3) * phys["k_eff_r"])) * ay * 1e3

    z_free_mm = {
        "LF": zf_mean_free + 0.5 * dz_f,
        "RF": zf_mean_free - 0.5 * dz_f,
        "LR": zr_mean_free + 0.5 * dz_r,
        "RR": zr_mean_free - 0.5 * dz_r,
    }

    bs_rates_n_mm = setup.get("suspension", {}).get("bumpstop_rate_n_mm", [1000.0, 1000.0, 300.0, 300.0])
    z_pred_mm: dict[str, np.ndarray] = {}
    bs_force_pred_n: dict[str, np.ndarray] = {}
    z_obs_mm: dict[str, np.ndarray] = {}
    bs_force_obs_n: dict[str, np.ndarray] = {}

    for idx, c in enumerate(CORNERS):
        is_f = c in ("LF", "RF")
        kw_n_mm = (phys["kw_f"] if is_f else phys["kw_r"]) * 1e-3
        bs_thr = phys["bs_thresh_f_mm"] if is_f else phys["bs_thresh_r_mm"]
        # Scale progressive polymer stiffness relative to front 1000 N/mm reference (1.65 N/mm^2.73)
        k_prog = 1.65 * (float(bs_rates_n_mm[idx]) / 1000.0)
        z_eff, f_bs = _solve_progressive_bumpstop(z_free_mm[c], bs_thr, kw_n_mm, k_prog_coef=k_prog, p_exp=2.73)
        z_pred_mm[c] = z_eff
        bs_force_pred_n[c] = f_bs

        z_obs_mm[c] = df[f"SUS_TRAVEL_{c}"].to_numpy(dtype=float) if f"SUS_TRAVEL_{c}" in df.columns else z_eff.copy()
        # ACC MoTeC exports compression bumpstop force as negative values; take compression magnitude [N]
        bs_force_obs_n[c] = (
            np.maximum(0.0, -df[f"BUMPSTOP_FORCE_{c}"].to_numpy(dtype=float))
            if f"BUMPSTOP_FORCE_{c}" in df.columns
            else np.zeros(len(df), dtype=float)
        )

    zf_p = 0.5 * (z_pred_mm["LF"] + z_pred_mm["RF"])
    zr_p = 0.5 * (z_pred_mm["LR"] + z_pred_mm["RR"])
    zf_o = 0.5 * (z_obs_mm["LF"] + z_obs_mm["RF"])
    zr_o = 0.5 * (z_obs_mm["LR"] + z_obs_mm["RR"])

    pitch_deg_pred = np.rad2deg(np.arctan(((zf_p - z0_f_mm) - (zr_p - z0_r_mm)) * 1e-3 / car.wheelbase_m))
    pitch_deg_obs = np.rad2deg(np.arctan(((zf_o - z0_f_mm) - (zr_o - z0_r_mm)) * 1e-3 / car.wheelbase_m))

    return SuspensionVerificationState(
        fz_corners_n=fz_corners,
        fz_susp_total_n=fz_susp_total,
        fz_aero_static_total_n=fz_aero_static_total,
        z_pred_mm=z_pred_mm,
        z_obs_mm=z_obs_mm,
        bs_force_pred_n=bs_force_pred_n,
        bs_force_obs_n=bs_force_obs_n,
        pitch_deg_pred=pitch_deg_pred,
        pitch_deg_obs=pitch_deg_obs,
    )


def verify_stage3_scenarios(
    df_curated: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    fuel_liters: float = 2.0,
) -> tuple[SuspensionVerificationState, list[ScenarioMetrics]]:
    """Run all 5 continuous Barcelona suspension & bumpstop verification segments."""
    state = predict_suspension_state(df_curated, car, setup, fuel_liters=fuel_liters)
    dist = df_curated["LAP_DIST_M"].to_numpy(dtype=float)
    v_kmh = df_curated["SPEED"].to_numpy(dtype=float) * 3.6

    bs_obs_tot = sum(state.bs_force_obs_n[c] for c in CORNERS)
    bs_pred_tot = sum(state.bs_force_pred_n[c] for c in CORNERS)

    scenarios: list[ScenarioMetrics] = []
    for key, label, s_m, e_m in SCENARIO_WINDOWS_M:
        m_seg = (dist >= s_m) & (dist <= e_m)
        if not np.any(m_seg):
            continue
        corner_rmse: dict[str, float] = {}
        for c in CORNERS:
            _, rmse_c = _r2_rmse(state.z_obs_mm[c][m_seg], state.z_pred_mm[c][m_seg])
            corner_rmse[c] = round(rmse_c, 3)

        _, bs_rmse = _r2_rmse(bs_obs_tot[m_seg], bs_pred_tot[m_seg])
        scenarios.append(
            ScenarioMetrics(
                key=key,
                label=label,
                start_m=s_m,
                end_m=e_m,
                n_samples=int(np.sum(m_seg)),
                speed_range_kmh=(round(float(v_kmh[m_seg].min()), 1), round(float(v_kmh[m_seg].max()), 1)),
                mean_corner_rmse_mm=round(float(np.mean(list(corner_rmse.values()))), 3),
                corner_rmse_mm=corner_rmse,
                max_bs_force_obs_n=round(float(bs_obs_tot[m_seg].max()), 1),
                max_bs_force_pred_n=round(float(bs_pred_tot[m_seg].max()), 1),
                bs_force_rmse_n=round(bs_rmse, 1),
            )
        )

    return state, scenarios
