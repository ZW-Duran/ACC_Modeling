"""Black-box click-to-SI parameter identification for ACC MoTeC telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

from src.data.motec_loader import CORE_CHANNELS, load_motec_csv

CORNERS: tuple[str, ...] = ("LF", "RF", "LR", "RR")
FRONT_CORNERS: tuple[str, str] = ("LF", "RF")
REAR_CORNERS: tuple[str, str] = ("LR", "RR")
GRAVITY = 9.80665
AIR_DENSITY = 1.225

CALIBRATION_CHANNELS: tuple[str, ...] = (
    *CORE_CHANNELS,
    *(
        f"{prefix}_{corner}"
        for prefix in ("SUS_TRAVEL", "BUMPSTOP_FORCE", "BUMPSTOPUP_RIDE", "BUMPSTOPDN_RIDE")
        for corner in CORNERS
    ),
    "RPMS",
    "GEAR",
    "TC",
    "ABS",
)


def _load_calib_csv(filepath: Path) -> pd.DataFrame:
    df, _ = load_motec_csv(filepath, channels=CALIBRATION_CHANNELS)
    return df


def _smooth_roll_gradients(df: pd.DataFrame) -> tuple[float, float]:
    st_lf = savgol_filter(df["SUS_TRAVEL_LF"].to_numpy(dtype=float), 41, 3)
    st_rf = savgol_filter(df["SUS_TRAVEL_RF"].to_numpy(dtype=float), 41, 3)
    st_lr = savgol_filter(df["SUS_TRAVEL_LR"].to_numpy(dtype=float), 41, 3)
    st_rr = savgol_filter(df["SUS_TRAVEL_RR"].to_numpy(dtype=float), 41, 3)
    vs_lf = savgol_filter(df["SUS_TRAVEL_LF"].to_numpy(dtype=float), 41, 3, deriv=1, delta=0.005)
    vs_rf = savgol_filter(df["SUS_TRAVEL_RF"].to_numpy(dtype=float), 41, 3, deriv=1, delta=0.005)
    ay = savgol_filter(df["G_LAT"].to_numpy(dtype=float) / 9.80665, 41, 3)

    no_bs = np.ones(len(df), dtype=bool)
    for c in CORNERS:
        no_bs &= (df[f"BUMPSTOPUP_RIDE_{c}"].to_numpy() == 0) & (df[f"BUMPSTOPDN_RIDE_{c}"].to_numpy() == 0)

    mask = (
        no_bs
        & (np.abs(ay) > 0.6)
        & (np.abs(ay) < 1.6)
        & (np.abs(vs_lf - vs_rf) < 8.0)
        & (df["SPEED"].to_numpy(dtype=float) > 25.0)
    )
    if mask.sum() < 20:
        raise ValueError(f"Insufficient steady cornering samples ({mask.sum()}) for roll gradient fit.")
    slope_f, _ = np.polyfit(ay[mask], (st_lf - st_rf)[mask], 1)
    slope_r, _ = np.polyfit(ay[mask], (st_lr - st_rr)[mask], 1)
    return float(abs(slope_f)), float(abs(slope_r))


def _bumpstop_threshold_mm(df: pd.DataFrame, corners: tuple[str, str]) -> float:
    vals: list[float] = []
    for c in corners:
        bu = df[f"BUMPSTOPUP_RIDE_{c}"].to_numpy()
        st = df[f"SUS_TRAVEL_{c}"].to_numpy(dtype=float)
        engaged = st[bu == 1]
        if len(engaged) == 0:
            raise ValueError(f"No BUMPSTOPUP_RIDE_{c} == 1 samples found to identify bumpstop range.")
        vals.append(float(np.min(engaged)))
    return float(np.median(vals))


def verify_setup_consistency(calib_dir: str | Path) -> dict[str, Any]:
    """Audit tire pressures, bumpstop engagement thresholds, roll gradients, and Vmax across CSVs."""
    cdir = Path(calib_dir)
    names = ("baseline", "test1", "test2", "test3", "test4", "test5", "test6", "test7", "test9", "test9_out")
    sessions: dict[str, dict[str, Any]] = {}

    for name in names:
        csv_path = cdir / f"{name}.csv"
        if not csv_path.exists():
            continue
        df = _load_calib_csv(csv_path)
        thresholds = {
            c: float(df.loc[df[f"BUMPSTOPUP_RIDE_{c}"] == 1, f"SUS_TRAVEL_{c}"].min())
            for c in CORNERS
            if (df[f"BUMPSTOPUP_RIDE_{c}"] == 1).any()
        }
        try:
            roll_f, roll_r = _smooth_roll_gradients(df)
            roll_f, roll_r = round(roll_f, 3), round(roll_r, 3)
        except ValueError:
            roll_f, roll_r = float("nan"), float("nan")

        sessions[name] = {
            "n_rows": int(len(df)),
            "vmax_kmh": round(float(df["SPEED"].max() * 3.6), 2),
            "mean_pressures_psi": [round(float(df[f"TYRE_PRESS_{c}"].mean()), 2) for c in CORNERS],
            "bumpstop_thresholds_mm": thresholds,
            "roll_grad_mm_per_g": {"front": roll_f, "rear": roll_r},
        }

    test2_matches = False
    if "baseline" in sessions and "test2" in sessions:
        b, t2 = sessions["baseline"], sessions["test2"]
        tp_diff = max(abs(x - y) for x, y in zip(b["mean_pressures_psi"], t2["mean_pressures_psi"]))
        bs_same = b["bumpstop_thresholds_mm"] == t2["bumpstop_thresholds_mm"]
        roll_diff = abs(b["roll_grad_mm_per_g"]["front"] - t2["roll_grad_mm_per_g"]["front"])
        test2_matches = bool(tp_diff <= 0.15 and bs_same and roll_diff <= 0.8)

    return {"sessions": sessions, "test2_matches_baseline_chassis": test2_matches}


def calibrate_bumpstop_range(
    baseline_df: pd.DataFrame,
    test5_df: pd.DataFrame,
    base_clicks: tuple[int, int] = (5, 14),
    test5_clicks: tuple[int, int] = (2, 10),
) -> dict[str, float]:
    """Solve linear bumpstop engagement range [mm] vs. UI clicks from ``baseline`` and ``test5``.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        200 Hz telemetry DataFrame at baseline bumpstop clicks (e.g. ``[5, 5, 14, 14]``).
    test5_df : pd.DataFrame
        200 Hz telemetry DataFrame at reduced bumpstop clicks (e.g. ``[2, 2, 10, 10]``).
    base_clicks, test5_clicks : tuple[int, int], optional
        ``(front_click, rear_click)`` UI values for each session.

    Returns
    -------
    dict[str, float]
        ``front_base_mm``, ``front_per_click_mm``, ``rear_base_mm``, ``rear_per_click_mm``
        using the ``click = 0`` intercept convention. Mass-independent (scales as ``m^0``).
    """
    f_base_thresh = _bumpstop_threshold_mm(baseline_df, FRONT_CORNERS)
    r_base_thresh = _bumpstop_threshold_mm(baseline_df, REAR_CORNERS)
    f_t5_thresh = _bumpstop_threshold_mm(test5_df, FRONT_CORNERS)
    r_t5_thresh = _bumpstop_threshold_mm(test5_df, REAR_CORNERS)

    f_per_click = (f_base_thresh - f_t5_thresh) / float(base_clicks[0] - test5_clicks[0])
    r_per_click = (r_base_thresh - r_t5_thresh) / float(base_clicks[1] - test5_clicks[1])

    return {
        "front_base_mm": round(f_base_thresh - base_clicks[0] * f_per_click, 3),
        "front_per_click_mm": round(f_per_click, 3),
        "rear_base_mm": round(r_base_thresh - base_clicks[1] * r_per_click, 3),
        "rear_per_click_mm": round(r_per_click, 3),
    }


def _estimate_aero_downforce(
    df: pd.DataFrame,
    mass_kg: float,
    ref_area_m2: float,
    wheelbase_m: float,
    cg_height_m: float,
    k_f_axle_n_mm: float,
    k_r_axle_n_mm: float,
    z0_f_mm: float,
    z0_r_mm: float,
) -> tuple[float, float, float, float]:
    v = df["SPEED"].to_numpy(dtype=float)
    ax = savgol_filter(v, 51, 3, deriv=1, delta=0.005)
    ay_g = df["G_LAT"].to_numpy(dtype=float) / 9.80665
    f_tr = 0.5 * (df["SUS_TRAVEL_LF"] + df["SUS_TRAVEL_RF"]).to_numpy(dtype=float)
    r_tr = 0.5 * (df["SUS_TRAVEL_LR"] + df["SUS_TRAVEL_RR"]).to_numpy(dtype=float)

    dfz_pitch = mass_kg * ax * (cg_height_m / wheelbase_m)
    fz_aero_f = (f_tr - z0_f_mm) * k_f_axle_n_mm + dfz_pitch
    fz_aero_r = (r_tr - z0_r_mm) * k_r_axle_n_mm - dfz_pitch

    no_bs = np.ones(len(df), dtype=bool)
    for c in CORNERS:
        no_bs &= df[f"BUMPSTOPUP_RIDE_{c}"].to_numpy() == 0

    mask = (v > 50.0) & (np.abs(ay_g) < 0.15) & (df["THROTTLE"].to_numpy() == 100) & no_bs
    q = 0.5 * AIR_DENSITY * ref_area_m2 * (v[mask] ** 2)
    q_sq = float(np.sum(q**2))
    cl_f = float(np.sum(q * fz_aero_f[mask]) / q_sq)
    cl_r = float(np.sum(q * fz_aero_r[mask]) / q_sq)
    cl_tot = cl_f + cl_r
    return cl_f, cl_r, cl_tot, cl_f / cl_tot


def _wot_ax_curve(df: pd.DataFrame, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = df["SPEED"].to_numpy(dtype=float)
    ax = savgol_filter(v, 81, 3, deriv=1, delta=0.005)
    ay_g = df["G_LAT"].to_numpy(dtype=float) / 9.80665
    wot = (
        (df["THROTTLE"].to_numpy() == 100)
        & (df["BRAKE"].to_numpy() == 0)
        & (df["TC"].to_numpy() == 0)
        & (np.abs(ay_g) < 0.12)
        & (df["GEAR"].to_numpy() >= 5)
    )
    vb, axb = [], []
    for i in range(len(bins) - 1):
        bm = wot & (v >= bins[i]) & (v < bins[i + 1])
        if bm.sum() >= 10:
            vb.append(0.5 * (bins[i] + bins[i + 1]))
            axb.append(float(np.median(ax[bm])))
    return np.asarray(vb), np.asarray(axb)


def _diff_cd(
    curve_a: tuple[np.ndarray, np.ndarray],
    curve_b: tuple[np.ndarray, np.ndarray],
    mass_kg: float,
    ref_area_m2: float,
) -> float:
    v1, a1 = curve_a
    v2, a2 = curve_b
    common = np.intersect1d(v1, v2)
    idx1 = np.isin(v1, common)
    idx2 = np.isin(v2, common)
    q = 0.5 * AIR_DENSITY * ref_area_m2 * (common**2)
    df_drag = -mass_kg * (a1[idx1] - a2[idx2])
    return float(np.sum(q * df_drag) / np.sum(q**2))


def calibrate_aero(
    calib_dir: str | Path,
    mass_kg: float = 1285.0,
    ref_area_m2: float = 2.08,
    wheelbase_m: float = 2.70,
    cg_height_m: float = 0.42,
    k_f_axle_n_mm: float = 306.0,
    k_r_axle_n_mm: float = 326.0,
    z0_f_mm: float = 15.0,
    z0_r_mm: float = 17.0,
) -> dict[str, Any]:
    """Identify aero_baseline (Cl, front dist) and rear-wing / brake-duct click sensitivities."""
    cdir = Path(calib_dir)
    specs = [
        ("test1", 0),
        ("test2", 3),
        ("baseline", 5),
        ("test3", 5),
        ("test4", 5),
    ]
    dfs = {name: _load_calib_csv(cdir / f"{name}.csv") for name, _ in specs}

    cl_summary: dict[str, dict[str, float]] = {}
    for name, _ in specs:
        cl_f, cl_r, cl_tot, f_dist = _estimate_aero_downforce(
            dfs[name],
            mass_kg=mass_kg,
            ref_area_m2=ref_area_m2,
            wheelbase_m=wheelbase_m,
            cg_height_m=cg_height_m,
            k_f_axle_n_mm=k_f_axle_n_mm,
            k_r_axle_n_mm=k_r_axle_n_mm,
            z0_f_mm=z0_f_mm,
            z0_r_mm=z0_r_mm,
        )
        cl_summary[name] = {
            "cl_front": round(cl_f, 4),
            "cl_rear": round(cl_r, 4),
            "cl_total": round(cl_tot, 4),
            "aero_front_dist": round(f_dist, 4),
        }

    wings = np.array([0.0, 3.0, 5.0])
    cl_vals = np.array([cl_summary[n]["cl_total"] for n in ("test1", "test2", "baseline")])
    cl_f_vals = np.array([cl_summary[n]["cl_front"] for n in ("test1", "test2", "baseline")])
    cl_r_vals = np.array([cl_summary[n]["cl_rear"] for n in ("test1", "test2", "baseline")])

    wing_dcl, _ = np.polyfit(wings, cl_vals, 1)
    wing_dcl_f, _ = np.polyfit(wings, cl_f_vals, 1)
    wing_dcl_r, _ = np.polyfit(wings, cl_r_vals, 1)

    bins = np.linspace(58.0, 74.0, 17)
    curves = {n: _wot_ax_curve(dfs[n], bins) for n in ("test1", "test2", "baseline", "test3", "test4")}
    dcd_w3_w0 = _diff_cd(curves["test2"], curves["test1"], mass_kg, ref_area_m2)
    dcd_w5_w0 = _diff_cd(curves["baseline"], curves["test1"], mass_kg, ref_area_m2)
    wing_dcd, _ = np.polyfit(wings, np.array([0.0, dcd_w3_w0, dcd_w5_w0]), 1)

    dcd_rear_duct = _diff_cd(curves["test3"], curves["baseline"], mass_kg, ref_area_m2)
    dcd_front_duct = _diff_cd(curves["test4"], curves["baseline"], mass_kg, ref_area_m2)

    return {
        "aero_baseline": {
            "base_cl": round(cl_summary["test1"]["cl_total"], 3),
            "base_aero_front_dist": round(cl_summary["test1"]["aero_front_dist"], 3),
        },
        "aero_clicks": {
            "rear_wing_dcl_per_click": round(float(wing_dcl), 4),
            "rear_wing_dcd_per_click": round(float(wing_dcd), 4),
            "front_splitter_dcl_per_click": 0.0,
            "front_splitter_df_dist_per_click": 0.0,
            "front_brake_duct_dcd_per_click": round(float(dcd_front_duct / 3.0), 4),
            "rear_brake_duct_dcd_per_click": round(float(dcd_rear_duct / 4.0), 4),
        },
        "diagnostics": {
            "by_session": cl_summary,
            "rear_wing_dcl_front_per_click": round(float(wing_dcl_f), 4),
            "rear_wing_dcl_rear_per_click": round(float(wing_dcl_r), 4),
            "rear_duct_dcd_per_click": round(float(dcd_rear_duct / 4.0), 5),
            "front_duct_dcd_per_click": round(float(dcd_front_duct / 3.0), 5),
        },
    }


def calibrate_arb_stiffness(
    calib_dir: str | Path,
    mass_kg: float = 1285.0,
    front_weight_dist: float = 0.43,
    cg_height_m: float = 0.42,
    track_f_m: float = 1.67,
    track_r_m: float = 1.65,
    kw_f_n_m: float = 153000.0,
    kw_r_n_m: float = 163000.0,
) -> dict[str, float]:
    """Solve differential ARB stiffness [N/m at wheel] from steady-state cornering roll gradients.

    Parameters
    ----------
    calib_dir : str | Path
        Directory containing 200 Hz calibration CSVs (``baseline``, ``test2``, ``test3``,
        ``test4``, ``test6`` ``[4,0]``, ``test7`` ``[0,2]``).
    mass_kg : float
        Total vehicle test mass [kg] (including 10L Hotlap fuel).
    front_weight_dist : float, default=0.43
        Static front axle mass fraction [-].
    cg_height_m : float, default=0.42
        Center-of-gravity height [m].
    track_f_m, track_r_m : float
        Front and rear track widths [m].
    kw_f_n_m, kw_r_n_m : float
        Front and rear spring wheel rates [N/m].

    Returns
    -------
    dict[str, float]
        ``front_base``, ``front_per_click``, ``rear_base``, ``rear_per_click`` [N/m at wheel],
        referenced to ``click = 0`` in the ACC setup UI. Scales linearly with ``mass_kg``.

    Notes
    -----
    Assumes **200 Hz** (``dt = 0.005 s``) telemetry for Savitzky-Golay roll-velocity filtering.
    """
    cdir = Path(calib_dir)
    dfs = {n: _load_calib_csv(cdir / f"{n}.csv") for n in ("baseline", "test2", "test3", "test4", "test6", "test7")}

    grads = {n: _smooth_roll_gradients(df) for n, df in dfs.items()}
    dz_f_42 = float(np.mean([grads[n][0] for n in ("baseline", "test2", "test3", "test4")]))
    dz_r_42 = float(np.mean([grads[n][1] for n in ("baseline", "test2", "test3", "test4")]))
    dz_r_40 = grads["test6"][1]
    dz_f_02 = grads["test7"][0]

    mf = mass_kg * front_weight_dist
    mr = mass_kg * (1.0 - front_weight_dist)

    # Fz_L - Fz_R = 2 * m_i * g * h_cg / t_i per g = K_eff * |dz / ay|
    k_eff_f0 = (2.0 * mf * GRAVITY * cg_height_m) / (track_f_m * (dz_f_02 * 1e-3))
    k_eff_f4 = (2.0 * mf * GRAVITY * cg_height_m) / (track_f_m * (dz_f_42 * 1e-3))
    k_eff_r0 = (2.0 * mr * GRAVITY * cg_height_m) / (track_r_m * (dz_r_40 * 1e-3))
    k_eff_r2 = (2.0 * mr * GRAVITY * cg_height_m) / (track_r_m * (dz_r_42 * 1e-3))

    front_base = k_eff_f0 - kw_f_n_m
    front_per_click = (k_eff_f4 - k_eff_f0) / 4.0
    rear_base = k_eff_r0 - kw_r_n_m
    rear_per_click = (k_eff_r2 - k_eff_r0) / 2.0

    return {
        "front_base": round(float(front_base), 1),
        "front_per_click": round(float(front_per_click), 1),
        "rear_base": round(float(rear_base), 1),
        "rear_per_click": round(float(rear_per_click), 1),
    }


def _collect_ringdown_damping(
    dfs: list[pd.DataFrame],
    kw_f: float,
    kw_r: float,
    ms_f: float,
    ms_r: float,
    knee_velocity_mps: float,
) -> dict[str, list[float]]:
    b_hp, a_hp = butter(2, 1.2 / 100.0, btype="high")
    regimes: dict[str, list[float]] = {"SB": [], "FB": [], "SR": [], "FR": []}
    corner_specs = (
        ("LF", kw_f, ms_f),
        ("RF", kw_f, ms_f),
        ("LR", kw_r, ms_r),
        ("RR", kw_r, ms_r),
    )

    for df in dfs:
        moving = (df["SPEED"] > 15.0).to_numpy()
        for corner, kw, ms in corner_specs:
            z_raw = df[f"SUS_TRAVEL_{corner}"].to_numpy(dtype=float) * 1e-3
            z_osc = filtfilt(b_hp, a_hp, savgol_filter(z_raw, 15, 3))
            v_osc = np.gradient(z_osc, 0.005)
            peaks, _ = find_peaks(z_osc, distance=10, prominence=0.0008)
            troughs, _ = find_peaks(-z_osc, distance=10, prominence=0.0008)
            extrema = np.sort(np.concatenate([peaks, troughs]))
            c_crit = 2.0 * np.sqrt(kw * ms)

            for i in range(len(extrema) - 1):
                i1, i2 = int(extrema[i]), int(extrema[i + 1])
                if not (moving[i1] and moving[i2]) or not (8 <= (i2 - i1) <= 90):
                    continue
                z1, z2 = abs(z_osc[i1]), abs(z_osc[i2])
                if z2 >= z1 or z1 < 0.0015:
                    continue
                is_bump = z_osc[i2] > z_osc[i1]
                v_max = float(np.max(np.abs(v_osc[i1 : i2 + 1])))
                c_val = float((np.log(z1 / z2) / np.pi) * c_crit)
                if is_bump:
                    regimes["FB" if v_max > knee_velocity_mps else "SB"].append(c_val)
                else:
                    regimes["FR" if v_max > knee_velocity_mps else "SR"].append(c_val)

    return regimes


def calibrate_dampers(
    calib_dir: str | Path,
    kw_f_n_m: float = 153000.0,
    kw_r_n_m: float = 163000.0,
    mass_kg: float = 1285.0,
    front_weight_dist: float = 0.43,
    knee_velocity_mps: float = 0.05,
) -> dict[str, float]:
    """Identify slow/fast bump and rebound damping (N*s/m) via half-cycle ringdown."""
    cdir = Path(calib_dir)
    base_df = _load_calib_csv(cdir / "baseline.csv")
    t9_dfs = [_load_calib_csv(cdir / n) for n in ("test9.csv", "test9_out.csv") if (cdir / n).exists()]
    if not t9_dfs:
        raise FileNotFoundError("Missing test9.csv / test9_out.csv in calibration directory.")

    ms_f = 0.5 * mass_kg * front_weight_dist
    ms_r = 0.5 * mass_kg * (1.0 - front_weight_dist)

    r_base = _collect_ringdown_damping([base_df], kw_f_n_m, kw_r_n_m, ms_f, ms_r, knee_velocity_mps)
    r_t9 = _collect_ringdown_damping(t9_dfs, kw_f_n_m, kw_r_n_m, ms_f, ms_r, knee_velocity_mps)

    # Mean front/rear clicks in baseline (R8_A): SB=21, FB=19, SR=23.5, FR=27.5 vs test9=5
    click_pairs = {
        "SB": ("bump_slow", 21.0, 5.0),
        "FB": ("bump_fast", 19.0, 5.0),
        "SR": ("rebound_slow", 23.5, 5.0),
        "FR": ("rebound_fast", 27.5, 5.0),
    }

    out: dict[str, float] = {"knee_velocity_mps": knee_velocity_mps}
    for key, (prefix, c_hi, c_lo) in click_pairs.items():
        val_hi = float(np.median(r_base[key]))
        val_lo = float(np.median(r_t9[key]))
        per_click = (val_hi - val_lo) / (c_hi - c_lo)
        base = val_lo - c_lo * per_click
        out[f"{prefix}_base"] = round(base, 1)
        out[f"{prefix}_per_click"] = round(per_click, 2)

    return out


def _stationary_travel_datum_mm(calib_dir: Path) -> tuple[float, float]:
    """Extract stationary (v < 0.5 m/s) front/rear suspension zero travel from 10L test9_out.csv."""
    out_csv = calib_dir / "test9_out.csv"
    if not out_csv.exists():
        return 15.0, 17.0
    df = _load_calib_csv(out_csv)
    static = df[df["SPEED"] < 0.5]
    if static.empty:
        return 15.0, 17.0
    z0_f = float(np.median(0.5 * (static["SUS_TRAVEL_LF"] + static["SUS_TRAVEL_RF"])))
    z0_r = float(np.median(0.5 * (static["SUS_TRAVEL_LR"] + static["SUS_TRAVEL_RR"])))
    return z0_f, z0_r


def calibrate_mass(
    calib_dir: str | Path,
    low_fuel_liters: float = 2.0,
    high_fuel_liters: float = 110.0,
    fuel_density_kg_l: float = 0.75,
    k_f_axle_n_mm: float = 306.0,
    k_r_axle_n_mm: float = 326.0,
    wheelbase_m: float = 2.70,
    cg_height_m: float = 0.42,
) -> dict[str, Any]:
    """Identify dry chassis mass [kg] and fuel tank front/rear distribution from ``2L.csv`` vs. ``110L.csv``.

    Parameters
    ----------
    calib_dir : str | Path
        Directory containing 200 Hz ``2L.csv`` and ``110L.csv`` recorded under identical setup.
    low_fuel_liters, high_fuel_liters : float
        Fuel loads [L] in the low- and high-fuel sessions (default ``2.0`` and ``110.0``).
    fuel_density_kg_l : float, default=0.75
        Racing fuel density [kg/L] (``108 L -> 81.0 kg``).
    k_f_axle_n_mm, k_r_axle_n_mm : float
        Total front and rear axle wheel rates [N/mm].

    Returns
    -------
    dict[str, Any]
        Contains ``dry_mass_kg`` [kg], ``low_fuel_mass_kg`` [kg], ``fuel_delta_mass_kg`` [kg],
        ``fuel_front_weight_dist`` [-], and ``suspension_delta_mm`` [mm].

    Notes
    -----
    - Dry mass is identified via the pure 5th-gear WOT straight-line acceleration ratio
      (``60-67 m/s``): ``m_low = dm_fuel * a_high / (a_low - a_high)``.
    - Fuel tank front/rear share is identified from speed-matched (``50-72 m/s``)
      pitch-corrected straight-line suspension compression deltas.
    - Assumes **200 Hz** (``dt = 0.005 s``) telemetry.
    """
    cdir = Path(calib_dir)
    low_df = _load_calib_csv(cdir / f"{int(low_fuel_liters)}L.csv")
    high_df = _load_calib_csv(cdir / f"{int(high_fuel_liters)}L.csv")

    dm_fuel_kg = (high_fuel_liters - low_fuel_liters) * fuel_density_kg_l

    # 1. Pure 5th-gear WOT straight-line acceleration ratio: m_low * a_low = m_high * a_high
    ax_bins = np.linspace(60.0, 67.0, 8)
    v_low, a_low = _wot_ax_curve(low_df, ax_bins)
    v_high, a_high = _wot_ax_curve(high_df, ax_bins)
    common = np.intersect1d(v_low, v_high)
    idx_l = np.isin(v_low, common)
    idx_h = np.isin(v_high, common)
    diff = a_low[idx_l] - a_high[idx_h]
    valid_ax = np.abs(diff) > 1e-4
    if not np.any(valid_ax):
        raise ValueError("Insufficient WOT acceleration separation between low and high fuel runs.")
    m_low_bins = dm_fuel_kg * a_high[idx_h][valid_ax] / diff[valid_ax]
    m_low_kg = float(np.median(m_low_bins))
    dry_mass_kg = m_low_kg - low_fuel_liters * fuel_density_kg_l

    # 2. Speed-matched straight-line suspension compression for fuel cell F/R distribution
    z_bins = np.linspace(50.0, 72.0, 12)
    z_medians: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, df in (("low", low_df), ("high", high_df)):
        v = df["SPEED"].to_numpy(dtype=float)
        ax = savgol_filter(v, 51, 3, deriv=1, delta=0.005)
        ay_g = df["G_LAT"].to_numpy(dtype=float) / GRAVITY
        no_bs = np.ones(len(df), dtype=bool)
        for c in CORNERS:
            no_bs &= df[f"BUMPSTOPUP_RIDE_{c}"].to_numpy() == 0
        mask = (v > 50.0) & (np.abs(ay_g) < 0.15) & (df["THROTTLE"].to_numpy() == 100) & no_bs
        f_tr = 0.5 * (df["SUS_TRAVEL_LF"] + df["SUS_TRAVEL_RF"]).to_numpy(dtype=float)
        r_tr = 0.5 * (df["SUS_TRAVEL_LR"] + df["SUS_TRAVEL_RR"]).to_numpy(dtype=float)
        f_corr = f_tr + (m_low_kg * ax * cg_height_m / wheelbase_m) / k_f_axle_n_mm
        r_corr = r_tr - (m_low_kg * ax * cg_height_m / wheelbase_m) / k_r_axle_n_mm
        fb, rb = [], []
        for i in range(len(z_bins) - 1):
            bm = mask & (v >= z_bins[i]) & (v < z_bins[i + 1])
            if bm.sum() >= 10:
                fb.append(float(np.median(f_corr[bm])))
                rb.append(float(np.median(r_corr[bm])))
            else:
                fb.append(np.nan)
                rb.append(np.nan)
        z_medians[label] = (np.asarray(fb), np.asarray(rb))

    dz_f = z_medians["high"][0] - z_medians["low"][0]
    dz_r = z_medians["high"][1] - z_medians["low"][1]
    valid_z = ~np.isnan(dz_f) & ~np.isnan(dz_r)
    dz_f_mm = float(np.median(dz_f[valid_z]))
    dz_r_mm = float(np.median(dz_r[valid_z]))
    dm_f_kg = dz_f_mm * k_f_axle_n_mm / GRAVITY
    dm_r_kg = dz_r_mm * k_r_axle_n_mm / GRAVITY
    fuel_front_dist = float(dm_f_kg / (dm_f_kg + dm_r_kg))

    return {
        "dry_mass_kg": round(dry_mass_kg, 1),
        "low_fuel_mass_kg": round(m_low_kg, 1),
        "fuel_delta_mass_kg": round(dm_fuel_kg, 1),
        "fuel_front_weight_dist": round(fuel_front_dist, 3),
        "suspension_delta_mm": {
            "front": round(dz_f_mm, 3),
            "rear": round(dz_r_mm, 3),
            "apparent_mass_kg": round(dm_f_kg + dm_r_kg, 2),
        },
    }


def run_gt3_blackbox_calibration(
    calib_dir: str | Path,
    report_path: str | Path | None = None,
    dry_mass_kg: float | None = None,
    hotlap_fuel_liters: float = 10.0,
    fuel_density_kg_l: float = 0.75,
) -> dict[str, Any]:
    """Run the full Audi R8 LMS GT3 Evo II black-box identification pipeline on 200 Hz calibration data.

    Parameters
    ----------
    calib_dir : str | Path
        Directory containing ``baseline``, ``test1..test7``, ``test9``, ``test9_out``, and
        optionally ``2L.csv`` / ``110L.csv`` (all exported at **200 Hz**, ``dt = 0.005 s``).
    report_path : str | Path | None, optional
        If provided, writes the full JSON calibration report to this path.
    dry_mass_kg : float | None, optional
        Dry chassis mass [kg]. If ``None`` and ``2L.csv`` + ``110L.csv`` exist, automatically
        identifies ``dry_mass_kg`` via :func:`calibrate_mass`; otherwise defaults to ``1285.0``.
    hotlap_fuel_liters : float, default=10.0
        Fuel load [L] locked during ACC Hotlap calibration runs.
    fuel_density_kg_l : float, default=0.75
        Fuel density [kg/L].

    Returns
    -------
    dict[str, Any]
        Complete calibration dictionary with ``dry_mass_kg``, ``calibration_mass_kg``,
        ``aero_baseline``, ``calibration_slots``, ``units_and_formulas``, and diagnostics.
    """
    cdir = Path(calib_dir)
    mass_cal: dict[str, Any] | None = None
    if dry_mass_kg is None:
        if (cdir / "2L.csv").exists() and (cdir / "110L.csv").exists():
            mass_cal = calibrate_mass(cdir, fuel_density_kg_l=fuel_density_kg_l)
            dry_mass_kg = float(mass_cal["dry_mass_kg"])
        else:
            dry_mass_kg = 1285.0

    test_mass_kg = round(float(dry_mass_kg + hotlap_fuel_liters * fuel_density_kg_l), 1)
    z0_f_mm, z0_r_mm = _stationary_travel_datum_mm(cdir)

    consistency = verify_setup_consistency(cdir)
    baseline_df = _load_calib_csv(cdir / "baseline.csv")
    test5_df = _load_calib_csv(cdir / "test5.csv")

    bumpstop = calibrate_bumpstop_range(baseline_df, test5_df)
    aero = calibrate_aero(cdir, mass_kg=test_mass_kg, z0_f_mm=z0_f_mm, z0_r_mm=z0_r_mm)
    arb = calibrate_arb_stiffness(cdir, mass_kg=test_mass_kg)
    dampers = calibrate_dampers(cdir, mass_kg=test_mass_kg)

    report = {
        "vehicle_id": "audi_r8_gt3_evo2",
        "dry_mass_kg": float(dry_mass_kg),
        "calibration_mass_kg": test_mass_kg,
        "hotlap_fuel_liters": float(hotlap_fuel_liters),
        "mass_calibration": mass_cal,
        "stationary_datum_mm": {"front": z0_f_mm, "rear": z0_r_mm},
        "setup_consistency": consistency,
        "aero_baseline": aero["aero_baseline"],
        "calibration_slots": {
            "bumpstop_range_mm": bumpstop,
            "arb_stiffness_n_m": arb,
            "dampers_n_s_m": dampers,
            "aero_clicks": aero["aero_clicks"],
        },
        "units_and_formulas": {
            "convention": "y(click) = base + per_click * click, where `base` is always the value at click = 0 in the ACC setup UI.",
            "bumpstop_range_mm": {
                "unit": "mm (MoTeC SUS_TRAVEL threshold where BUMPSTOPUP_RIDE == 1)",
                "formula": "threshold_mm(click) = base_mm + per_click_mm * click; dynamic_gap_mm = threshold_mm - static_travel_mm (front 15 mm, rear 17 mm)",
                "mass_scaling": "independent of m (m^0)",
            },
            "arb_stiffness_n_m": {
                "unit": "N/m at wheel (effective differential axle roll stiffness above wheel rate k_w)",
                "formula": "K_roll_diff(click) = k_w + base + per_click * click = (2 * m_axle * g * h_cg) / (track_width * |dz / ay_g|)",
                "mass_scaling": f"linear in m (m / m_cal), where m_cal = {test_mass_kg} kg",
            },
            "dampers_n_s_m": {
                "unit": "N*s/m at wheel (vertical velocity threshold knee_velocity_mps = 0.05 m/s)",
                "formula": "c(click) = base + per_click * click; c_eff = 2 * (ln(z1 / z2) / pi) * sqrt(k_w * m_corner)",
                "mass_scaling": f"sqrt(m / m_cal), where m_cal = {test_mass_kg} kg",
            },
            "aero_clicks": {
                "unit": "dimensionless dCl, dCd referenced to A = 2.08 m^2, rho = 1.225 kg/m^3",
                "formula": "dFz_aero = q * dCl_per_click * click; dFd_aero = q * (rear_wing_dcd * w + front_duct_dcd * df + rear_duct_dcd * dr)",
                "mass_scaling": "dCl ~ m^0 (spring stroke); dCd linear in m (m / m_cal)",
            },
        },
        "aero_diagnostics": aero["diagnostics"],
    }

    if report_path is not None:
        rpath = Path(report_path)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
