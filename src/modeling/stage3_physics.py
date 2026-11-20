"""Stage 3: First-Principles 4-Wheel Contact-Patch & Suspension Physics Modeling (No SINDy).

Uses calibrated vehicle constants and click-to-SI setup parameters from ``vehicle.yaml``
and ``setup.yaml`` (mass, geometry, spring wheel rates, ARB roll stiffness distribution,
bilinear dampers, bumpstops, and aerodynamic downforce) combined with a 4-corner Pacejka
Magic Formula tire model fitted via nonlinear least squares (``scipy.optimize.least_squares``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.config import CarConfig, ExperimentConfig
from src.modeling.evaluator import EvaluationResult

GRAVITY = 9.80665
AIR_DENSITY = 1.225
CORNERS: tuple[str, ...] = ("LF", "RF", "LR", "RR")


@dataclass
class Stage3PhysicsParams:
    """Identified tire-road & yaw-inertia physical parameters for Stage 3."""

    mu0: float = 1.45
    pacejka_b: float = 10.0
    pacejka_c: float = 1.5
    pacejka_e: float = -0.5
    load_sensitivity: float = 0.15
    iz_kg_m2: float = 2100.0
    vy_compliance_s: float = -0.05

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.mu0,
                self.pacejka_b,
                self.pacejka_c,
                self.pacejka_e,
                self.load_sensitivity,
                self.iz_kg_m2,
                self.vy_compliance_s,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Stage3PhysicsParams:
        return cls(
            mu0=float(arr[0]),
            pacejka_b=float(arr[1]),
            pacejka_c=float(arr[2]),
            pacejka_e=float(arr[3]),
            load_sensitivity=float(arr[4]),
            iz_kg_m2=float(arr[5]),
            vy_compliance_s=float(arr[6]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "mu0": round(self.mu0, 5),
            "pacejka_b": round(self.pacejka_b, 5),
            "pacejka_c": round(self.pacejka_c, 5),
            "pacejka_e": round(self.pacejka_e, 5),
            "load_sensitivity": round(self.load_sensitivity, 5),
            "iz_kg_m2": round(self.iz_kg_m2, 2),
            "vy_compliance_s": round(self.vy_compliance_s, 5),
        }


@dataclass
class Stage3SimulationResult:
    """Outputs of the 4-wheel Stage 3 physics forward simulation."""

    fz_corners: dict[str, np.ndarray]
    fz_susp_total: np.ndarray
    fz_aero_static_total: np.ndarray
    fy_corners: dict[str, np.ndarray]
    alpha_corners_rad: dict[str, np.ndarray]
    ay_pred_mps2: np.ndarray
    r_dot_pred_rad_s2: np.ndarray
    r_next_pred_rad_s: np.ndarray
    eval_ay: EvaluationResult
    eval_r_dot: EvaluationResult
    eval_r_next: EvaluationResult


def _resolve_setup_physics(car: CarConfig, setup: dict[str, Any], fuel_liters: float = 2.0) -> dict[str, float]:
    """Extract SI chassis, aero, spring, ARB, and steering parameters from calibrated configs."""
    mass_kg = float(car.mass_kg + fuel_liters * 0.75)
    susp = setup.get("suspension", {})
    wr = susp.get("wheel_rate_n_m", [153000.0, 153000.0, 163000.0, 163000.0])
    kw_f = float(wr[0])
    kw_r = float(wr[2])
    arb_clicks = susp.get("arb_clicks", [4, 2])

    arb_slots = car.calibration_slots.get("arb_stiffness_n_m", {})
    k_arb_f = float(arb_slots.get("front_base", 25000.0)) + float(arb_slots.get("front_per_click", 4300.0)) * float(
        arb_clicks[0]
    )
    k_arb_r = float(arb_slots.get("rear_base", 75000.0)) + float(arb_slots.get("rear_per_click", 5000.0)) * float(
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

    return {
        "mass_kg": mass_kg,
        "kw_f": kw_f,
        "kw_r": kw_r,
        "k_arb_f": k_arb_f,
        "k_arb_r": k_arb_r,
        "roll_front_share": roll_front_share,
        "cl_tot": cl_tot,
        "cl_f": cl_f,
        "cl_r": cl_r,
        "ref_area_m2": float(car.aero_baseline.reference_area_m2 or 2.08),
        "steering_ratio": float(setup.get("steering_ratio", 16.0)),
    }


def compute_corner_normal_loads(
    df: pd.DataFrame,
    car: CarConfig,
    setup: dict[str, Any],
    fuel_liters: float = 2.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute 4-corner normal loads ``F_z(t)`` [N] and cross-check against suspension telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Curated SI telemetry DataFrame containing ``SPEED`` [m/s], ``G_LAT`` [m/s^2],
        ``AX_SPEED`` (or ``G_LON``) [m/s^2], and optionally ``SUS_TRAVEL_*`` [mm] &
        ``BUMPSTOP_FORCE_*`` [N].
    car : CarConfig
        Calibrated vehicle physical constants & black-box slots from ``vehicle.yaml``.
    setup : dict[str, Any]
        Per-track setup dictionary from ``setup.yaml``.
    fuel_liters : float, default=2.0
        Fuel load [L] during the run (``0.75 kg/L``).

    Returns
    -------
    fz_corners : dict[str, np.ndarray]
        Per-corner normal loads [N] for ``LF``, ``RF``, ``LR``, ``RR``.
    fz_susp_total : np.ndarray
        Sum of spring + bumpstop forces [N] from ``SUS_TRAVEL_*`` & ``BUMPSTOP_FORCE_*``
        (equals ``fz_aero_static_total`` if suspension columns are absent).
    fz_aero_static_total : np.ndarray
        Total vertical load ``m*g + 0.5*rho*A*Cl*v^2`` [N].
    """
    phys = _resolve_setup_physics(car, setup, fuel_liters=fuel_liters)
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
        # Reference static travel datum (15 mm front, 17 mm rear)
        z0_f, z0_r = 0.015, 0.017
        fz_susp = (
            m * GRAVITY
            + phys["kw_f"] * ((z_lf - z0_f) + (z_rf - z0_f))
            + phys["kw_r"] * ((z_lr - z0_r) + (z_rr - z0_r))
        )
        for c in CORNERS:
            bs_col = f"BUMPSTOP_FORCE_{c}"
            if bs_col in df.columns:
                fz_susp = fz_susp + df[bs_col].to_numpy(dtype=float)
        fz_susp_total = fz_susp
    else:
        fz_susp_total = fz_aero_static_total.copy()

    return fz_corners, fz_susp_total, fz_aero_static_total


@dataclass
class Stage3WheelCornerModel:
    """First-principles 4-wheel Pacejka + load-transfer vehicle dynamics model (No SINDy)."""

    car: CarConfig
    setup: dict[str, Any]
    fuel_liters: float = 2.0
    params: Stage3PhysicsParams = field(default_factory=Stage3PhysicsParams)
    ref_mean_r_next: float = 0.0
    ref_std_r_next: float = 1.0

    def _corner_eta(self, df: pd.DataFrame, corner: str) -> np.ndarray:
        col = f"TYRE_PRESS_{corner}"
        if col not in df.columns:
            return np.ones(len(df), dtype=float)
        p = df[col].to_numpy(dtype=float)
        return np.exp(-((p - self.car.tire_p_opt_psi) ** 2) / (2.0 * (self.car.tire_sigma_p_psi**2)))

    def _forward(self, df: pd.DataFrame, p: Stage3PhysicsParams) -> Stage3SimulationResult:
        phys = _resolve_setup_physics(self.car, self.setup, fuel_liters=self.fuel_liters)
        m = phys["mass_kg"]
        lf = (1.0 - self.car.front_weight_dist) * self.car.wheelbase_m
        lr = self.car.front_weight_dist * self.car.wheelbase_m
        wf = self.car.track_width_front_m
        wr = self.car.track_width_rear_m

        t = df["Time"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
        v = np.clip(df["SPEED"].to_numpy(dtype=float), 5.0, None)
        r = df["ROTY"].to_numpy(dtype=float)
        ay_obs = df["G_LAT"].to_numpy(dtype=float)
        r_dot_obs = df["R_DOT"].to_numpy(dtype=float)
        delta_w = df["STEERANGLE"].to_numpy(dtype=float) / phys["steering_ratio"]

        fz_corners, fz_susp_total, fz_aero_static_total = compute_corner_normal_loads(
            df, self.car, self.setup, fuel_liters=self.fuel_liters
        )
        vy = p.vy_compliance_s * ay_obs

        alpha_corners = {
            "LF": delta_w - np.arctan2(vy + lf * r, np.clip(v - 0.5 * wf * r, 3.0, None)),
            "RF": delta_w - np.arctan2(vy + lf * r, np.clip(v + 0.5 * wf * r, 3.0, None)),
            "LR": -np.arctan2(vy - lr * r, np.clip(v - 0.5 * wr * r, 3.0, None)),
            "RR": -np.arctan2(vy - lr * r, np.clip(v + 0.5 * wr * r, 3.0, None)),
        }

        fy_corners: dict[str, np.ndarray] = {}
        for c in CORNERS:
            fz = fz_corners[c]
            eta = self._corner_eta(df, c)
            d_peak = p.mu0 * ((fz / 3500.0) ** (-p.load_sensitivity)) * eta * fz
            x = p.pacejka_b * alpha_corners[c]
            fy_corners[c] = d_peak * np.sin(
                p.pacejka_c * np.arctan(x - p.pacejka_e * (x - np.arctan(x)))
            )

        fy_front = (fy_corners["LF"] + fy_corners["RF"]) * np.cos(delta_w)
        fy_rear = fy_corners["LR"] + fy_corners["RR"]
        ay_pred = (fy_front + fy_rear) / m
        r_dot_pred = (lf * fy_front - lr * fy_rear) / max(p.iz_kg_m2, 100.0)

        # 1-step Euler forward yaw-rate prediction r_{k+1} = r_k + dt * r_dot_pred_k
        r_next_true = r[1:]
        r_next_pred = r[:-1] + dt * r_dot_pred[:-1]

        def _make_eval(y_t: np.ndarray, y_p: np.ndarray, mu_r: float | None = None, sig_r: float | None = None) -> EvaluationResult:
            res = y_t - y_p
            ss_res = float(np.sum(res**2))
            ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(np.mean(res**2)))
            mae = float(np.mean(np.abs(res)))
            mu = float(np.mean(res)) if mu_r is None else mu_r
            sig = float(np.std(res, ddof=1)) if sig_r is None else sig_r
            return EvaluationResult(
                y_true=y_t,
                y_pred=y_p,
                residuals=res,
                z_scores=(res - mu) / max(sig, 1e-12),
                r2=r2,
                rmse=rmse,
                mae=mae,
                residual_mean=float(np.mean(res)),
                residual_std=float(np.std(res, ddof=1)),
            )

        return Stage3SimulationResult(
            fz_corners=fz_corners,
            fz_susp_total=fz_susp_total,
            fz_aero_static_total=fz_aero_static_total,
            fy_corners=fy_corners,
            alpha_corners_rad=alpha_corners,
            ay_pred_mps2=ay_pred,
            r_dot_pred_rad_s2=r_dot_pred,
            r_next_pred_rad_s=r_next_pred,
            eval_ay=_make_eval(ay_obs, ay_pred),
            eval_r_dot=_make_eval(r_dot_obs, r_dot_pred),
            eval_r_next=_make_eval(r_next_true, r_next_pred, self.ref_mean_r_next, self.ref_std_r_next),
        )

    def fit(self, df_train: pd.DataFrame, max_nfev: int = 200) -> Stage3SimulationResult:
        """Fit Pacejka tire parameters & yaw inertia ``I_z`` via nonlinear least squares.

        Parameters
        ----------
        df_train : pd.DataFrame
            Curated 20 Hz SI training DataFrame (with ``STEERANGLE``, ``ROTY``, ``SPEED``,
            ``G_LAT``, ``R_DOT``, ``AX_SPEED``, and ``TYRE_PRESS_*``).
        max_nfev : int, default=200
            Maximum function evaluations for ``scipy.optimize.least_squares``.

        Returns
        -------
        Stage3SimulationResult
            Simulation & evaluation metrics on ``df_train``.
        """
        x0 = self.params.to_array()
        lower = np.array([0.8, 2.0, 1.0, -2.0, 0.0, 1000.0, -0.5], dtype=float)
        upper = np.array([2.5, 25.0, 2.0, 1.0, 0.4, 3500.0, 0.5], dtype=float)

        def residuals_fn(x: np.ndarray) -> np.ndarray:
            sim = self._forward(df_train, Stage3PhysicsParams.from_array(x))
            return np.concatenate(
                [
                    sim.eval_ay.residuals / 10.0,
                    sim.eval_r_dot.residuals,
                ]
            )

        opt = least_squares(residuals_fn, x0=x0, bounds=(lower, upper), max_nfev=max_nfev)
        self.params = Stage3PhysicsParams.from_array(opt.x)
        sim_fit = self._forward(df_train, self.params)
        self.ref_mean_r_next = sim_fit.eval_r_next.residual_mean
        self.ref_std_r_next = sim_fit.eval_r_next.residual_std
        return self._forward(df_train, self.params)

    def evaluate(self, df: pd.DataFrame) -> Stage3SimulationResult:
        """Evaluate the fitted Stage 3 physics model out-of-sample on ``df``."""
        return self._forward(df, self.params)

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "stage3_wheel_corner_physics",
                    "vehicle_id": self.car.car_id,
                    "fuel_liters": self.fuel_liters,
                    "params": self.params.to_dict(),
                },
                f,
                indent=2,
            )


def run_stage3_pipeline(
    cfg: ExperimentConfig,
    df_train: pd.DataFrame,
    df_ver: pd.DataFrame | None = None,
    fuel_liters: float = 2.0,
) -> tuple[Stage3WheelCornerModel, Stage3SimulationResult, Stage3SimulationResult | None]:
    """Fit Stage 3 4-wheel physics model on ``df_train`` and optionally evaluate on ``df_ver``."""
    model = Stage3WheelCornerModel(car=cfg.car, setup=cfg.setup, fuel_liters=fuel_liters)
    train_res = model.fit(df_train)
    ver_res = model.evaluate(df_ver) if df_ver is not None else None
    return model, train_res, ver_res
