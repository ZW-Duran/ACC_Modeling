# `src/` API Reference & Module Index

Complete reference of all public modules, classes, and functions under `src/` for cross-project reuse.  
All primary symbols are re-exported at top level via `src/__init__.py` (`from src import ...`).

---

## Global Conventions (Units & Sampling Rates)

| Quantity | Column / Variable | Unit | Notes |
| :--- | :--- | :--- | :--- |
| Time | `Time` | `s` | Stably sorted ascending on load (`load_motec_csv`) |
| Lap Distance | `Distance`, `LAP_DIST_M` | `m` | Direct MoTeC `Distance` channel used when exported (fallback: trapezoidal integration of `SPEED`) |
| Speed | `SPEED` | `m/s` | Converted from `km/h` |
| Yaw rate | `ROTY` ($r$) | `rad/s` | Converted from `deg/s` |
| Yaw acceleration | `R_DOT` ($\dot{r}$) | `rad/s^2` | Analytical SG 1st derivative of `ROTY` |
| Lateral accel | `G_LAT` | `m/s^2` | Exported by ACC in `g` (`~[-3, +3]`) even when labeled `m/s2`; auto-scaled by `9.80665` |
| Longitudinal accel | `AX_SPEED` | `m/s^2` | Computed via `d(SPEED)/dt` (`G_LON` in ACC CSV is integer-quantized/unusable) |
| Steering angle / rate | `STEERANGLE`, `STEER_DOT` | `rad`, `rad/s` | Converted from `deg` |
| Tire pressure & grip | `TYRE_PRESS_*`, `ETA_P` | `psi`, `[-]` | $\eta_P = \exp(-(\bar{P} - P_{\mathrm{opt}})^2 / 2\sigma_P^2)$ |
| Suspension travel | `SUS_TRAVEL_*` | `mm` | Positive = compression |

- **20 Hz (`dt = 0.05 s`, `savgol_window = 5`)**: Used by Stage 2 SINDy and Stage 3 full-chassis verification (`training_Barcelona_GT3.csv`, `verification_Barcelona_GT3.csv`) to match ACC's native 20 Hz `ROTY` / `G_LAT` update rate.
- **200 Hz (`dt = 0.005 s`)**: Required by `src/calibration/blackbox.py` to resolve `> 1.2 Hz` damper ringdown and bumpstop impact velocities.

---

## 1. Top-Level Entry Points (`src/`)

### `src/__init__.py`
Re-exports all public symbols across subpackages (`__version__ = "0.2.0"`).

### `src/cli.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `run_pipeline` | `run_pipeline(config_path, car_path=None, track_path=None, curation_path=None, modeling_path=None, setup_path=None)` | `str \| Path` YAML paths | `dict` (full report) | Runs curation $\to$ SINDy fit $\to$ bootstrap $\to$ diagnostics/plots $\to$ verification; writes `outputs/<exp>/` & `models/<exp>.json` |
| `main` | `main()` | CLI `--config`, `--car`, `--setup`, `--track`, `--curation`, `--modeling` | `None` | Entry point for `acc-modeling` console script |

### `src/config.py`
| Symbol | Kind | Key Fields / Signature | Notes |
| :--- | :--- | :--- | :--- |
| `AeroBaselineConfig` | `dataclass` | `reference_area_m2`, `base_cl`, `base_cd`, `base_aero_front_dist` | `.from_dict(raw)` |
| `CarConfig` | `dataclass` | `car_id`, `mass_kg`, `front_weight_dist`, `wheelbase_m`, `track_width_front_m`, `track_width_rear_m`, `cg_height_m`, `tire_p_opt_psi`, `aero_baseline`, `calibration_slots` | Supports nested `vehicle.yaml` schema via `.from_dict(raw)` |
| `TrackConfig`, `CornerRegion` | `dataclass` | `track_id`, `name`, `length_m`, `corners: list[CornerRegion(name, start_m, end_m)]` | Corner boundaries in `[m]` along lap |
| `DataConfig` | `dataclass` | `clean_csv`, `race_csv`, `clean_trim_mode`, `clean_speed_threshold_kmh`, `race_trim_mode` | Paths and trimming modes |
| `CurationConfig` | `dataclass` | `savgol_window=5`, `savgol_polyorder=3`, `p_opt_psi=27.0`, `sigma_p_psi=1.5`, ... | Defaults tuned for 20 Hz |
| `ModelingConfig` | `dataclass` | `formulation="planar_force"`, `stlsq_threshold=0.02`, `stlsq_alpha=0.05`, `normalize_columns=True` | PySINDy STLSQ hyperparameters |
| `StatsConfig` | `dataclass` | `bootstrap_n_resamples=500`, `bootstrap_block_seconds=2.0`, `random_seed=42`, `z_anomaly_threshold=5.0` | Block bootstrap & anomaly threshold |
| `ExperimentConfig` | `dataclass` | `experiment_name`, `car`, `setup`, `track`, `data`, `curation`, `modeling`, `stats` | Top-level composed experiment config |
| `load_config` | `function` | `load_config(experiment_path, car_path=None, track_path=None, curation_path=None, modeling_path=None, setup_path=None) -> ExperimentConfig` | Resolves relative YAML paths & merges overrides |

---

## 2. Data Loading, Curation & Setup Conversion (`src/data/`)

### `src/data/motec_loader.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `TelemetryMetadata` | `dataclass` | — | `filepath`, `venue`, `vehicle`, `sample_rate_hz`, `duration_s`, `beacon_markers_s`, `units` | Parsed from 18-line MoTeC CSV header |
| `load_motec_csv` | `load_motec_csv(filepath, channels=CORE_CHANNELS)` | `filepath: str \| Path`, `channels: Iterable[str]` | `(pd.DataFrame, TelemetryMetadata)` | Stably sorts by `Time`; converts to SI; auto-rescales ACC `G_LAT`/`G_LON` from `g` to `m/s^2` when `max <= 5.0` |

### `src/data/curation.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `trim_session` | `trim_session(df, meta, mode="speed_threshold", speed_threshold_kmh=100.0)` | `df: pd.DataFrame`, `meta: TelemetryMetadata` | `pd.DataFrame` | `speed_threshold` drops *only* initial pit-exit rows before first hit; slow corners `< 100 km/h` are kept |
| `compute_tire_grip_factor` | `compute_tire_grip_factor(df, p_opt_psi=27.0, sigma_p_psi=1.5)` | `df: pd.DataFrame` (`TYRE_PRESS_*` `[psi]`) | `(p_mean, eta_p)` (`np.ndarray`) | Returns `(p_opt_psi, 1.0)` if pressure columns absent |
| `reconstruct_lap_distance` | `reconstruct_lap_distance(time_s, speed_ms, beacon_markers_s)` | `time_s` `[s]`, `speed_ms` `[m/s]`, `beacon_markers_s` `[s]` | `(lap_idx, lap_dist_m)` (`np.ndarray`) | Trapezoidal integration per lap |
| `curate_telemetry` | `curate_telemetry(df, meta, cfg, trim_mode="speed_threshold", speed_threshold_kmh=100.0)` | `df`, `meta`, `cfg: CurationConfig` | `pd.DataFrame` | Adds `R_DOT` `[rad/s^2]`, `AY_DOT` `[m/s^3]`, `AX_SPEED` `[m/s^2]`, `P_MEAN_PSI`, `ETA_P`, `LAP_INDEX`, `LAP_DIST_M`; uses `dt = median(diff(Time))` |

### `src/data/acc_setup.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `AccSetupMapping` | `dataclass` | — | Per-car UI offset & lookup tables | Maps ACC JSON indices $\leftrightarrow$ garage UI values |
| `convert_acc_dict` | `convert_acc_dict(raw, mapping, setup_name)` | `raw: dict`, `mapping: AccSetupMapping`, `setup_name: str` | `dict` | Converts raw ACC setup JSON dict to human-readable YAML schema |
| `convert_acc_json_file` | `convert_acc_json_file(json_path, mapping, setup_name=None, output_path=None)` | `json_path: str \| Path`, optional `output_path` | `dict` | Reads ACC JSON and optionally writes `.yaml` |
| `run_converter_cli` | `run_converter_cli(mapping, description)` | `mapping: AccSetupMapping`, `description: str` | `None` | Shared CLI helper for per-car `convert_setup.py` |

---

## 3. Black-Box Setup Calibration & Stage 3 Verification (`src/calibration/`)

### `src/calibration/blackbox.py` *(200 Hz Identification)*
> **Sampling Assumption**: Assumes **200 Hz** (`dt = 0.005 s`) MoTeC telemetry exported from controlled single-variable Hotlap runs. All click coefficients follow the **`click = 0` intercept convention**: $y(\text{click}) = y_{\text{base}} + k \cdot \text{click}$.

| Symbol | Signature | Inputs / Key Args | Returns / Units | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `verify_setup_consistency` | `verify_setup_consistency(calib_dir)` | `calib_dir: str \| Path` | `dict` (pressures, bumpstop, roll gradients, $V_{\max}$) | Audits setup consistency across `baseline`, `test1..test9` |
| `calibrate_mass` | `calibrate_mass(calib_dir, low_fuel_liters=2.0, high_fuel_liters=110.0, fuel_density_kg_l=0.75, ...)` | `2L.csv` & `110L.csv` | `dict`: `dry_mass_kg` `[kg]`, `fuel_front_weight_dist` `[-]` | Uses 5th-gear WOT `ax` ratio (`60–67 m/s`) + straight suspension delta (`50–72 m/s`) |
| `calibrate_bumpstop_range` | `calibrate_bumpstop_range(baseline_df, test5_df, base_clicks=(5, 14), test5_clicks=(2, 10))` | `baseline_df`, `test5_df` | `dict` `[mm]` (`front_base_mm`, `front_per_click_mm`, `rear_*`) | Threshold where `BUMPSTOPUP_RIDE == 1`; mass-independent ($m^0$) |
| `calibrate_aero` | `calibrate_aero(calib_dir, mass_kg=1285.0, ref_area_m2=2.08, ...)` | `test1`, `test2`, `baseline`, `test3`, `test4` | `dict`: `aero_baseline` (`base_cl`, `base_aero_front_dist`), `aero_clicks` (`dCl`, `dCd` `[-]`) | Straight-line pitch-corrected $F_z/q$ and WOT $\Delta a_x$ |
| `calibrate_arb_stiffness` | `calibrate_arb_stiffness(calib_dir, mass_kg=1285.0, ...)` | `baseline`, `test2..test4`, `test6`, `test7` | `dict` `[N/m at wheel]` (`front_base`, `front_per_click`, `rear_*`) | Steady-state roll gradient; scales linearly with `mass_kg` |
| `calibrate_dampers` | `calibrate_dampers(calib_dir, kw_f_n_m=153000.0, kw_r_n_m=163000.0, mass_kg=1285.0, ...)` | `baseline`, `test9`, `test9_out` | `dict` `[N*s/m at wheel]` (`bump/rebound`, `slow/fast` base & slope) | `> 1.2 Hz` half-cycle ringdown log-decrement; scales as $\sqrt{m}$ |
| `run_gt3_blackbox_calibration` | `run_gt3_blackbox_calibration(calib_dir, report_path=None, dry_mass_kg=None, hotlap_fuel_liters=10.0, fuel_density_kg_l=0.75)` | `calib_dir: str \| Path` | `dict` (full JSON report) | Auto-runs `calibrate_mass` if `2L.csv` & `110L.csv` exist; updates all slots consistently |

### `src/calibration/verification.py` *(Stage 3 — Zero-Shot Calibration Verification)*
| Symbol | Signature / Members | Inputs / Args | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `resolve_setup_physics` | `resolve_setup_physics(car, setup, fuel_liters=2.0)` | `car: CarConfig`, `setup: dict` | `dict[str, float]` | Resolves SI `mass_kg`, wheel rates, ARB `k_eff`, roll share, aero `Cl`, bumpstop thresholds |
| `CalibrationVerificationResult` | `bumpstop`, `axle`, `corners`, time-series travel arrays, `.to_dict()` | — | — | Holds Stage 3 zero-shot verification metrics and arrays |
| `verify_calibrated_coefficients` | `verify_calibrated_coefficients(df_raw, df_curated, car, setup, fuel_liters=2.0)` | Raw & curated 20 Hz SI telemetry | `CalibrationVerificationResult` | Verifies bumpstop `BUMPSTOPUP_RIDE`, ARB roll $\Delta z_{L-R}$, aero heave, and 4-corner `SUS_TRAVEL` with zero fitting |

---

## 4. Feature Libraries (`src/features/`)

### `src/features/base.py`
| Symbol | Kind | Members / Signature | Notes |
| :--- | :--- | :--- | :--- |
| `FeatureMatrix` | `dataclass` | `theta: np.ndarray`, `target: np.ndarray`, `feature_names: list[str]`, `latex_names: list[str]`, `target_name: str` | Container passed to SINDy trainer/evaluator |
| `CandidateLibrary` | `ABC` | `build(df: pd.DataFrame) -> FeatureMatrix` | Abstract base class |

### `src/features/planar_force.py` *(Stage 2 — Active)*
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `PlanarForceLibrary.build` | `build(df)` | `df` (`N >= 3`) with `ROTY`, `G_LAT`, `SPEED`, `ETA_P`, `AX_SPEED` (or `G_LON`) | `FeatureMatrix`: `theta` `(N-2, 6)`, `target` `r_{k+1}` `(N-2,)` `[rad/s]` | Discrete-time 6-term library `[r_k, a_y_k, a_y_k-1, a_y_k*a_x_k, a_y_k*(a_x_k^2+a_y_k^2)/eta_P_k^2, r_k*|r_k|]`; align `df` with `df.iloc[2:]` |

### `src/features/wheel_corner.py` & `src/features/full_chassis.py` *(Reference Contract Scaffolds)*
- `estimate_corner_normal_load(df, corner, car, aero_cl_a=0.0, air_density=1.225)`
- `WheelCornerLibrary`, `ChassisSetupSpec`, `FullChassisLibrary`

---

## 5. Modeling & Evaluation (`src/modeling/`)

### `src/modeling/trainer.py` *(Stage 2 SINDy)*
| Symbol | Signature / Members | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `FittedYawModel` | `coefficients`, `feature_names`, `latex_names`, `optimizer`, `target_name` | — | `.predict(theta)`, `.active_mask()`, `.format_equation()`, `.to_dict()`, `.save_json(path)` | Stores fitted PySINDy STLSQ model |
| `fit_yaw_sindy` | `fit_yaw_sindy(features, cfg)` | `features: FeatureMatrix`, `cfg: ModelingConfig` | `FittedYawModel` | Sparse regression via `pysindy.STLSQ` |

### `src/modeling/evaluator.py`
| Symbol | Signature / Members | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `EvaluationResult` | `y_true`, `y_pred`, `residuals`, `z_scores`, `r2`, `rmse`, `mae`, `residual_mean`, `residual_std` | — | — | Residual vector is `y_true - y_pred` |
| `evaluate_model` | `evaluate_model(model, features, ref_mean=None, ref_std=None)` | `model: FittedYawModel`, `features: FeatureMatrix` | `EvaluationResult` | Pass training `ref_mean`/`ref_std` for out-of-sample Z-scores |

### `src/modeling/stage3_verification.py` *(Stage 3 — Full-Chassis Physics, Aero Drag & Bumpstop Verification)*
| Symbol | Signature / Members | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `AeroDragVerificationResult` | `dist_m`, `v_obs_kmh`, `v_pred_kmh`, `v_no_ducts_kmh`, `rmse_kmh`, `end_v_*_kmh`, `dcd_wing`, `dcd_ducts` | — | `.to_dict()` | Straight-line wing + brake-duct drag speed curve (`0–700 m`) |
| `SuspensionVerificationState` | `fz_corners_n`, `fz_susp_total_n`, `fz_aero_static_total_n`, `z_pred_mm`, `z_obs_mm`, `bs_force_pred_n`, `bs_force_obs_n`, `pitch_deg_pred`, `pitch_deg_obs` | — | — | Full-lap suspension travel, progressive bumpstop force, and normal load predictions |
| `ScenarioMetrics` | `key`, `label`, `start_m`, `end_m`, `n_samples`, `speed_range_kmh`, `mean_corner_rmse_mm`, `corner_rmse_mm`, `max_bs_force_*_n`, `bs_force_rmse_n` | — | `.to_dict()` | Per-segment verification summary metrics |
| `compute_corner_normal_loads` | `compute_corner_normal_loads(df, car, setup, fuel_liters=2.0)` | `df`, `car: CarConfig`, `setup: dict` | `(fz_corners, fz_susp_total, fz_aero_static_total)` `[N]` | Combines pitch (`AX_SPEED`), calibrated ARB roll share, and $v^2$ aero downforce |
| `verify_straight_aero_drag` | `verify_straight_aero_drag(df, car, setup, calib_dir, fuel_liters=2.0, ...)` | `df`, `car`, `setup`, `calib_dir` | `AeroDragVerificationResult` | Integrates `wing = 0` `test1.csv` WOT acceleration + `dcd_wing` + `dcd_ducts` over `0–700 m` |
| `predict_suspension_state` | `predict_suspension_state(df, car, setup, fuel_liters=2.0)` | `df`, `car`, `setup` | `SuspensionVerificationState` | Newton-Raphson progressive bumpstop equilibrium ($F_{\mathrm{bs}} \propto (z - z_{\mathrm{thr}})^{2.73}$) + spring + ARB + aero |
| `verify_stage3_scenarios` | `verify_stage3_scenarios(df, car, setup, fuel_liters=2.0)` | `df`, `car`, `setup` | `(SuspensionVerificationState, list[ScenarioMetrics])` | Evaluates all 5 continuous Barcelona segments (`SCENARIO_WINDOWS_M`, including `T10–T16`) |

---

## 6. Statistical Validation & Anomaly Mapping (`src/stats/`)

### `src/stats/diagnostics.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `analyze_residuals` | `analyze_residuals(residuals, z_scores, shapiro_max_n=5000, seed=42, z_sweep=(1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0))` | `residuals`, `z_scores: np.ndarray` | `ResidualDiagnostics` | Mean, std, skewness, excess kurtosis, KS, Shapiro-Wilk, `within_3sigma_pct`, Type-I omission sweep |

### `src/stats/bootstrap.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `block_bootstrap_sindy` | `block_bootstrap_sindy(features, point_coefficients, modeling_cfg, block_size=400, n_resamples=500, seed=42)` | `features`, `point_coefficients`, `modeling_cfg`, `block_size` (frames) | `BootstrapSummary` | Overlapping circular moving-block bootstrap; computes 95% CI & two-sided $p$-values per term |

### `src/stats/anomaly.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `map_anomalies_to_corners` | `map_anomalies_to_corners(df, z_scores, corners, threshold_z=5.0)` | `df`, `z_scores`, `corners: list[CornerRegion]` | `AnomalySummary` | Auto-slices `df.iloc[2:]` if `len(df) == len(z_scores) + 2` |

---

## 7. Visualization (`src/visualization/`)

### `src/visualization/diagnostics.py`
| Symbol | Signature | Inputs | Returns | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `plot_yaw_acceleration_fit` | `plot_yaw_acceleration_fit(df, eval_res, title, save_path, window_seconds=None)` | `df`, `eval_res: EvaluationResult` | `Path` | Full-lap (`window_seconds=None`) observed vs. predicted + residual trace; auto-aligns `N-2` to `Time[2:]` |
| `plot_finite_diff_yaw_accel` | `plot_finite_diff_yaw_accel(df, eval_res, title, save_path, dt=None)` | `df`, `eval_res: EvaluationResult` | `dict` (`r2_rd`, `rmse_rd`, `lag_frames`, `lag_ms`, `save_path`) | Strips `r_k` persistence via `(r_{k+1}-r_k)/dt` vs. raw `ROTY` difference & computes lag |
| `plot_residual_distribution` | `plot_residual_distribution(eval_res, title, save_path)` | `eval_res: EvaluationResult` | `Path` | Integer **Count** histogram + empirical KDE, **Laplace**, and **Student-$t$** fits + Laplace Q-Q plot |
| `plot_race_anomaly_trace` | `plot_race_anomaly_trace(df, z_scores, z_thresh, title, save_path)` | `df`, `z_scores`, `z_thresh: float` | `Path` | Standardized $Z(t)$ trace with $\pm 3\sigma$ and $\pm Z_{\mathrm{thresh}}$ outlier markers; auto-aligns `N-2` to `Time[2:]` |
