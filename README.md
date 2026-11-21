# ACC_Modeling: Multi-Stage Vehicle Dynamics Modeling & Telemetry Calibration

Physics-informed vehicle dynamics modeling, black-box setup calibration, and benchmark identification from **Assetto Corsa Competizione (ACC) MoTeC telemetry** (**20 Hz** for whole-car planar-force modeling; **200 Hz** for suspension/damper black-box calibration).

---

## Engineering Evolution & 4-Stage Modeling Roadmap

This project systematically climbs the vehicle-dynamics modeling hierarchy—using **Stage 2 (SINDy)** as a sparse whole-vehicle benchmark, validating full-chassis physics in **Stage 3 (`stage3_Verification`)**, and delivering a **Stage 4 closed-loop 7-DOF inverse-dynamics diagnostic & counterfactual setup simulation CLI (`main.py`)**:

### Stage 1 — Driver-Input Kinematic Regression *(Deprecated)*
- **Approach**: Initial prototypes regressed coupled vehicle states (`SPEED`, `G_LAT`, `G_LON`, `ROTY`) directly against raw driver inputs (`STEERANGLE`, `THROTTLE`, `BRAKE`) using generic polynomials.
- **Why Deprecated**: Driver control inputs decouple from chassis response whenever tires saturate (mid-corner understeer, counter-steering, TC/ABS interventions, dirty air, and traffic in race stints), creating severe non-stationary variance and unphysical terms.

### Stage 2 — Whole-Vehicle Discrete-Time Planar Force SINDy *(Completed Benchmark)*
- **Telemetry Sampling Policy**: `training_Barcelona_GT3.csv` and `verification_Barcelona_GT3.csv` are exported at **20 Hz** (`dt = 0.05 s`, `savgol_window=5`), matching the native 20 Hz update rate of ACC's `ROTY` and `G_LAT` channels and eliminating zero-order-hold differentiation ringing. Meanwhile, `data/calibration/` retains **200 Hz** exports for high-frequency suspension/damper velocity calibration.
- **Physical Rationale**: Because ACC's `G_LON` channel is integer-quantized (`{-2, -1, 0, 1} g`), longitudinal acceleration $a_x \approx \dot{v}$ is computed directly by differentiating `SPEED` (`m/s`). To resolve the $90^\circ$ phase-orthogonality between $r$ and $\dot{r}$, Stage 2 models the discrete-time state transition $r_{k+1} = F_A(\dots)$ via **PySINDy (`STLSQ`)**:
  $$r_{k+1} = F_A\left(r_k,\; a_{y,k},\; a_{y,k-1},\; a_{y,k} a_{x,k},\; \frac{a_{y,k}(a_{x,k}^2 + a_{y,k}^2)}{\eta_{P,k}^2},\; r_k|r_k|\right)$$
- **Finding & Benchmark Role**: On Audi R8 LMS GT3 Evo II (`Barcelona R8_A`), the discrete-time formulation achieves **$R^2 = 0.9865$** (`RMSE = 0.0393 rad/s`) on full-lap training and **$R^2 = 0.9829$** (`RMSE = 0.0454 rad/s`) on out-of-sample verification, with consistent `0.97–0.99` $R^2$ across individual corners (`T3`, `T4`, `T5`, `T13–T15`). Stage 2 serves as our **baseline benchmark**.

### Stage 3 — Full-Chassis Physics, Aero Drag & Bumpstop Verification *(No SINDy)*
- **Unified Verification (`src/modeling/stage3_verification.py`, `notebooks/stage3_Verification.ipynb`)**: Combines zero-shot black-box calibration verification (`src/calibration/verification.py`) with full-chassis suspension travel, progressive bumpstop force, and main-straight aerodynamic drag verification using `AX_SPEED = d(SPEED)/dt` (since `G_LON` is unusable), `G_LAT` (converted from `g` to `m/s^2`), and direct MoTeC `Distance`:
  - **Main-Straight Aero Drag (`0–700 m`)**: Integrates `wing = 0` WOT acceleration plus setup `dcd_wing` and `dcd_ducts` (`RMSE = 0.90 km/h` Training, `1.03 km/h` Verification).
  - **Zero-Shot Calibration & Progressive Bumpstop**: `22.0 mm` front / `38.0 mm` rear engagement thresholds (`96.7–99.5%` match accuracy) plus ACC's progressive bumpstop law ($F_{\mathrm{bs}} = k_{\mathrm{prog}}(z - z_{\mathrm{thr}})^{2.73}$, matching negative-compression MoTeC `BUMPSTOP_FORCE`), reducing in-bumpstop front travel error (`T1` heavy braking `LF RMSE = 2.17 mm`).
  - **5 Continuous Barcelona Segments**:
    1. **Main Straight (`0–710 m`)**: Aero drag & pitch attitude (`Mean Travel RMSE = 1.75 mm`).
    2. **T1 Heavy Braking (`700–960 m`)**: Longitudinal pitch transfer & progressive front bumpstop (`Mean Travel RMSE = 2.71 mm`).
    3. **T3 High-Speed Long Corner (`960–1550 m`)**: Aero + ARB lateral load transfer (`Mean Travel RMSE = 1.87 mm`).
    4. **T4–T5 Trail Braking (`1550–2350 m`)**: Combined diagonal load transfer (`Mean Travel RMSE = 2.20 mm`).
    5. **T10–T16 Continuous Complex to Finish (`3150–4586 m`)**: T10 heavy braking + T11 curb strike + T12–T16 chicane (`Mean Travel RMSE = 2.90 mm` Training / `2.76 mm` Verification; plotted as separate full-width Front and Rear figures).

#### Black-Box Setup Parameter Calibration *(Click-to-SI Identification)*
Because ACC exposes most garage settings as integer UI clicks, [calibration/blackbox.py](src/calibration/blackbox.py) identifies physical parameters for `configs/cars/<vehicle_id>/vehicle.yaml` from controlled single-variable Hotlap telemetry using the **`click = 0` intercept convention** $y(\text{click}) = y_{\text{base}} + k \cdot \text{click}$:
- **Mass Identification (`calibrate_mass`, unit: $\mathrm{kg}$)**: Solves dry mass from 5th-gear WOT acceleration ratio between `2L.csv` and `110L.csv` (`1161.2 kg` dry) and fuel front share (`29.3%`) from straight-line spring compression.
- **Aerodynamics (`aero_baseline`, `aero_clicks`, dimensionless $C_L, C_D$)**: Identifies $C_L$ and front aero balance from pitch-corrected straight-line spring compression $\Delta F_z = k_w \Delta z \propto \frac{1}{2}\rho A v^2$, and separates rear-wing, front brake-duct, and rear brake-duct $\Delta C_D$ via speed-matched full-throttle acceleration differences $\Delta F_d = -m \Delta a_x$.
- **Bumpstop Engagement (`bumpstop_range_mm`, unit: $\mathrm{mm}$)**: Solves the linear engagement `SUS_TRAVEL` threshold where `BUMPSTOPUP_RIDE == 1` across two bumpstop window settings.
- **Anti-Roll Bars (`arb_stiffness_n_m`, unit: $\mathrm{N/m}$ at wheel)**: Extracts differential wheel-referenced roll stiffness $K_{\mathrm{roll,diff}} - k_w$ from steady-state cornering roll gradients $|\Delta z_{\mathrm{L-R}} / a_y|$ with bumpstop-free filtering.
- **Virtual Damper Dyno (`dampers_n_s_m`, unit: $\mathrm{N\cdot s/m}$ at wheel)**: Extracts bump/rebound and slow/fast damping around `knee_velocity_mps = 0.05 m/s` using high-pass half-cycle logarithmic ringdown $\zeta = \ln(z_1/z_2)/\pi,\; c = 2\zeta\sqrt{k_w m_{\mathrm{corner}}}$.

### Stage 4 — 7-DOF Inverse-Dynamics Grip Utilization & Closed-Loop Setup CLI (`main.py`)
- **Design Principle**: Purely deterministic physical telemetry feature extraction (`diagnose`) + external AI multi-parameter reasoning via `ai_handoff.json` (`physics_model_spec`, `valid_setup_bounds`, `expected_ai_response_schema`) + local counterfactual 7-DOF replay (`simulate`). Zero hardcoded setup recommendation rules in Python.
- **CLI Subcommands (`src/modeling/stage4_diagnostic.py`, `main.py`)**:
  - `python main.py diagnose`: Outputs `ai_handoff.json` and 4-quadrant / instantaneous LLTD plots across Entry/Apex/Exit of every corner in `configs/tracks/barcelona_catalunya.yaml` (front/rear grip utilization $\rho_F, \rho_R$, instantaneous LLTD jumps, bumpstop margins in mm, alignment proxies, and critical damping ratios $\zeta_F, \zeta_R$).
  - `python main.py simulate`: Replays an AI-generated `setup_v2.json` through the coupled 7-DOF nonlinear chassis + progressive bumpstop + aero + alignment model against baseline `setup_v1.yaml`, generating `simulation_diff_report.json` and comparative traces.

---

## Stage 2 Benchmark Results (Circuit de Barcelona-Catalunya)

Controlled mid-engine V10 SINDy benchmark between **Audi R8 LMS GT3 Evo II** (high aerodynamic downforce) and **Audi R8 LMS GT4** (mechanical-grip baseline):

| Metric | Audi R8 LMS GT3 Evo II | Audi R8 LMS GT4 |
| :--- | :---: | :---: |
| **Clean Practice Samples** | `215,563` | `251,723` |
| **Clean Practice $R^2$ / RMSE** | **`0.4086`** / `0.5189 rad/s²` | **`0.5251`** / `0.3208 rad/s²` |
| **Race Stress-Test Samples** | `131,557` | `141,960` |
| **Race $R^2$ / RMSE** | `0.4457` / `0.4802 rad/s²` | `0.4227` / `0.4019 rad/s²` |
| **Top Block-Bootstrap Terms** | `a_y_dot/v` ($p = 7.65\times 10^{-15}$)<br>`a_y_dot` ($p = 6.06\times 10^{-5}$) | `a_y_dot` ($p = 2.77\times 10^{-4}$)<br>`a_y - v*r` ($p = 0.014$) |
| **Race $\pm 3\sigma_0$ Coverage** | `98.73%` | `97.68%` |
| **Race $>5\sigma_0$ Anomalies** | `732` frames (Gaussian exp: `0.08`) | `1,412` frames (Gaussian exp: `0.08`) |
| **Top Anomaly Corners** | `T9 Campsa` (261), `Sector 3` (141), `T5 Seat` (118) | `Sector 3` (684), `T3 Renault` (331), `T7-T8` (190) |

### Diagnostic Visualizations

#### Audi R8 LMS GT3 Evo II
![GT3 Yaw Acceleration Fit](outputs/stage2_barcelona_gt3_r8_A/clean_yaw_fit.png)
![GT3 Residual Distribution & Q-Q](outputs/stage2_barcelona_gt3_r8_A/clean_residuals_qq.png)
![GT3 Race Anomaly Trace](outputs/stage2_barcelona_gt3_r8_A/race_anomaly_trace.png)

#### Audi R8 LMS GT4 (Mechanical-Grip Baseline)
> **Status:** `Pending Re-calibration` — *Currently migrating from the legacy 200 Hz continuous-derivative pipeline to the updated 20 Hz discrete-time formulation to match native ACC yaw-rate update frequencies.*
---

## Repository Structure

```text
ACC_Modeling/
├── configs/
│   ├── cars/
│   │   ├── audi_r8_gt3_evo2/
│   │   │   ├── vehicle.yaml      # Chassis constants, aero_baseline, calibrated click-to-SI slots
│   │   │   ├── convert_setup.py  # ACC setup .json -> .yaml converter for GT3 Evo II
│   │   │   └── setups/
│   │   │       └── Barcelona/    # Track-grouped setups matching ACC profile names (R8_A.json, r8_A.yaml)
│   │   └── audi_r8_gt4/
│   │       ├── vehicle.yaml      # GT4 chassis constants, aero_baseline, calibration slots
│   │       ├── convert_setup.py  # ACC setup .json -> .yaml converter for GT4
│   │       └── setups/
│   │           └── Barcelona/    # Track-grouped setups matching ACC profile names (a1.json, a1.yaml)
│   ├── tracks/                   # Circuit metadata & corner distance boundaries
│   ├── curation/                 # Savitzky-Golay & session-trimming presets
│   ├── modeling/                 # Identification optimizer & statistical validation presets
│   └── experiments/
│       ├── stage2_planar_force/             # Stage 2 whole-vehicle planar force SINDy manifests
│       └── stage3_verification/             # Stage 3 full-chassis, aero drag & bumpstop verification manifests
├── data/                         # MoTeC telemetry (.csv/.ld/.ldx, gitignored; see data/README.md)
│   └── calibration/              # Controlled single-variable Hotlap CSVs for black-box calibration
├── models/                       # Exported Stage 2 fitted models (.json)
├── outputs/
│   ├── calibration/                                     # Black-box identification reports (.json)
│   ├── stage2_barcelona_gt3_r8_A/                       # GT3 Stage 2 full-lap & corner diagnostic plots & report
│   ├── stage3_barcelona_gt3_verification/               # GT3 Stage 3 aero drag, bumpstop & scenario verification plots & report
│   └── stage4_diagnosis/                                # Stage 4 ai_handoff.json, simulation_diff_report.json & plots
├── main.py                       # Top-level closed-loop CLI (`diagnose` & `simulate`)
├── notebooks/
│   ├── stage2_gt3_discrete_sindy.ipynb                  # Stage 2 GT3 discrete-time SINDy modeling (benchmark baseline)
│   └── stage3_Verification.ipynb                        # Stage 3 unified aero drag, bumpstop & full-chassis verification
├── src/                          # Source package (`from src import ...`; see src/README.md for full API index)
│   ├── README.md                 # Complete API Reference & Module Index (signatures, units, sampling rates)
│   ├── cli.py                    # Modular CLI entry point (`acc-modeling`)
│   ├── config.py                 # Composable YAML loader (Car, Setup, Track, Curation, Modeling)
│   ├── calibration/              # Click-to-SI black-box calibration from controlled Hotlap telemetry
│   ├── data/                     # MoTeC parser, session trimming, SG curation, ACC setup conversion
│   ├── features/
│   │   ├── base.py               # Abstract CandidateLibrary / feature-matrix interface
│   │   ├── planar_force.py       # Stage 2: Whole-vehicle planar force SINDy library
│   │   ├── wheel_corner.py       # Stage 3: Single-wheel load-transfer & suspension schema
│   │   └── full_chassis.py       # Stage 4: Full-chassis & setup parameter interface
│   ├── modeling/                 # Stage 2 SINDy STLSQ trainer & evaluator
│   ├── stats/                    # Normality/omission sweeps, Block Bootstrap, corner anomaly mapping
│   └── visualization/            # Residual Q-Q, time-series fit, and race anomaly plots
├── tests/                        # Pytest unit & calibration tests
└── pyproject.toml
```

## Naming Conventions

To prevent mismatches between ACC setups, MoTeC telemetry exports, and experiment configs:

- **Per-Car Setups**: `configs/cars/<vehicle_id>/setups/<Track>/<profile>.yaml`
  - Example: `configs/cars/audi_r8_gt3_evo2/setups/Barcelona/r8_A.yaml` (paired with `R8_A.json`)
- **Telemetry CSVs**: `data/<track>__<car>__<setupProfile>__<session>.csv`
  - Example: `data/barcelona_catalunya__audi_r8_gt3_evo2__r8_A__practice.csv`
- **Experiments & Outputs**: `configs/experiments/stage<N>_*/<track>__<car>__<setupProfile>.yaml`

## Installation & Usage

```bash
pip install -e ".[dev]"
pytest

# Run a Stage 2 SINDy benchmark experiment
acc-modeling --config configs/experiments/stage2_planar_force/barcelona__audi_r8_gt3_evo2__r8_A.yaml
acc-modeling --config configs/experiments/stage2_planar_force/barcelona__audi_r8_gt4__a1.yaml

# Or swap individual modular configs on the fly
acc-modeling --config configs/experiments/stage2_planar_force/barcelona__audi_r8_gt3_evo2__r8_A.yaml \
  --car configs/cars/audi_r8_gt4/vehicle.yaml \
  --setup configs/cars/audi_r8_gt4/setups/Barcelona/a1.yaml

# Run a Stage 4 closed-loop AI diagnostic handoff & counterfactual simulation
python main.py diagnose \
  --csv data/training_Barcelona_GT3.csv \
  --vehicle configs/cars/audi_r8_gt3_evo2/vehicle.yaml \
  --setup configs/cars/audi_r8_gt3_evo2/setups/Barcelona/r8_A.yaml \
  --track configs/tracks/barcelona_catalunya.yaml \
  --out outputs/stage4_diagnosis/ai_handoff.json

python main.py simulate \
  --baseline-csv data/training_Barcelona_GT3.csv \
  --vehicle configs/cars/audi_r8_gt3_evo2/vehicle.yaml \
  --base-setup configs/cars/audi_r8_gt3_evo2/setups/Barcelona/r8_A.yaml \
  --candidate-setup outputs/stage4_diagnosis/setup_v2.json \
  --track configs/tracks/barcelona_catalunya.yaml \
  --out outputs/stage4_diagnosis/simulation_diff_report.json

# Convert an ACC setup .json into repository setup YAML format
python configs/cars/audi_r8_gt3_evo2/convert_setup.py \
  configs/cars/audi_r8_gt3_evo2/setups/Barcelona/R8_A.json \
  -o configs/cars/audi_r8_gt3_evo2/setups/Barcelona/r8_A.yaml
```
