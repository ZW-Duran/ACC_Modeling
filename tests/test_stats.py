import numpy as np
import pandas as pd

from src.config import CornerRegion, ModelingConfig
from src.features.planar_force import PlanarForceLibrary
from src.stats.anomaly import map_anomalies_to_corners
from src.stats.bootstrap import block_bootstrap_sindy
from src.stats.diagnostics import analyze_residuals


def test_residual_diagnostics_gaussian():
    rng = np.random.default_rng(42)
    res = rng.normal(0.0, 0.2, size=2000)
    z = res / 0.2
    diag = analyze_residuals(res, z, seed=42)
    assert diag.ks_pvalue > 0.05
    assert diag.shapiro_pvalue > 0.05
    assert diag.within_3sigma_pct > 99.0
    assert diag.beyond_5sigma_count == 0


def test_block_bootstrap_and_anomaly_mapping():
    rng = np.random.default_rng(7)
    n = 600
    r = rng.normal(0.0, 0.3, size=n)
    ax = rng.normal(0.0, 4.0, size=n)
    ay = rng.normal(0.0, 6.0, size=n)
    eta_p = np.ones(n)
    for k in range(1, n - 1):
        r[k + 1] = 0.95 * r[k] + 0.01 * ay[k] + rng.normal(0.0, 0.005)

    df = pd.DataFrame(
        {
            "Time": np.arange(n) * 0.05,
            "LAP_INDEX": np.ones(n, dtype=int),
            "LAP_DIST_M": np.linspace(400.0, 1200.0, n),
            "SPEED": rng.uniform(20.0, 60.0, size=n),
            "ROTY": r,
            "AX_SPEED": ax,
            "G_LAT": ay,
            "ETA_P": eta_p,
        }
    )

    feats = PlanarForceLibrary().build(df)
    coef = np.array([0.95, 0.01, 0.0, 0.0, 0.0, 0.0])
    boot = block_bootstrap_sindy(
        feats,
        coef,
        ModelingConfig(stlsq_threshold=0.002, stlsq_alpha=0.0, normalize_columns=True),
        block_size=50,
        n_resamples=30,
        seed=7,
    )
    assert len(boot.terms) == 6
    assert boot.terms[0].ci_95_low < 0.98 and boot.terms[0].ci_95_high > 0.92

    z = np.zeros(n - 2)
    z[10] = 6.2
    z[500] = -5.5
    corners = [
        CornerRegion("T1-T2 Elf", 350, 850),
        CornerRegion("T3 Renault", 850, 1450),
    ]
    anom = map_anomalies_to_corners(df, z, corners, threshold_z=5.0)
    assert anom.total_anomalies == 2
    assert anom.corner_counts["T1-T2 Elf"] == 1
    assert anom.corner_counts["T3 Renault"] == 1
