"""Statistical validation, block bootstrapping, and anomaly georeferencing."""

from src.stats.anomaly import AnomalySummary, map_anomalies_to_corners
from src.stats.bootstrap import BootstrapSummary, TermConfidence, block_bootstrap_sindy
from src.stats.diagnostics import ResidualDiagnostics, analyze_residuals

__all__ = [
    "ResidualDiagnostics",
    "analyze_residuals",
    "TermConfidence",
    "BootstrapSummary",
    "block_bootstrap_sindy",
    "AnomalySummary",
    "map_anomalies_to_corners",
]
