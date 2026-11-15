"""Statistical validation, block bootstrapping, and anomaly georeferencing."""

from sindy_piml.stats.anomaly import AnomalySummary, map_anomalies_to_corners
from sindy_piml.stats.bootstrap import BootstrapSummary, TermConfidence, block_bootstrap_sindy
from sindy_piml.stats.diagnostics import ResidualDiagnostics, analyze_residuals

__all__ = [
    "ResidualDiagnostics",
    "analyze_residuals",
    "TermConfidence",
    "BootstrapSummary",
    "block_bootstrap_sindy",
    "AnomalySummary",
    "map_anomalies_to_corners",
]
