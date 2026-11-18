"""MoTeC i2 Pro CSV telemetry loader with metadata & unit-aware SI conversion."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CORE_CHANNELS: tuple[str, ...] = (
    "Time",
    "ROTY",
    "STEERANGLE",
    "SPEED",
    "G_LAT",
    "G_LON",
    "THROTTLE",
    "BRAKE",
    "TYRE_PRESS_LF",
    "TYRE_PRESS_RF",
    "TYRE_PRESS_LR",
    "TYRE_PRESS_RR",
)


@dataclass
class TelemetryMetadata:
    """Metadata parsed from the MoTeC CSV preamble."""

    filepath: Path
    venue: str = ""
    vehicle: str = ""
    sample_rate_hz: float = 200.0
    duration_s: float = 0.0
    beacon_markers_s: list[float] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)


def _parse_preamble(filepath: Path, n_header_lines: int = 18) -> tuple[TelemetryMetadata, list[str]]:
    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        lines = [next(reader) for _ in range(n_header_lines)]

    meta = TelemetryMetadata(filepath=filepath)

    for row in lines[:14]:
        if len(row) >= 2:
            key = row[0].strip()
            val = row[1].strip()
            if key == "Venue":
                meta.venue = val
            elif key == "Vehicle":
                meta.vehicle = val
            elif key == "Sample Rate":
                try:
                    meta.sample_rate_hz = float(val)
                except ValueError:
                    pass
            elif key == "Duration":
                try:
                    meta.duration_s = float(val)
                except ValueError:
                    pass
            elif key == "Beacon Markers" and val:
                meta.beacon_markers_s = [float(x) for x in val.split() if x.strip()]

    columns = [c.strip() for c in lines[14]]
    units = [u.strip() for u in lines[15]]
    meta.units = {col: unit for col, unit in zip(columns, units)}
    return meta, columns


def _convert_to_si(df: pd.DataFrame, units: dict[str, str]) -> pd.DataFrame:
    """Convert telemetry channels to SI units based on header unit metadata."""
    out = df.copy()

    if "STEERANGLE" in out.columns:
        unit = units.get("STEERANGLE", "deg").lower()
        if "deg" in unit:
            out["STEERANGLE"] = out["STEERANGLE"] * (np.pi / 180.0)

    if "ROTY" in out.columns:
        unit = units.get("ROTY", "deg/s").lower()
        if "deg" in unit:
            out["ROTY"] = out["ROTY"] * (np.pi / 180.0)

    if "SPEED" in out.columns:
        unit = units.get("SPEED", "km/h").lower()
        if "km/h" in unit or "kph" in unit:
            out["SPEED"] = out["SPEED"] / 3.6

    for g_col in ("G_LAT", "G_LON"):
        if g_col in out.columns:
            unit = units.get(g_col, "m/s2").lower()
            # ACC MoTeC exports often label G_LAT/G_LON as "m/s2" while storing values in g (~[-3, +3]).
            if unit == "g" or (not out[g_col].empty and float(out[g_col].abs().max()) <= 5.0):
                out[g_col] = out[g_col] * 9.80665

    return out


def load_motec_csv(
    filepath: str | Path,
    channels: Iterable[str] = CORE_CHANNELS,
) -> tuple[pd.DataFrame, TelemetryMetadata]:
    """Load an ACC MoTeC i2 Pro CSV export, parse preamble metadata, and convert channels to SI units.

    Parameters
    ----------
    filepath : str | Path
        Path to the MoTeC CSV telemetry file (expects 18-line MoTeC preamble:
        lines 0-13 session metadata, line 14 channel names, line 15 units, data from line 18+).
    channels : Iterable[str], optional
        Channel names to load if present in the file. Defaults to ``CORE_CHANNELS``.
        ``Time`` is mandatory.

    Returns
    -------
    df : pd.DataFrame
        Time-sorted telemetry DataFrame (rows with NaNs dropped, index reset) in SI units:
        - ``Time`` [s]
        - ``SPEED`` [m/s] (converted from ``km/h``)
        - ``ROTY`` [rad/s] (converted from ``deg/s``)
        - ``STEERANGLE`` [rad] (converted from ``deg``)
        - ``G_LAT``, ``G_LON`` [m/s^2] (multiplied by ``9.80665`` if unit is ``g`` or peak <= 5.0)
        - ``TYRE_PRESS_*`` [psi], ``THROTTLE``/``BRAKE`` [%], ``SUS_TRAVEL_*`` [mm] (if requested)
    meta : TelemetryMetadata
        Parsed session metadata including ``venue``, ``vehicle``, ``sample_rate_hz``,
        ``duration_s``, ``beacon_markers_s``, and raw header ``units``.

    Notes
    -----
    - Sampling rate is read dynamically from the CSV preamble (typically **20 Hz** for Stage 2
      modeling or **200 Hz** for suspension/damper calibration).
    - ACC's MoTeC exporter frequently labels ``G_LAT`` / ``G_LON`` as ``"m/s2"`` while writing
      values in ``g``; this loader detects ``max(|val|) <= 5.0`` and rescales to ``m/s^2``.
    - ``G_LON`` in ACC MoTeC exports is integer-quantized (``{-2, -1, 0, 1} g``); use
      ``AX_SPEED`` from :func:`src.data.curation.curate_telemetry` for continuous longitudinal acceleration.
    - Rows are stably sorted by ``Time`` to guard against out-of-order CSV exports.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    meta, all_columns = _parse_preamble(path)
    requested = list(dict.fromkeys(channels))
    present_cols = [c for c in requested if c in all_columns]
    if "Time" not in present_cols:
        raise ValueError(f"Required 'Time' column missing in {path}. Found: {all_columns[:10]}")

    df = pd.read_csv(
        path,
        skiprows=18,
        header=None,
        names=all_columns,
        usecols=present_cols,
        low_memory=False,
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    df = _convert_to_si(df, meta.units)
    return df, meta
