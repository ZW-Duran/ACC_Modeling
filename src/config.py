"""Modular typed configuration dataclasses and YAML loader for ACC_Modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class AeroBaselineConfig:
    reference_area_m2: float | None = None
    base_cl: float | None = None
    base_cd: float | None = None
    base_aero_front_dist: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> AeroBaselineConfig:
        if not raw:
            return cls()
        return cls(
            reference_area_m2=raw.get("reference_area_m2"),
            base_cl=raw.get("base_cl"),
            base_cd=raw.get("base_cd"),
            base_aero_front_dist=raw.get("base_aero_front_dist"),
        )


@dataclass
class CarConfig:
    car_id: str = "generic_car"
    name: str = "Unknown Vehicle"
    vehicle_class: str = "GT3"
    drivetrain: str = "RWD"
    layout: str = "Mid-Engine"
    mass_kg: float = 1300.0
    front_weight_dist: float = 0.45
    wheelbase_m: float = 2.65
    track_width_front_m: float = 1.65
    track_width_rear_m: float = 1.65
    cg_height_m: float = 0.45
    tire_p_opt_psi: float = 27.0
    tire_sigma_p_psi: float = 1.5
    aero_baseline: AeroBaselineConfig = field(default_factory=AeroBaselineConfig)
    calibration_slots: dict[str, Any] = field(default_factory=dict)
    setup_bounds: dict[str, Any] = field(default_factory=dict)

    @property
    def vehicle_id(self) -> str:
        return self.car_id

    @property
    def cog_height_m(self) -> float:
        return self.cg_height_m

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CarConfig:
        """Build CarConfig from either nested `vehicle.yaml` schema or legacy flat dicts."""
        chassis = raw.get("chassis") if isinstance(raw.get("chassis"), dict) else raw
        aero_raw = raw.get("aero_baseline") if isinstance(raw.get("aero_baseline"), dict) else None
        cal_raw = raw.get("calibration_slots") if isinstance(raw.get("calibration_slots"), dict) else {}
        bounds_raw = raw.get("setup_bounds") if isinstance(raw.get("setup_bounds"), dict) else {}

        return cls(
            car_id=str(raw.get("vehicle_id", raw.get("car_id", "generic_car"))),
            name=str(raw.get("name", "Unknown Vehicle")),
            vehicle_class=str(raw.get("class", raw.get("vehicle_class", "GT3"))),
            drivetrain=str(raw.get("drivetrain", "RWD")),
            layout=str(raw.get("layout", "Mid-Engine")),
            mass_kg=float(chassis.get("mass_kg", 1300.0)),
            front_weight_dist=float(chassis.get("front_weight_dist", 0.45)),
            wheelbase_m=float(chassis.get("wheelbase_m", 2.65)),
            track_width_front_m=float(chassis.get("track_width_front_m", 1.65)),
            track_width_rear_m=float(chassis.get("track_width_rear_m", 1.65)),
            cg_height_m=float(chassis.get("cog_height_m", chassis.get("cg_height_m", 0.45))),
            tire_p_opt_psi=float(chassis.get("target_hot_pressure_psi", chassis.get("tire_p_opt_psi", 27.0))),
            tire_sigma_p_psi=float(chassis.get("tire_sigma_p_psi", 1.5)),
            aero_baseline=AeroBaselineConfig.from_dict(aero_raw),
            calibration_slots=dict(cal_raw),
            setup_bounds=dict(bounds_raw),
        )


@dataclass
class CornerRegion:
    name: str
    start_m: float
    end_m: float
    speed_class: Literal["low", "medium", "high"] = "medium"
    aero_sensitivity: Literal["low", "medium", "high"] = "medium"
    kerb_usage: Literal["none", "low", "high"] = "low"


@dataclass
class TrackConfig:
    track_id: str = "generic_track"
    name: str = "Unknown Circuit"
    length_m: float = 4657.0
    corners: list[CornerRegion] = field(default_factory=list)


@dataclass
class DataConfig:
    clean_csv: str = ""
    race_csv: str = ""
    clean_trim_mode: Literal["speed_threshold", "first_beacon", "none"] = "speed_threshold"
    clean_speed_threshold_kmh: float = 100.0
    race_trim_mode: Literal["first_beacon", "speed_threshold", "none"] = "first_beacon"


@dataclass
class CurationConfig:
    savgol_window: int = 5
    savgol_polyorder: int = 3
    p_opt_psi: float = 27.0
    sigma_p_psi: float = 1.5
    clean_trim_mode: Literal["speed_threshold", "first_beacon", "none"] = "speed_threshold"
    clean_speed_threshold_kmh: float = 100.0
    race_trim_mode: Literal["first_beacon", "speed_threshold", "none"] = "first_beacon"


@dataclass
class ModelingConfig:
    formulation: Literal["planar_force", "wheel_corner", "full_chassis"] = "planar_force"
    stlsq_threshold: float = 0.01
    stlsq_alpha: float = 0.05
    normalize_columns: bool = True


@dataclass
class StatsConfig:
    bootstrap_n_resamples: int = 500
    bootstrap_block_seconds: float = 2.0
    random_seed: int = 42
    z_anomaly_threshold: float = 5.0


@dataclass
class ExperimentConfig:
    experiment_name: str
    car: CarConfig = field(default_factory=CarConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    setup: dict[str, Any] = field(default_factory=dict)

    @property
    def vehicle(self) -> str:
        return self.car.name

    @property
    def circuit(self) -> str:
        return self.track.name

    @property
    def formulation(self) -> str:
        return self.modeling.formulation

    @property
    def circuit_corners(self) -> list[CornerRegion]:
        return self.track.corners


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_car_config(path: str | Path) -> CarConfig:
    """Load a standalone ``vehicle.yaml`` into :class:`CarConfig`."""
    return CarConfig.from_dict(_read_yaml(path))


def load_track_config(path: str | Path) -> TrackConfig:
    """Load a standalone track YAML file into :class:`TrackConfig`."""
    raw = _read_yaml(path)
    corners = [CornerRegion(**c) for c in raw.get("corners", [])]
    return TrackConfig(
        track_id=raw.get("track_id", Path(path).stem),
        name=raw.get("name", "Unknown Circuit"),
        length_m=float(raw.get("length_m", 4657.0)),
        corners=corners,
    )


def load_setup_file(path: str | Path) -> dict[str, Any]:
    """Load a garage setup YAML or candidate JSON file."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    return _read_yaml(p)


def _resolve_subpath(subpath: str | Path, base_dir: Path) -> Path:
    p = Path(subpath)
    if p.is_absolute() or p.exists():
        return p
    rel = base_dir / p
    if rel.exists():
        return rel
    # Walk up to repo root if experiment file is nested under configs/experiments/
    for parent in base_dir.parents:
        candidate = parent / p
        if candidate.exists():
            return candidate
    return p


def load_config(
    experiment_path: str | Path,
    car_path: str | Path | None = None,
    track_path: str | Path | None = None,
    curation_path: str | Path | None = None,
    modeling_path: str | Path | None = None,
    setup_path: str | Path | None = None,
) -> ExperimentConfig:
    """Load an experiment config composed from modular car, setup, track, curation, and modeling YAMLs."""
    exp_path = Path(experiment_path)
    base_dir = exp_path.resolve().parent
    raw = _read_yaml(exp_path)

    # Load car config (nested vehicle.yaml or inline dict)
    car_ref = car_path or raw.get("car_config")
    if car_ref:
        car_raw = _read_yaml(_resolve_subpath(car_ref, base_dir))
    elif "car" in raw:
        car_raw = raw["car"]
    else:
        car_raw = {"name": raw.get("vehicle", "Unknown Vehicle")}
    car_cfg = CarConfig.from_dict(car_raw)

    # Load optional per-car garage setup config
    setup_ref = setup_path or raw.get("setup_config")
    if setup_ref:
        setup_raw = _read_yaml(_resolve_subpath(setup_ref, base_dir))
    else:
        setup_raw = dict(raw.get("setup", {}))

    # Load track config
    track_ref = track_path or raw.get("track_config")
    if track_ref:
        track_raw = _read_yaml(_resolve_subpath(track_ref, base_dir))
    elif "track" in raw:
        track_raw = raw["track"]
    else:
        track_raw = {
            "name": raw.get("circuit", "Unknown Circuit"),
            "corners": raw.get("circuit_corners", []),
        }
    corners = [CornerRegion(**c) for c in track_raw.get("corners", [])]
    track_cfg = TrackConfig(
        track_id=track_raw.get("track_id", "generic_track"),
        name=track_raw.get("name", "Unknown Circuit"),
        length_m=float(track_raw.get("length_m", 4657.0)),
        corners=corners,
    )

    # Load curation config
    cur_ref = curation_path or raw.get("curation_config")
    cur_raw = _read_yaml(_resolve_subpath(cur_ref, base_dir)) if cur_ref else dict(raw.get("curation", {}))
    cur_raw.setdefault("p_opt_psi", car_cfg.tire_p_opt_psi)
    cur_raw.setdefault("sigma_p_psi", car_cfg.tire_sigma_p_psi)
    curation_cfg = CurationConfig(**cur_raw)

    # Load modeling + stats config
    mod_ref = modeling_path or raw.get("modeling_config")
    mod_raw = _read_yaml(_resolve_subpath(mod_ref, base_dir)) if mod_ref else dict(raw.get("modeling", {}))

    stats_raw = dict(raw.get("stats", {}))
    for key in ("bootstrap_n_resamples", "bootstrap_block_seconds", "random_seed", "z_anomaly_threshold"):
        if key in mod_raw and key not in stats_raw:
            stats_raw[key] = mod_raw.pop(key)
    stats_cfg = StatsConfig(**stats_raw)

    modeling_fields = {
        k: v
        for k, v in mod_raw.items()
        if k in ("formulation", "stlsq_threshold", "stlsq_alpha", "normalize_columns")
    }
    if "formulation" in raw and "formulation" not in modeling_fields:
        modeling_fields["formulation"] = raw["formulation"]
    modeling_cfg = ModelingConfig(**modeling_fields)

    # Data paths & trim overrides
    data_raw = dict(raw.get("data", {}))
    if "clean_csv" in raw:
        data_raw.setdefault("clean_csv", raw["clean_csv"])
    if "race_csv" in raw:
        data_raw.setdefault("race_csv", raw["race_csv"])
    data_raw.setdefault("clean_trim_mode", curation_cfg.clean_trim_mode)
    data_raw.setdefault("clean_speed_threshold_kmh", curation_cfg.clean_speed_threshold_kmh)
    data_raw.setdefault("race_trim_mode", curation_cfg.race_trim_mode)
    data_cfg = DataConfig(**data_raw)

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", exp_path.stem),
        car=car_cfg,
        track=track_cfg,
        data=data_cfg,
        curation=curation_cfg,
        modeling=modeling_cfg,
        stats=stats_cfg,
        setup=setup_raw,
    )
