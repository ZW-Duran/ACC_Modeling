"""Convert Assetto Corsa Competizione JSON setups for Audi R8 LMS GT3 Evo II to YAML.

Mapping verified against Barcelona/R8_A.json <-> barcelona_r8_A.yaml and
Barcelona/a1.json <-> barcelona_a1.yaml.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure standalone invocation works without an editable install
_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.data.acc_setup import AccSetupMapping, convert_acc_json_file, run_converter_cli

GT3_EVO2_MAPPING = AccSetupMapping(
    vehicle_id="audi_r8_gt3_evo2",
    tyre_pressure_base_psi=20.3,
    tyre_pressure_step_psi=0.1,
    toe_base_deg=-0.40,
    toe_step_deg=0.01,
    camber_front_base_deg=-4.0,
    camber_rear_base_deg=-3.5,
    camber_step_deg=0.1,
    caster_base_deg=9.0,
    caster_step_deg=5.5 / 24.0,  # 24 -> 14.5 deg, 14 -> 12.2 deg
    wheel_rate_front_base_nm=111000,
    wheel_rate_front_step_nm=21000,
    wheel_rate_rear_base_nm=143000,
    wheel_rate_rear_step_nm=10000,
    bumpstop_rate_base_n_mm=300,
    bumpstop_rate_step_n_mm=100,
    steer_ratio_base=12.0,
    steer_ratio_step=1.0,
    preload_base_nm=20.0,
    preload_step_nm=10.0,
    brake_bias_base_pct=50.0,
    brake_bias_step_pct=0.2,
    damper_slow_offset=0,
    damper_fast_offset=0,
    ride_height_rear_base_mm=54,
    ride_height_front_base_mm=54,
    fuel_offset_l=2,
)


def convert_acc_json_to_yaml_dict(json_path: str | Path, setup_name: str | None = None) -> dict:
    """Convert an Audi R8 LMS GT3 Evo II ACC JSON file into a setup dictionary."""
    return convert_acc_json_file(json_path, GT3_EVO2_MAPPING, setup_name=setup_name)


if __name__ == "__main__":
    run_converter_cli(
        GT3_EVO2_MAPPING,
        "Convert Audi R8 LMS GT3 Evo II ACC setup JSON into UI/SI YAML.",
    )
