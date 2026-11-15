"""Convert Assetto Corsa Competizione JSON setups for Audi R8 LMS GT4 to YAML.

Mapping verified against Barcelona/a1.json <-> barcelona_a1.yaml.
Note: slopes where only one reference setup was supplied (e.g., caster and front wheelRate
steps) use the inferred ACC step sizes and should be double-checked when a second GT4 setup is added.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure standalone invocation works without an editable install
_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sindy_piml.data.acc_setup import AccSetupMapping, convert_acc_json_file, run_converter_cli

GT4_MAPPING = AccSetupMapping(
    vehicle_id="audi_r8_gt4",
    tyre_pressure_base_psi=17.0,
    tyre_pressure_step_psi=0.1,
    toe_base_deg=-0.40,
    toe_step_deg=0.01,
    camber_front_base_deg=-4.5,
    camber_rear_base_deg=-3.5,
    camber_step_deg=0.1,
    caster_base_deg=8.7,
    caster_step_deg=0.1,
    wheel_rate_front_base_nm=145000,
    wheel_rate_front_step_nm=15000,
    wheel_rate_rear_base_nm=146000,
    wheel_rate_rear_step_nm=14000,
    bumpstop_rate_base_n_mm=300,
    bumpstop_rate_step_n_mm=100,
    steer_ratio_base=14.0,
    steer_ratio_step=1.0,
    preload_base_nm=10.0,
    preload_step_nm=10.0,
    brake_bias_base_pct=50.0,
    brake_bias_step_pct=0.2,
    damper_slow_offset=1,
    damper_fast_offset=0,
    ride_height_rear_base_mm=112,
    ride_height_front_base_mm=105,
    fuel_offset_l=2,
)


def convert_acc_json_to_yaml_dict(json_path: str | Path, setup_name: str | None = None) -> dict:
    """Convert an Audi R8 LMS GT4 ACC JSON file into a setup dictionary."""
    return convert_acc_json_file(json_path, GT4_MAPPING, setup_name=setup_name)


if __name__ == "__main__":
    run_converter_cli(
        GT4_MAPPING,
        "Convert Audi R8 LMS GT4 ACC setup JSON into UI/SI YAML.",
    )
