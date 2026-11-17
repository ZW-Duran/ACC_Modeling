"""Convert Assetto Corsa Competizione (ACC) setup JSON files into human/SI setup YAMLs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AccSetupMapping:
    """Car-specific click-to-UI conversion table between ACC setup JSON indices and YAML values."""

    vehicle_id: str
    tyre_pressure_base_psi: float
    tyre_pressure_step_psi: float = 0.1
    toe_base_deg: float = -0.40
    toe_step_deg: float = 0.01
    camber_front_base_deg: float = -4.0
    camber_rear_base_deg: float = -3.5
    camber_step_deg: float = 0.1
    caster_base_deg: float = 9.0
    caster_step_deg: float = 5.5 / 24.0
    wheel_rate_front_base_nm: int = 111000
    wheel_rate_front_step_nm: int = 21000
    wheel_rate_rear_base_nm: int = 143000
    wheel_rate_rear_step_nm: int = 10000
    bumpstop_rate_base_n_mm: int = 300
    bumpstop_rate_step_n_mm: int = 100
    steer_ratio_base: float = 12.0
    steer_ratio_step: float = 1.0
    preload_base_nm: float = 20.0
    preload_step_nm: float = 10.0
    brake_bias_base_pct: float = 50.0
    brake_bias_step_pct: float = 0.2
    damper_slow_offset: int = 0
    damper_fast_offset: int = 0
    ride_height_rear_base_mm: int = 54
    ride_height_front_base_mm: int = 54
    fuel_offset_l: int = 2


def convert_acc_dict(
    raw: dict[str, Any],
    mapping: AccSetupMapping,
    setup_name: str = "setup",
) -> dict[str, Any]:
    """Translate a parsed ACC setup JSON payload into a ACC_Modeling setup dictionary."""
    basic = raw["basicSetup"]
    tyres = basic["tyres"]
    align = basic["alignment"]
    strat = basic["strategy"]

    adv = raw["advancedSetup"]
    mech = adv["mechanicalBalance"]
    damp = adv["dampers"]
    aero = adv["aeroBalance"]
    drive = adv["drivetrain"]

    # Alignment: [FL, FR, RL, RR]
    cold_pressure = [
        round(mapping.tyre_pressure_base_psi + mapping.tyre_pressure_step_psi * int(x), 1)
        for x in tyres["tyrePressure"]
    ]
    toe = [round(mapping.toe_base_deg + mapping.toe_step_deg * int(x), 2) for x in align["toe"]]
    camber = [
        round(
            (mapping.camber_front_base_deg if i < 2 else mapping.camber_rear_base_deg)
            + mapping.camber_step_deg * int(x),
            1,
        )
        for i, x in enumerate(align["camber"])
    ]
    caster = [
        round(mapping.caster_base_deg + mapping.caster_step_deg * int(align["casterLF"]), 1),
        round(mapping.caster_base_deg + mapping.caster_step_deg * int(align["casterRF"]), 1),
    ]

    # Suspension: wheelRate, bumpstops, ARBs
    wheel_rate = [
        (mapping.wheel_rate_front_base_nm + mapping.wheel_rate_front_step_nm * int(x))
        if i < 2
        else (mapping.wheel_rate_rear_base_nm + mapping.wheel_rate_rear_step_nm * int(x))
        for i, x in enumerate(mech["wheelRate"])
    ]
    bumpstop_rate = [
        mapping.bumpstop_rate_base_n_mm + mapping.bumpstop_rate_step_n_mm * int(x)
        for x in mech["bumpStopRateUp"]
    ]

    # ACC JSON brakeDuct is [Front, Rear]; YAML convention stores [Rear, Front]
    bd_front, bd_rear = int(aero["brakeDuct"][0]), int(aero["brakeDuct"][1])
    # ACC JSON rideHeight is [FL, FR, RL, RR]; YAML convention stores [Rear, Front]
    rh_rear = mapping.ride_height_rear_base_mm + int(aero["rideHeight"][2])
    rh_front = mapping.ride_height_front_base_mm + int(aero["rideHeight"][0])

    return {
        "setup_name": setup_name,
        "vehicle_id": mapping.vehicle_id,
        "tyres_and_alignment": {
            "cold_pressure_psi": cold_pressure,
            "toe_deg": toe,
            "camber_deg": camber,
            "caster_deg": caster,
        },
        "suspension": {
            "wheel_rate_n_m": wheel_rate,
            "bumpstop_rate_n_mm": bumpstop_rate,
            "bumpstop_range_clicks": [int(x) for x in mech["bumpStopWindow"]],
            "arb_clicks": [int(mech["aRBFront"]), int(mech["aRBRear"])],
        },
        "steering_ratio": round(mapping.steer_ratio_base + mapping.steer_ratio_step * int(align["steerRatio"]), 1),
        "drivetrain_and_brakes": {
            "preload_diff_nm": round(mapping.preload_base_nm + mapping.preload_step_nm * int(drive["preload"]), 1),
            "brake_bias_pct": round(mapping.brake_bias_base_pct + mapping.brake_bias_step_pct * int(mech["brakeBias"]), 1),
        },
        "dampers_clicks": {
            "bump_slow": [int(x) + mapping.damper_slow_offset for x in damp["bumpSlow"]],
            "bump_fast": [int(x) + mapping.damper_fast_offset for x in damp["bumpFast"]],
            "rebound_slow": [int(x) + mapping.damper_slow_offset for x in damp["reboundSlow"]],
            "rebound_fast": [int(x) + mapping.damper_fast_offset for x in damp["reboundFast"]],
        },
        "aero_clicks": {
            "rear_wing": int(aero["rearWing"]),
            "front_splitter": int(aero["splitter"]),
            "brake_ducts": [bd_rear, bd_front],
        },
        "ride_height_mm": [rh_rear, rh_front],
        "fuel": mapping.fuel_offset_l + int(strat["fuel"]),
    }


def convert_acc_json_file(
    json_path: str | Path,
    mapping: AccSetupMapping,
    setup_name: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load an ACC JSON file, convert it, and optionally write the resulting YAML."""
    src = Path(json_path)
    with open(src, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if setup_name is not None:
        name = setup_name
    elif src.parent.parent.name == "setups":
        name = f"{src.parent.name.lower()}_{src.stem}"
    else:
        name = src.stem
    converted = convert_acc_dict(raw, mapping, setup_name=name)

    if output_path is not None:
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            yaml.safe_dump(converted, f, sort_keys=False, default_flow_style=None)

    return converted


def run_converter_cli(mapping: AccSetupMapping, description: str) -> None:
    """Shared CLI runner for per-vehicle ACC setup conversion scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("json_path", type=Path, help="Input ACC setup .json file.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .yaml file path.")
    parser.add_argument("--name", type=str, default=None, help="Override setup_name field.")
    args = parser.parse_args()

    out = args.output
    data = convert_acc_json_file(args.json_path, mapping, setup_name=args.name, output_path=out)
    if out is None:
        print(yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
