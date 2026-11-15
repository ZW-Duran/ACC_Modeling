import importlib.util
from pathlib import Path

import yaml


def _load_converter(script_path: Path):
    spec = importlib.util.spec_from_file_location(f"conv_{script_path.parent.name}", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_setup_fields_match(converted: dict, reference_yaml_path: Path) -> None:
    with open(reference_yaml_path, "r", encoding="utf-8") as f:
        ref = yaml.safe_load(f)

    for key in (
        "tyres_and_alignment",
        "suspension",
        "steering_ratio",
        "drivetrain_and_brakes",
        "dampers_clicks",
        "aero_clicks",
        "ride_height_mm",
        "fuel",
    ):
        assert converted[key] == ref[key], f"Mismatch on '{key}': {converted[key]} != {ref[key]}"


def test_gt3_acc_json_conversion_matches_ui_transcript():
    repo_root = Path(__file__).resolve().parents[1]
    gt3_dir = repo_root / "configs/cars/audi_r8_gt3_evo2"
    conv = _load_converter(gt3_dir / "convert_setup.py")

    r8_a = conv.convert_acc_json_to_yaml_dict(gt3_dir / "setups/Barcelona/R8_A.json")
    assert r8_a["vehicle_id"] == "audi_r8_gt3_evo2"
    assert r8_a["setup_name"] == "barcelona_R8_A"
    _assert_setup_fields_match(r8_a, gt3_dir / "setups/Barcelona/r8_A.yaml")


def test_gt4_acc_json_conversion_matches_ui_transcript():
    repo_root = Path(__file__).resolve().parents[1]
    gt4_dir = repo_root / "configs/cars/audi_r8_gt4"
    conv = _load_converter(gt4_dir / "convert_setup.py")

    a1 = conv.convert_acc_json_to_yaml_dict(gt4_dir / "setups/Barcelona/a1.json")
    assert a1["vehicle_id"] == "audi_r8_gt4"
    assert a1["setup_name"] == "barcelona_a1"
    _assert_setup_fields_match(a1, gt4_dir / "setups/Barcelona/a1.yaml")
