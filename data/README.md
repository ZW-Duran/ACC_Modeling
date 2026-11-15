# Telemetry Data Naming Convention

Raw MoTeC `.csv`, `.ld`, and `.ldx` files in this directory are gitignored (`data/**/*.csv`, `data/**/*.ld`, `data/**/*.ldx`).

To keep telemetry runs unambiguously tied to their circuit, chassis, and ACC in-game setup profile, name imported files as:

```text
<track>__<car>__<setupProfile>__<session>.csv
```

### Examples
- `barcelona_catalunya__audi_r8_gt3_evo2__r8_A__practice.csv`
- `barcelona_catalunya__audi_r8_gt3_evo2__r8_A__race.csv`
- `barcelona_catalunya__audi_r8_gt4__a1__practice.csv`
- `barcelona_catalunya__audi_r8_gt4__a1__race.csv`

Where:
- `<track>` matches `track_id` in `configs/tracks/<track>.yaml`
- `<car>` matches `vehicle_id` in `configs/cars/<car>/vehicle.yaml`
- `<setupProfile>` matches the ACC setup profile filename in `configs/cars/<car>/setups/<Track>/<setupProfile>.yaml`
- `<session>` is the session type (`practice`, `race`, `qualifying`, `bb_test`, etc.)
