# GR00T LeRobot conversion

Converts a LeRobot **v3.0** dataset (what `nova-export` produces) into the
LeRobot **v2.1** layout that NVIDIA Isaac GR00T expects, preserving the
`meta/modality.json` descriptor written by the GR00T export head.

## Why this is a separate, pinned subproject

GR00T reads LeRobot **v2.1** (per-episode parquet + per-episode MP4 +
`episodes.jsonl` / `tasks.jsonl`). The `nova-data-cli` export writes LeRobot
**v3.0** via `lerobot>=0.6`, which has no v2.1 writer and no backward converter.

The only maintained v3.0→v2.1 converter is NVIDIA's
[`convert_v3_to_v2.py`](https://github.com/NVIDIA/Isaac-GR00T/tree/main/scripts/lerobot_conversion),
which pins an **older LeRobot commit** and **Python < 3.12**. That conflicts
with the export CLI's environment (Python ≥ 3.12) on both axes — you cannot
install both in one env. So this converter lives here as its own subproject
with its own pinned dependencies, run as a standalone CLI. It shares no
environment with the export CLI.

- `convert_v3_to_v2.py` — vendored **unmodified** from NVIDIA/Isaac-GR00T.
- `groot_convert.py` — thin wrapper: local-directory invocation + forwards
  `meta/modality.json` (which the upstream script would otherwise drop).

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH` (the converter splits videos with `ffmpeg -c copy` —
  lossless stream copy, no re-encode).

## Usage

Normally you don't invoke this directly. When the export config has
`"format": "groot"`, `nova-data-cli` runs the export **and** this conversion
automatically (it shells out to this pinned env for you):

```bash
nova-data-cli --dataset my-dataset --config groot_export.json --output ./exports/my-dataset
```

To run this converter standalone on an existing v3.0 export:

```bash
uv run --project export/tools/groot_lerobot_conversion \
    groot-convert ./exports/my-dataset
```

Either way, afterwards:

- `./exports/my-dataset` — the GR00T-ready **v2.1** dataset (incl.
  `meta/modality.json`).
- `./exports/my-dataset_v3.0` — the original v3.0 dataset, preserved.

The first `uv run` provisions the pinned environment (downloads the pinned
LeRobot commit and its deps); subsequent runs reuse it.
