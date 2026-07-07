# nova-export

A local command-line tool (`nova-data-cli`) that exports NOVA `.rrd` recordings
into robot-learning datasets. It resamples video and action/state streams to a
fixed FPS and writes them via pluggable **export heads**.

## Formats

- **`lerobot_v3`** — LeRobot v3.0 dataset (Parquet + MP4).
- **`groot`** — LeRobot v3.0 + `meta/modality.json`, then auto-converted to the
  GR00T-compatible LeRobot **v2.1** layout. See
  [`tools/groot_lerobot_conversion/`](tools/groot_lerobot_conversion/README.md).

## Install

```bash
uv sync
```

## Usage

```bash
uv run nova-data-cli \
    --dataset ./recordings/pick-and-place-demo \
    --config examples/lerobot_export.json \
    --output ./exports/my-dataset
```

`--dataset` is either a direct path to the dataset directory (as above) or a name
resolved under `--recordings-dir` (default `$STORAGE_DIR` or `./recordings`), i.e.
`--recordings-dir ./recordings --dataset pick-and-place-demo`. Recordings are
expected at `<recordings-dir>/<dataset>/<recording_id>/recording.rrd`.
For `groot`, add `"format": "groot"` to the config and the CLI runs the v2.1
conversion automatically (needs `uv` + `ffmpeg`).

## Config

A JSON file selecting the format, FPS, and which sources map to action / state /
cameras. Examples: [`examples/`](examples/). Schema:
[`ExportConfig`](src/nova_export/export/config.py).

## Tests

```bash
uv run --group dev pytest          # add --run-slow for RRD integration tests
```
