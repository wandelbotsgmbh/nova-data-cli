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

**📖 See the [Export guide](docs/export-guide.md)** for a walkthrough of every
config field, the export formats, camera resizing, and how the trimming modes
choose episode boundaries (with figures).

## Live sync + parallel export (`tools/pipeline.sh`)

Pulls recordings while collection is still running and exports them in
parallel as they finish, instead of waiting for collection to end and then
exporting one at a time. Worker count and memory budget are computed from the
machine's own RAM/cores at startup. See [`tools/AGENT.md`](tools/AGENT.md) for
the full design (crash-safety, bisection, why it's shaped this way).

**Two machines** (collector elsewhere, export runs here) — needs passwordless
SSH to the collector (`ssh-copy-id`):

```bash
tools/pipeline.sh                    # default: --mode remote
```

Edit `REMOTE_HOST`/`REMOTE_DIRS`/`WATCH_DIR` at the top of the script, or
override per-run via `PIPELINE_REMOTE_HOST`, `PIPELINE_REMOTE_DIRS`, etc.
(every config value is a `PIPELINE_*` env var — see the top of the script).

**One machine** (collection already finished, or writing straight into a
local dir) — no network involved:

```bash
PIPELINE_MODE=local PIPELINE_WATCH_DIR=/path/to/recordings tools/pipeline.sh
```

Both modes end the same way: once nothing new has shown up for
`PIPELINE_IDLE_MINUTES` (or immediately, for a backlog that was never live),
it merges every batch into one dataset at `<EXPORT_ROOT>_merged`. Restarting
after a crash/kill is always safe — already-exported recordings are never
redone.

## Tests

```bash
uv run --group dev pytest          # add --run-slow for RRD integration tests
```
