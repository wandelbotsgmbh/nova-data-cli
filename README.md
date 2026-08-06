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

## Live sync + export (`tools/sync_loop.sh`)

Pulls recordings from a remote machine while collection is still running there,
and exports them in the background as each one completes — instead of waiting
for collection to finish before exporting anything.

- **Two machines (collector + this one):** requires passwordless SSH to the
  remote host (`ssh-copy-id`), since the script polls it every `POLL_SECONDS`
  via `rsync`/`ssh`. Edit `REMOTE_HOST`, `REMOTE_DIRS`, and `LOCAL_DEST` at the
  top of the script first.
- **Same machine:** SSH isn't needed if collection and export run on one box —
  point `REMOTE_DIRS`/`LOCAL_DEST` at local paths and swap `sync_once`'s
  `rsync` for a local copy (or skip syncing and export straight from the
  collection dir). Not built yet; the script currently assumes a remote host.
- The script stops polling once the remote dir has been idle for
  `IDLE_MINUTES`, does one final sync + export pass, then merges all batches
  into one LeRobot dataset via `tools/merge_batches.py` (nova-data-cli itself
  has no incremental/append mode, so each batch is a separate `--output` dir
  until merged).

```bash
tools/sync_loop.sh
```

## Tests

```bash
uv run --group dev pytest          # add --run-slow for RRD integration tests
```
