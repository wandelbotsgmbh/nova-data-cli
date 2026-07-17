---
name: lerobot-export
description: Export a local NOVA .rrd recordings directory into a LeRobot v3 (or GR00T) dataset using nova-data-cli. Use whenever the user asks to export, convert, or turn a recordings/dataset directory into LeRobot/GR00T format, e.g. "export ~/data/<x> to lerobot", "convert this dataset for training", "run the export CLI on <recordings-dir>".
---

# LeRobot export via nova-data-cli

This repo's CLI (`nova-data-cli`) turns a directory of NOVA `.rrd` recordings into a
LeRobot v3.0 (or GR00T) dataset. The tricky part is never running the CLI — it's
writing a config that actually matches the sources present in the recordings you
were pointed at. Example configs under `examples/` are illustrative, not universal;
don't assume their source names apply to a new dataset.

## Step 1 — Inspect the actual recording layout and sources

Recordings live at `<recordings-dir>/<dataset>/<recording_id>/{recording.rrd,config.json,meta.json}`.
Before writing a config, read `config.json` (source list, per-source `name`) and
`meta.json` (per-source sample counts) from a couple of recordings under the target
path — do **not** guess source names from an existing example config:

```bash
ls <path-to-dataset>              # one dir per recording_id
cat <path-to-dataset>/<recording_id>/config.json   # authoritative source names
cat <path-to-dataset>/<recording_id>/meta.json     # task, duration, per-source sample counts
```

Spot-check a handful of recordings (not just the first) to confirm the source list
is consistent across the whole dataset — sample with a fixed seed if there are many:

```bash
python3 - <<'EOF'
import json, os, random
base = os.path.expanduser("<path-to-dataset>")
dirs = os.listdir(base)
random.seed(42)
for d in random.sample(dirs, min(6, len(dirs))):
    c = json.load(open(os.path.join(base, d, "config.json")))
    print(d, [s["name"] for s in c["sources"]])
EOF
```

If names differ between recordings, flag that to the user before proceeding —
don't silently paper over it.

## Step 2 — Write the export config

The config is `ExportConfig` (schema: `src/nova_export/export/config.py`, full field
reference: `docs/export-guide.md`). Key fields:

- `format`: `"lerobot_v3"` or `"groot"`.
- `fps`: fixed resample rate (default 15).
- `action` / `state`: lists of **recording source names**, concatenated in order
  into one vector each.
- `cameras`: list of `{"source": ...}`, optionally `width`/`height` to resize.
- `trimming`: `all_present` (default, no trimming), `signal_presence` (first/last
  non-null sample of one source), or `signal_change` (first/last detected motion —
  needs `source`, `threshold` ~0.01, `tail_ms` ~500). Use `signal_change` on a
  motion-bearing state source (e.g. joint positions) to cut idle lead-in/out.
- `task_description`, `dataset_id` (LeRobot `repo_id`).

**Critical gotcha:** a source name must not appear in _both_ `action` and `state`.
The exporter builds one DataFusion projection from `action_columns() + state_columns()`,
and a repeated column name fails the whole export with:

```
DataFusion error: Error during planning: Projections require unique expression
names but the expression "?table?./<source>:Scalars:scalars" ... have the same name
```

This isn't a partial failure — it kills every episode. If a signal (e.g. a gripper
channel) conceptually belongs in both action and state, pick one placement (action
is the common convention for a commanded/gripper signal) rather than listing it
twice.

Ask the user to confirm the action/state source mapping and trimming choice if it
isn't obvious from context — there's often no single "right" mapping from raw
source names to action/state, and guessing wrong means a full re-export.

Save the config under `examples/<something>.json` alongside the existing examples.

## Step 3 — Run the export

```bash
uv run nova-data-cli \
    --dataset <path-to-dataset-or-name> \
    --config examples/<your-config>.json \
    --output ./exports/<dataset-name>
```

- `--dataset` accepts either a direct path (containing `recording.rrd` directly, or
  a directory of `*/recording.rrd`) or a bare name resolved under `--recordings-dir`.
- `--output` **must not already exist** — `rm -rf` it first if you're re-running
  after fixing a config (only do this for output dirs you just created in this
  session, never an arbitrary user path without confirming).
- Runs can take minutes for large recording sets — launch as a background command
  and wait for it rather than blocking with a short timeout.

## Step 4 — Read the result, don't just check exit code

Tail the run output for `ERROR` and the final summary line. Two failure classes look
similar but mean different things:

- **A handful of episodes with `Sampler returned None (likely no video frames or
invalid time range)` / `No overlapping time range`** — normal attrition; some
  individual recordings just don't have overlapping streams in the trim window.
  Fine as long as most episodes succeed.
- **Every episode failing with the same DataFusion/config error** — a config bug
  (see the gotcha above), not a data problem. Fix the config and re-run rather than
  accepting a dataset with 0 episodes.

Report to the user: episode count exported vs. skipped/failed, and why, before
calling the export done.
