# Per-episode language instructions + generalized episode metadata

## Problem

The exporter needs to support VLA training datasets, where each episode carries
its own natural-language task instruction (e.g. "Pick the purple cube up from
the left of the table and place it onto the yellow square target position."),
alongside per-episode debug metadata (e.g. where the cube actually started:
`cube_x_mm`, `cube_y_mm`, `cube_z_mm`, `cube_yaw_rad`, `cube_color`).

Today:

- `config.task_description` is a single fixed string written to *every* frame
  of *every* episode (`heads/lerobot.py:_sample_to_frame`) — there is no way
  to vary it per episode.
- `config.episode_metadata` (a list of `meta.json` field names) already reads
  per-episode values and writes them as extra columns on
  `meta/episodes/*.parquet` — but `exporter.py:_load_episode_metadata` types
  values as `dict[str, float]` and defaults a missing field to `0.0`. Real
  `meta.json` data includes non-numeric fields (`cube_color: "purple"`),
  which this silently mishandles.

Both mechanisms must stay **optional**: a plain imitation-learning dataset
with no task and no extra metadata must export exactly as it does today —
nothing here is a new requirement.

## Non-goals

- No per-frame-varying task (only per-episode, matching how LeRobot's
  `task`/`task_index` mechanism actually works).
- No schema/typing for `episode_metadata` fields beyond "whatever JSON scalar
  is in `meta.json`" (str / float / int / bool) — no validation that a field
  is consistently typed across episodes.
- No change to GR00T's own code — it already gets per-episode task strings
  for free via LeRobot's `task_index`, which its `modality.json` references.

## Design

### 1. Per-episode task instruction

Add one new optional field to `ExportConfig` (`config.py`):

```python
task_field: str | None = Field(
    default=None,
    description=(
        "meta.json field name (e.g. 'task') holding this episode's "
        "natural-language task instruction. When set, each episode's LeRobot "
        "'task' is read from its own meta.json instead of the fixed "
        "task_description. Falls back to task_description when the field is "
        "missing for a given episode, or when no local meta.json is "
        "available (e.g. exporting from catalog_url). Requires local "
        "rrd_paths export, like episode_metadata."
    ),
)
```

Default `None` — behavior is byte-for-byte identical to today (one constant
`task_description` for the whole dataset) unless a user opts in.

**Loading**: `exporter.py`'s existing `_load_episode_metadata` already opens
each recording's sibling `meta.json` once per recording. Extend it to also
pull `task_field`'s value in that same read (avoid opening the file twice),
returning it alongside the metadata dict rather than adding a second loader.

**Threading through**: `Episode` (in `episode_sampler.py`) gets one new
optional attribute, `task: str | None`, set in `exporter.py` next to where
`episode.extra_metadata` is already set today — `episode.task` is the
resolved per-episode string (meta.json value if present, else
`config.task_description`, with a warning on fallback — mirroring the
existing missing-field warning for `episode_metadata`).

**Writing**: `heads/lerobot.py:_sample_to_frame` takes the resolved task
string as a parameter (from the enclosing `episode.task` in `write_episode`)
instead of always reading `self.config.task_description` directly. LeRobot
natively supports a per-episode-varying `task` column (that's what its
`tasks`/`task_index` table exists for) — no changes needed to
`initialize()`/`finalize()`.

### 2. Generalize `episode_metadata`

In `exporter.py`:

- `_load_episode_metadata`'s return type becomes
  `dict[str, dict[str, Any]]` (was `dict[str, dict[str, float]]`) — the
  underlying read (`meta[f]`) already preserves whatever JSON type is
  present; only the type annotation was wrong.
- `episode.extra_metadata = {f: found.get(f, 0.0) for f in ...}` →
  `{f: found.get(f) for f in ...}` (default `None`, not `0.0`). Update the
  existing "filled with 0.0" warning log to say "filled with null".

No change needed in `heads/lerobot.py:_write_episode_metadata_columns` —
pandas/pyarrow already handle a `None`-containing or string-typed column
into Parquet without modification.

## Data flow (updated)

```
meta.json (per recording, optional)
   │
   ├─ task_field value ──────────► episode.task ──► frame["task"] (per sample)
   │                                  (falls back to config.task_description)
   │
   └─ episode_metadata values ───► episode.extra_metadata ──► meta/episodes/*.parquet
                                      (any JSON scalar; missing → null)
```

Neither path does anything when its config field is unset/empty — a bare
imitation-learning dataset with no `meta.json` metadata at all exports
exactly as today.

## Error handling

- Missing `meta.json` entirely (no local rrd_paths, i.e. `catalog_url`
  export): both `task_field` and `episode_metadata` are skipped with a
  warning (existing behavior for `episode_metadata`, extended to
  `task_field`).
- `task_field` missing from one recording's `meta.json` while present in
  others: that episode falls back to `config.task_description`, with a
  per-episode warning — never a hard failure (matches how a missing
  `episode_metadata` field is already handled: warn and fill, not abort).

## Compatibility with standard LeRobot / other datasets

Verified against the installed `lerobot` package, not just assumed:

- `frame["task"]` → `meta/tasks.parquet` + `task_index` is the mechanism
  every standard LeRobot dataset uses for language conditioning, and what
  `aggregate_datasets`'s schema check (`validate_all_metadata` /
  `features_equal_for_merge`) and every stock policy config (ACT, diffusion,
  pi0, SmolVLA) actually read. Using it (rather than inventing a parallel
  column) is what makes this dataset minglable with datasets from other
  sources for training, not just internally consistent.
- This `lerobot` version also has a separate, newer `language_persistent`/
  `language_events` schema (structured subtask/plan/memory/VQA annotations
  with roles and timestamps) — but it's populated by a standalone offline
  annotation tool (`lerobot_annotate` / `steerable_pipeline`), not by
  exporters, and essentially no external dataset will have it. Out of scope
  here; revisit only if a target policy specifically requires it.
- `episode_metadata`'s extra `meta/episodes/*.parquet` columns never
  participate in `aggregate_datasets`'s validation (which only compares
  frame-level `features`), so they can't affect mixing with other datasets
  for training — they're purely additive/inert to any standard tooling.

**Cross-checked against real published community datasets, not just the
local package:**

- [DROID](https://github.com/google-deepmind/open_x_embodiment), a major
  Open X-Embodiment dataset, stores its primary instruction as a per-episode
  `language_instruction` string feeding the same `task`/`tasks.jsonl`/
  `task_index` mechanism this design uses — confirming it as the real-world
  standard for the *portable* instruction.
- [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026),
  a large published LeRobot-format dataset, layers custom per-episode
  annotations (subtask instructions, object bounding boxes) in
  `meta/info.json` *alongside*, not replacing, the standard `tasks.jsonl`
  mechanism — explicitly to stay compatible with standard training
  pipelines. Same shape as this design: one portable `task` string plus
  optional non-standard extra metadata that doesn't affect portability.

## Testing

- `tests/test_export.py`: unit tests for `_load_episode_metadata` covering
  (a) a string field (`cube_color`) round-tripping without corruption, (b) a
  missing field defaulting to `None` (not `0.0`), (c) `task_field` present →
  used verbatim, (d) `task_field` absent for one episode → falls back to
  `task_description` with a warning, (e) neither field configured → output
  identical to current behavior (regression guard for the optionality
  requirement).
- `tests/test_export.py` (LeRobotHead-level): an episode written with a
  distinct `task_field` value produces that string in the frame's `task`,
  and multiple episodes with different task strings both land in the
  dataset's task table (proves LeRobot's per-episode task mechanism is being
  used correctly, not just a per-dataset constant renamed).
