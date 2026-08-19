# Per-episode task instruction + generalized episode metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the exporter vary the LeRobot `task` string per episode (read from each recording's `meta.json`, e.g. `"task"`) instead of one fixed dataset-wide string, and fix the existing `episode_metadata` mechanism to handle any JSON scalar type (not just floats) — both fully optional, defaulting to today's exact behavior when unset.

**Architecture:** Reuse the existing `meta.json`-reading path in `exporter.py` (`_load_episode_metadata`) for both features — extend it to also resolve a per-episode task string, and fix its value typing/missing-field default along the way. Thread the resolved value through `Episode.task` (new field, `episode_sampler.py`) into `LeRobotHead._sample_to_frame` (`heads/lerobot.py`), which already writes `frame["task"]` per sample — it just currently always reads `config.task_description` there instead of a per-episode value.

**Tech Stack:** Python 3.13, pydantic (`ExportConfig`), pytest.

## Global Constraints

- `task_field` defaults to `None`; `episode_metadata` defaults to `[]`. With both unset, export output must be byte-for-byte identical to current behavior — every task must preserve this.
- Per-episode, not per-frame: the resolved task string is constant across all frames of one episode (matches how LeRobot's `task`/`tasks.parquet`/`task_index` mechanism actually works — see `docs/superpowers/specs/2026-08-11-per-episode-task-and-metadata-design.md`).
- `episode_metadata` values are `Any` JSON scalar (str/float/int/bool) — no schema/type validation across episodes.
- Missing `meta.json` field (or no local `meta.json` at all, e.g. `catalog_url` export): warn and fall back — `task_field` falls back to `config.task_description`; `episode_metadata` fields fall back to `None` (not `0.0`). Never a hard failure.
- Out of scope: `heads/groot.py` (unaffected — it already gets per-episode tasks for free via LeRobot's `task_index`), and LeRobot's `language_persistent`/`language_events` schema (a separate, unrelated annotation mechanism).

---

## File Structure

- Modify: `src/nova_export/export/config.py` — add `ExportConfig.task_field`.
- Modify: `src/nova_export/export/episode_sampler.py` — add `Episode.task`; widen `Episode.extra_metadata`'s type.
- Modify: `src/nova_export/export/exporter.py` — generalize `_load_episode_metadata` (typing + task_field support); resolve `episode.task`/`episode.extra_metadata` in `export_recordings`.
- Modify: `src/nova_export/export/heads/lerobot.py` — `_sample_to_frame`/`write_episode` use the resolved per-episode task instead of the config constant.
- Modify: `tests/test_export.py` — new/updated tests for all of the above.
- Modify: `docs/export-guide.md` — document `task_field`; correct `episode_metadata`'s description.

---

### Task 1: Add `ExportConfig.task_field`

**Files:**
- Modify: `src/nova_export/export/config.py:156-176`
- Test: `tests/test_export.py` (new test in a `TestExportConfig`-style block, or alongside existing config-via-`ExportConfig(...)` tests — there's no dedicated `TestExportConfig` class yet; add one near the top of the file, after the imports/helpers, before `TestFrameCache`)

**Interfaces:**
- Produces: `ExportConfig.task_field: str | None` (default `None`) — consumed by Task 3.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_export.py` (a new class, placed after the helper functions around line 172 and before `class TestFrameCache:`):

```python
class TestExportConfigTaskField:
    """Tests for ExportConfig.task_field."""

    def test_task_field_defaults_to_none(self):
        config = ExportConfig(fps=15)
        assert config.task_field is None

    def test_task_field_can_be_set(self):
        config = ExportConfig(fps=15, task_field="task")
        assert config.task_field == "task"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestExportConfigTaskField -v`
Expected: FAIL — `ExportConfig` has no field `task_field` (pydantic raises on the second test since it's an unknown kwarg... actually pydantic's default `extra` behavior is to raise `ValidationError` for unknown fields, or the first test fails with `AttributeError: 'ExportConfig' object has no attribute 'task_field'`).

- [ ] **Step 3: Add the field**

In `src/nova_export/export/config.py`, right after the existing `task_description` field (currently lines 156-159):

```python
    task_description: str = Field(
        default="task",
        description="Task label written to the dataset",
    )

    task_field: str | None = Field(
        default=None,
        description=(
            "meta.json field name (e.g. 'task') holding this episode's "
            "natural-language task instruction. When set, each episode's "
            "LeRobot 'task' is read from its own meta.json instead of the "
            "fixed task_description, mirroring how LeRobot's task/task_index "
            "mechanism is meant to vary per episode. Falls back to "
            "task_description when the field is missing for a given episode, "
            "or when no local meta.json is available (e.g. exporting from "
            "catalog_url). Requires local rrd_paths export, like "
            "episode_metadata."
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestExportConfigTaskField -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nova_export/export/config.py tests/test_export.py
git commit -m "feat: add optional ExportConfig.task_field for per-episode task instructions"
```

---

### Task 2: Generalize `episode_metadata` beyond floats

**Files:**
- Modify: `src/nova_export/export/episode_sampler.py:66`
- Modify: `src/nova_export/export/exporter.py:203-231` (`_load_episode_metadata`) and `:491-503` (the `episode.extra_metadata` assignment inside `export_recordings`)
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces: `_resolve_extra_metadata(found: dict[str, Any], fields: list[str]) -> dict[str, Any]` — a new module-level function in `exporter.py`, consumed by `export_recordings` in this same task and left untouched by Task 3. `Episode.extra_metadata: dict[str, Any] | None` (was `dict[str, float] | None`).
- Note for the implementer: `export_recordings`'s metadata block currently builds `episode.extra_metadata` inline (`{f: found.get(f, 0.0) for f in config.episode_metadata}`) — that inline dict comprehension is *not* independently unit-testable (it's buried inside a large loop over live Rerun segments), which is exactly why the bug this task fixes (`0.0` instead of `None` for a missing field) has no direct test today. This task extracts that expression into `_resolve_extra_metadata`, a pure function, specifically so the fix has a real test that fails before the fix and passes after — not a test that merely documents the intended expression.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_export.py`, inside a new `TestResolveExtraMetadata` class (place it right before `class TestSourceValidation:`, since it tests another `exporter.py` private helper the same way that class does):

```python
class TestResolveExtraMetadata:
    """Tests for exporter._resolve_extra_metadata."""

    def test_present_fields_pass_through_any_type(self):
        from nova_export.export.exporter import _resolve_extra_metadata

        found = {"cube_x_mm": -324.15, "cube_color": "purple"}

        result = _resolve_extra_metadata(found, ["cube_x_mm", "cube_color"])

        assert result == {"cube_x_mm": -324.15, "cube_color": "purple"}

    def test_missing_field_defaults_to_none_not_zero(self):
        from nova_export.export.exporter import _resolve_extra_metadata

        found = {"cube_x_mm": 1.5}  # cube_color absent

        result = _resolve_extra_metadata(found, ["cube_x_mm", "cube_color"])

        assert result == {"cube_x_mm": 1.5, "cube_color": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestResolveExtraMetadata -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_extra_metadata'` — the function doesn't exist yet.

- [ ] **Step 3: Add `_resolve_extra_metadata` and fix the typing**

In `src/nova_export/export/episode_sampler.py`, change line 66:

```python
    extra_metadata: dict[str, float] | None = None
```

to:

```python
    extra_metadata: dict[str, Any] | None = None
```

(`Any` is already imported at the top of this file.)

In `src/nova_export/export/exporter.py`, add `from typing import Any` near the top (this file currently has no `typing` import at all — add it next to the other stdlib imports), then add this new function directly above `_load_episode_metadata` (currently at line 203):

```python
def _resolve_extra_metadata(found: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Build one episode's extra_metadata dict from its meta.json values.

    A field missing from `found` becomes None, not 0.0 — 0.0 was only
    correct by accident for numeric fields and actively wrong for a field
    like cube_color ("purple").
    """
    return {f: found.get(f) for f in fields}
```

Also update `_load_episode_metadata`'s return-type annotation only (its body is already type-agnostic — the bug was purely in the caller):

```python
def _load_episode_metadata(
    rrd_paths: list[Path] | None, fields: list[str]
) -> dict[str, dict[str, Any]]:
    """Read episode_metadata fields from each recording's sibling meta.json.

    Keyed by segment_id, which for local rrd_paths exports is exactly the
    recording's directory name (<dataset>/<recording_id>/recording.rrd) —
    the same recording_id the collector assigns and rerun uses as the
    segment ID, so no separate ID plumbing is needed. Values are whatever
    JSON scalar type meta.json holds (str/float/int/bool) — not float-only.
    """
    if not fields:
        return {}
    if not rrd_paths:
        logger.warning(
            "episode_metadata {} configured but exporting from catalog_url "
            "(no local meta.json available) — skipping",
            fields,
        )
        return {}

    result: dict[str, dict[str, Any]] = {}
    for rrd_path in rrd_paths:
        meta_path = rrd_path.parent / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        segment_id = rrd_path.parent.name
        result[segment_id] = {f: meta[f] for f in fields if f in meta}
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestResolveExtraMetadata -v`
Expected: PASS

- [ ] **Step 5: Wire `_resolve_extra_metadata` into `export_recordings`**

In `src/nova_export/export/exporter.py`, change the caller (currently lines 491-503) — only the last two lines of this block change (the log message text, and the final assignment now calling the new function):

```python
                if config.episode_metadata:
                    found = episode_metadata_by_segment.get(segment_id, {})
                    missing = [f for f in config.episode_metadata if f not in found]
                    if missing:
                        logger.warning(
                            "Episode {} ({}): meta.json missing {} — filled with null",
                            episode_id,
                            segment_id[:8],
                            missing,
                        )
                    episode.extra_metadata = _resolve_extra_metadata(
                        found, config.episode_metadata
                    )
```

- [ ] **Step 6: Run the full test file to verify nothing regressed**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py -v -k "not RealIntegration"`
Expected: PASS (the `-k "not RealIntegration"` skips the slow real-`.rrd` integration test, which needs real recording fixtures and isn't affected by this change anyway)

- [ ] **Step 7: Commit**

```bash
git add src/nova_export/export/episode_sampler.py src/nova_export/export/exporter.py tests/test_export.py
git commit -m "fix: generalize episode_metadata to any JSON scalar, default missing to null"
```

---

### Task 3: Resolve per-episode `task` in `exporter.py`

**Files:**
- Modify: `src/nova_export/export/episode_sampler.py:60-66` (`Episode` dataclass)
- Modify: `src/nova_export/export/exporter.py:203-231` (`_load_episode_metadata`, extended) and `:395-397` + the loop body around `:491-503`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `ExportConfig.task_field` (Task 1), `_resolve_extra_metadata` and `Episode.extra_metadata: dict[str, Any] | None` (Task 2).
- Produces: `Episode.task: str | None` (new field, default `None`) — consumed by Task 4. `_resolve_task(found: dict[str, Any], task_field: str | None, task_description: str) -> str` — a new module-level function in `exporter.py`, mirroring `_resolve_extra_metadata`'s shape, for the same reason (a directly-testable pure function instead of inline logic buried in the export loop). `_load_episode_metadata(rrd_paths, fields, task_field=None) -> dict[str, dict[str, Any]]` — the per-segment dict now also carries the task_field's raw value under its own key (same dict, no new return shape) when `task_field` is passed.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_export.py`, in a new `TestResolveTask` class placed right after `TestResolveExtraMetadata`:

```python
class TestResolveTask:
    """Tests for exporter._resolve_task."""

    def test_task_field_unset_uses_task_description(self):
        from nova_export.export.exporter import _resolve_task

        result = _resolve_task({}, None, "fallback_task")

        assert result == "fallback_task"

    def test_task_field_present_used_verbatim(self):
        from nova_export.export.exporter import _resolve_task

        found = {"task": "Pick the purple cube up."}

        result = _resolve_task(found, "task", "fallback_task")

        assert result == "Pick the purple cube up."

    def test_task_field_missing_from_found_falls_back(self):
        from nova_export.export.exporter import _resolve_task

        found = {"other_field": 1}  # no "task" key

        result = _resolve_task(found, "task", "fallback_task")

        assert result == "fallback_task"
```

Also add these two cases to `TestLoadEpisodeMetadata` (rename `TestLoadEpisodeMetadata` if it doesn't already exist under that name — it was not introduced in Task 2, since Task 2's tests live in `TestResolveExtraMetadata` instead; create `TestLoadEpisodeMetadata` fresh here, placed right after `TestResolveTask`):

```python
class TestLoadEpisodeMetadata:
    """Tests for exporter._load_episode_metadata."""

    def test_string_field_round_trips(self, tmp_path):
        from nova_export.export.exporter import _load_episode_metadata

        rec_dir = tmp_path / "04cb4f25d3ef"
        rec_dir.mkdir()
        (rec_dir / "meta.json").write_text(
            '{"cube_color": "purple", "cube_x_mm": -324.15}'
        )
        rrd_path = rec_dir / "recording.rrd"
        rrd_path.touch()

        result = _load_episode_metadata([rrd_path], ["cube_color", "cube_x_mm"])

        assert result["04cb4f25d3ef"]["cube_color"] == "purple"
        assert result["04cb4f25d3ef"]["cube_x_mm"] == -324.15

    def test_missing_field_simply_absent_from_result(self, tmp_path):
        from nova_export.export.exporter import _load_episode_metadata

        rec_dir = tmp_path / "rec01"
        rec_dir.mkdir()
        (rec_dir / "meta.json").write_text('{"cube_x_mm": 1.5}')
        rrd_path = rec_dir / "recording.rrd"
        rrd_path.touch()

        result = _load_episode_metadata([rrd_path], ["cube_x_mm", "cube_color"])

        assert result["rec01"] == {"cube_x_mm": 1.5}
        assert "cube_color" not in result["rec01"]

    def test_task_field_value_included_in_result(self, tmp_path):
        from nova_export.export.exporter import _load_episode_metadata

        rec_dir = tmp_path / "04cb4f25d3ef"
        rec_dir.mkdir()
        (rec_dir / "meta.json").write_text(
            '{"task": "Pick the purple cube up.", "cube_x_mm": 1.0}'
        )
        rrd_path = rec_dir / "recording.rrd"
        rrd_path.touch()

        result = _load_episode_metadata([rrd_path], ["cube_x_mm"], task_field="task")

        assert result["04cb4f25d3ef"]["task"] == "Pick the purple cube up."
        assert result["04cb4f25d3ef"]["cube_x_mm"] == 1.0

    def test_task_field_none_does_not_add_task_key(self, tmp_path):
        from nova_export.export.exporter import _load_episode_metadata

        rec_dir = tmp_path / "rec01"
        rec_dir.mkdir()
        (rec_dir / "meta.json").write_text('{"task": "unused", "cube_x_mm": 1.0}')
        rrd_path = rec_dir / "recording.rrd"
        rrd_path.touch()

        result = _load_episode_metadata([rrd_path], ["cube_x_mm"])  # task_field omitted

        assert result["rec01"] == {"cube_x_mm": 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestResolveTask tests/test_export.py::TestLoadEpisodeMetadata -v`
Expected: FAIL — `TestResolveTask` fails with `ImportError: cannot import name '_resolve_task'`; `TestLoadEpisodeMetadata::test_task_field_value_included_in_result` fails with `TypeError: _load_episode_metadata() got an unexpected keyword argument 'task_field'`. The other two `TestLoadEpisodeMetadata` cases (`test_string_field_round_trips`, `test_missing_field_simply_absent_from_result`) already pass against Task 2's code — that's expected, they're regression coverage for behavior Task 2 already delivered, not new-in-this-task assertions.

- [ ] **Step 3: Add `_resolve_task` and extend `_load_episode_metadata`**

In `src/nova_export/export/exporter.py`, add this function directly below `_resolve_extra_metadata`:

```python
def _resolve_task(
    found: dict[str, Any], task_field: str | None, task_description: str
) -> str:
    """Resolve one episode's task string.

    task_field's value from meta.json when set and present; task_description
    otherwise (unset task_field, or the field missing from this episode's
    meta.json) — the same fallback either way, so a per-recording gap in
    metadata degrades to today's dataset-wide constant rather than failing.
    """
    if not task_field:
        return task_description
    value = found.get(task_field)
    return task_description if value is None else str(value)
```

Then replace `_load_episode_metadata` (as left by Task 2) with:

```python
def _load_episode_metadata(
    rrd_paths: list[Path] | None,
    fields: list[str],
    task_field: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Read episode_metadata fields (and optionally task_field) from each
    recording's sibling meta.json.

    Keyed by segment_id, which for local rrd_paths exports is exactly the
    recording's directory name (<dataset>/<recording_id>/recording.rrd) —
    the same recording_id the collector assigns and rerun uses as the
    segment ID, so no separate ID plumbing is needed. Values are whatever
    JSON scalar type meta.json holds (str/float/int/bool) — not float-only.

    When task_field is set, its value (if present) is included in the
    per-segment dict under its own key, alongside the requested
    episode_metadata fields — one meta.json read serves both.
    """
    if not fields and not task_field:
        return {}
    if not rrd_paths:
        logger.warning(
            "episode_metadata {} / task_field {!r} configured but exporting "
            "from catalog_url (no local meta.json available) — skipping",
            fields,
            task_field,
        )
        return {}

    result: dict[str, dict[str, Any]] = {}
    for rrd_path in rrd_paths:
        meta_path = rrd_path.parent / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        segment_id = rrd_path.parent.name
        entry: dict[str, Any] = {f: meta[f] for f in fields if f in meta}
        if task_field and task_field in meta:
            entry[task_field] = meta[task_field]
        result[segment_id] = entry
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestResolveTask tests/test_export.py::TestLoadEpisodeMetadata -v`
Expected: PASS (all cases)

- [ ] **Step 5: Add `Episode.task` and wire resolution into `export_recordings`**

In `src/nova_export/export/episode_sampler.py`, change the `Episode` dataclass (currently lines 60-66) to:

```python
class Episode:
    """A complete episode with metadata and samples."""

    segment_id: str
    episode_index: int
    samples: list[Sample]
    extra_metadata: dict[str, Any] | None = None
    task: str | None = None
```

In `src/nova_export/export/exporter.py`, change the call site (currently lines 395-397):

```python
        episode_metadata_by_segment = _load_episode_metadata(
            rrd_paths, config.episode_metadata, config.task_field
        )
```

And change the loop body's metadata block (as left by Task 2, currently around lines 491-503) to also resolve `episode.task`, right after the `episode.extra_metadata` assignment — `found` moves out of the `if config.episode_metadata:` guard since both blocks below need it:

```python
                found = episode_metadata_by_segment.get(segment_id, {})

                if config.episode_metadata:
                    missing = [f for f in config.episode_metadata if f not in found]
                    if missing:
                        logger.warning(
                            "Episode {} ({}): meta.json missing {} — filled with null",
                            episode_id,
                            segment_id[:8],
                            missing,
                        )
                    episode.extra_metadata = _resolve_extra_metadata(
                        found, config.episode_metadata
                    )

                if config.task_field and config.task_field not in found:
                    logger.warning(
                        "Episode {} ({}): meta.json missing task_field {!r} "
                        "— falling back to task_description",
                        episode_id,
                        segment_id[:8],
                        config.task_field,
                    )
                episode.task = _resolve_task(
                    found, config.task_field, config.task_description
                )
```

- [ ] **Step 6: Run the full test file to verify nothing regressed**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py -v -k "not RealIntegration"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/nova_export/export/episode_sampler.py src/nova_export/export/exporter.py tests/test_export.py
git commit -m "feat: resolve per-episode task from meta.json via config.task_field"
```

---

### Task 4: Write the resolved task in `LeRobotHead`, update docs

**Files:**
- Modify: `src/nova_export/export/heads/lerobot.py:128-171` (`write_episode`) and `:230-255` (`_sample_to_frame`)
- Modify: `docs/export-guide.md` (config reference table)
- Test: `tests/test_export.py` (`TestLeRobotHead`)

**Interfaces:**
- Consumes: `Episode.task: str | None` (Task 3).
- Produces: nothing new downstream — this is the terminal consumer of `episode.task`.

- [ ] **Step 1: Write the failing test**

Add to `TestLeRobotHead` in `tests/test_export.py`, right after `test_write_episode` (currently ending at line 710):

```python
    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_write_episode_uses_per_episode_task(self, mock_dataset_cls):
        """When Episode.task is set, every frame's 'task' must use it —
        not the dataset-wide config.task_description."""
        mock_dataset = MagicMock()
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(fps=15, task_description="fallback_task")

        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")
            head.initialize({"action": {"dtype": "float32", "shape": (7,)}})

            episode = create_test_episode(num_samples=3)
            episode.task = "Pick the purple cube up."
            head.write_episode(episode)

            for call in mock_dataset.add_frame.call_args_list:
                frame = call.args[0]
                assert frame["task"] == "Pick the purple cube up."

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_write_episode_falls_back_to_task_description(self, mock_dataset_cls):
        """When Episode.task is unset (None), fall back to config.task_description —
        this is the byte-for-byte-identical-to-today path."""
        mock_dataset = MagicMock()
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(fps=15, task_description="fallback_task")

        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")
            head.initialize({"action": {"dtype": "float32", "shape": (7,)}})

            episode = create_test_episode(num_samples=3)  # episode.task defaults to None

            head.write_episode(episode)

            for call in mock_dataset.add_frame.call_args_list:
                frame = call.args[0]
                assert frame["task"] == "fallback_task"

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_different_episodes_can_have_different_tasks(self, mock_dataset_cls):
        """Two episodes with distinct task strings both write their own value —
        proves this is LeRobot's per-episode task mechanism, not a renamed
        per-dataset constant."""
        mock_dataset = MagicMock()
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(fps=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")
            head.initialize({"action": {"dtype": "float32", "shape": (7,)}})

            episode_a = create_test_episode(segment_id="a", num_samples=1)
            episode_a.task = "Task A"
            episode_b = create_test_episode(segment_id="b", num_samples=1)
            episode_b.task = "Task B"

            head.write_episode(episode_a)
            head.write_episode(episode_b)

            frame_a = mock_dataset.add_frame.call_args_list[0].args[0]
            frame_b = mock_dataset.add_frame.call_args_list[1].args[0]
            assert frame_a["task"] == "Task A"
            assert frame_b["task"] == "Task B"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestLeRobotHead::test_write_episode_uses_per_episode_task -v`
Expected: FAIL — `frame["task"]` is `"fallback_task"` (from `config.task_description`) instead of `"Pick the purple cube up."`, since `_sample_to_frame` doesn't look at `episode.task` yet.

- [ ] **Step 3: Thread the resolved task through `write_episode`/`_sample_to_frame`**

In `src/nova_export/export/heads/lerobot.py`, change `write_episode` (currently lines 128-171) — only the body between the logging call and the `try` block changes:

```python
    def write_episode(self, episode: Episode) -> bool:
        """Write an episode to the dataset.

        Args:
            episode: Episode to write.

        Returns:
            True if successfully written.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not initialized. Call initialize() first.")

        if not episode.samples:
            logger.warning("Skipping empty episode {}", episode.episode_index)
            return False

        logger.info(
            "Writing episode {}: {} frames, {:.2f}s duration",
            episode.episode_index,
            episode.num_frames,
            episode.duration_s,
        )

        if self.config.episode_metadata and episode.extra_metadata:
            # LeRobot assigns its own sequential episode_index (meta.total_episodes)
            # when save_episode() runs, which is NOT episode.episode_index (that's
            # the exporter's raw segment-loop counter, and diverges as soon as any
            # earlier segment is skipped). Key by the index LeRobot is about to use.
            self._episode_metadata_by_index[self._dataset.meta.total_episodes] = (
                episode.extra_metadata
            )

        task = episode.task if episode.task is not None else self.config.task_description

        try:
            for sample in tqdm(episode.samples, desc="Frames", leave=False):
                frame = self._sample_to_frame(sample, task)
                self._dataset.add_frame(frame)

            self._dataset.save_episode()
            self._update_counts(episode)
            return True

        except Exception as e:
            logger.error("Error writing episode {}: {}", episode.episode_index, e)
            return False
```

And `_sample_to_frame` (currently lines 230-255):

```python
    def _sample_to_frame(self, sample: Sample, task: str) -> dict[str, Any]:
        """Convert a Sample to a LeRobot frame dict.

        Args:
            sample: Sample to convert.
            task: Resolved task string for this sample's episode (either
                Episode.task, when set, or config.task_description).

        Returns:
            Frame dict for LeRobotDataset.add_frame().
        """
        frame: dict[str, Any] = {}

        # Action
        frame["action"] = sample.action

        # State
        if len(sample.state) > 0:
            frame["observation.state"] = sample.state

        # Task
        frame["task"] = task

        # Images
        for cam_name, img_array in sample.images.items():
            frame[f"observation.images.{cam_name}"] = img_array

        return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py::TestLeRobotHead -v`
Expected: PASS (all `TestLeRobotHead` tests, including the three new ones and the pre-existing `test_write_episode` — that one still passes since `create_test_episode` leaves `episode.task=None`, falling back to `config.task_description`, exactly as it wrote before this change)

- [ ] **Step 5: Run the full test file**

Run: `cd /home/sebi/ws/nova-data-cli && uv run pytest tests/test_export.py -v -k "not RealIntegration"`
Expected: PASS

- [ ] **Step 6: Update `docs/export-guide.md`**

In the config reference table, change the `task_description` row's wording slightly and add a `task_field` row right after it (find the row starting with `| \`task_description\`` in the table around line 69):

```markdown
| `task_description`       | string                      | `"task"`          | Fallback task label written to every frame when `task_field` is unset (or its meta.json field is missing for a given episode).                                                                |
| `task_field`              | string \| null               | `null`             | meta.json field name (e.g. `"task"`) holding each episode's own natural-language instruction — lets `task` vary per episode instead of being fixed dataset-wide. Falls back to `task_description`. Requires local export (same as `episode_metadata`).            |
```

Also fix the existing `episode_metadata` row's description (search for `episode_metadata` in the table) to drop any float-specific wording if present, or add a one-line clarification directly below the table:

```markdown
`episode_metadata` values may be any JSON scalar type (string, number, boolean) — not float-only. A field missing from a given episode's `meta.json` is filled with `null` in that episode's row.
```

- [ ] **Step 7: Commit**

```bash
git add src/nova_export/export/heads/lerobot.py tests/test_export.py docs/export-guide.md
git commit -m "feat: write per-episode task string in LeRobotHead, document task_field"
```

---

## Self-Review Notes

- **Spec coverage:** Data flow section (meta.json → episode.task → frame["task"], and meta.json → episode.extra_metadata → parquet columns) — Tasks 2-4. Error handling (missing field/no meta.json → fallback + warning, never a hard failure) — Task 3 Step 5. Testing section's five `_load_episode_metadata`-level cases and the LeRobotHead-level multi-task-string case — Tasks 2-4's test steps. Optionality requirement (both features default to today's exact behavior) — every task's fallback-path test (`test_write_episode_falls_back_to_task_description`, Task 2's unaffected-when-unset defaults).
- **Type consistency checked:** `Episode.extra_metadata: dict[str, Any] | None` (Task 2) and `Episode.task: str | None` (Task 3) match how `exporter.py` sets them (Task 3 Step 5) and how `heads/lerobot.py` reads them (Task 4 Step 3). `_load_episode_metadata`'s signature gains `task_field` in Task 3 without breaking Task 2's call sites (default `None` keeps the two-arg call from Task 2's own tests working).
- **No placeholders:** every step has literal code, not a description of code.
