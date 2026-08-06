# tools/pipeline.sh test suite

Extensive tests for `tools/pipeline.sh` covering every deployment shape it
supports. None of these touch real collection data
(`/home/sebi/ws/Data/raw_data/choreo2`, `/home/sebi/ws/Data/choreo2_export`)
— every scenario runs against a disposable `/tmp/pipeline_test_*` dir (and,
for the remote scenarios, a dedicated test-only subdir under
`/mnt/data/sebastian/pipeline_test_fixtures/` on the real workstation,
never `raw_datasets/`). `lib.sh`'s `require_test_path` refuses to run any
cleanup against a path that doesn't literally contain `pipeline_test`, as a
backstop against a typo ever pointing somewhere real.

## Running

```
tools/tests/run_all.sh          # everything, ~15-20 min total
bash tools/tests/scenario_local_backlog.sh   # any one scenario individually
```

Each scenario prints `PASS`/`FAIL` per assertion and a summary line; `run_all.sh`
exits nonzero if anything failed.

## How pipeline.sh is made testable

`pipeline.sh`'s config block reads every value from a `PIPELINE_*` env var
with the real production value as the default — setting no env vars gives
you exactly today's production behavior. Tests override `PIPELINE_MODE`,
`PIPELINE_WATCH_DIR`, `PIPELINE_EXPORT_ROOT`, `PIPELINE_REMOTE_HOST`/`_DIRS`,
`PIPELINE_CHUNK`, `PIPELINE_POLL_SECONDS`, `PIPELINE_IDLE_MINUTES`, and
`PIPELINE_EXPORT_CLI` (the nova-data-cli invocation itself — see below).
A `sizing` role was added to the role-dispatch `case` purely as a test hook:
`pipeline.sh sizing --mode local` prints the computed `WORKERS` count and exits,
letting `scenario_sizing.sh` check `compute_workers()` in isolation.

## Two tiers

**Tier 1 (most of the coverage): `fake_nova_data_cli.py`.** Running the real
`nova-data-cli` for every scenario/edge case would take too long (~50s/episode)
and most scenarios need to control failure/skip/timing precisely, which real
`.rrd` data can't do on demand. `PIPELINE_EXPORT_CLI` swaps it in for
`uv run nova-data-cli`. It drives the *real* `lerobot.datasets.lerobot_dataset.LeRobotDataset`
writer (with `use_videos=False` to skip ffmpeg) so its output is a genuinely
mergeable LeRobot dataset, not a hand-rolled mock of the schema — the real
`aggregate_datasets` runs for real in every scenario's merge step.

Each fixture recording dir gets a `.behavior` file controlling what the stub
does with it:
- `success` (default) — writes one real episode
- `sleep:N` — sleeps N seconds, then succeeds (for concurrency proofs)
- `skip` — contributes 0 episodes, counted as skipped (exit 0)
- `crash` — aborts the whole batch immediately (exit 1) — mirrors the real
  CLI's `SystemExit(1)` path for a config/data problem, not a per-episode failure
- `flaky:N` — skips for the first N attempts, then succeeds — proves a
  transient failure recovers via retry instead of being quarantined. State is
  tracked in `$FAKE_CLI_STATE_DIR` (default `/tmp/fake_nova_data_cli_state`),
  deliberately *not* next to `.behavior` — that path is reached through
  `pipeline.sh`'s scratch symlink back into `$WATCH_DIR`, and writing there
  would reset local mode's idle-detection clock on every retry (a test-harness
  artifact; the real CLI never writes back into a source recording).

**Tier 2: `scenario_real_smoke.sh`.** One real end-to-end run through the
actual `nova-data-cli` (no stub) — two of the smallest sample recordings
under `/home/sebi/ws/Data/5-pick-cube-sim-raw/`, copied (never symlinked or
moved) into an isolated test dir, run through the real CLI, real
`export_summary.json`, real `aggregate_datasets`. Skips itself (exit 0, not a
failure) if that sample dir or the real export config isn't present on the
machine running the suite.

## Scenarios

| Script | Covers |
|---|---|
| `scenario_sizing.sh` | `compute_workers()` scales with `MEM_TARGET_FRACTION`/`WORKER_MEM_ESTIMATE_MB`, clamps to `[1, ~0.8*nproc]` |
| `scenario_concurrency.sh` | Genuine wall-clock proof workers overlap (`CHUNK=1`, N slow fixtures, span measured from first→last commit, compared against a fully-serial floor) |
| `scenario_local_backlog.sh` | Local mode, whole backlog present upfront ("collection already finished, still want parallelism"). Bisection isolating a bad recording without quarantining batch-mates, 3-strikes quarantine, a flaky recording recovering via retry, no duplicate/dropped recordings, correct merged episode count, >1 worker actually used |
| `scenario_local_live.sh` | Local mode, recordings trickle in via a background "collector" — export starts *before* collection finishes (explicit timestamp comparison), `collection_done` doesn't fire prematurely |
| `scenario_remote_backlog.sh` | Real SSH/rsync against the real workstation host, into a dedicated test-only remote dir, whole backlog upfront |
| `scenario_remote_live.sh` | Real SSH/rsync, recordings trickle in remotely, same overlap/idle-detection proofs as the local live scenario |
| `scenario_crash_restart.sh` | SIGKILLs the whole supervisor process group mid-run (some recordings genuinely in-flight, claimed but uncommitted), restarts against the same `EXPORT_ROOT`, verifies the restart isn't blocked by the dead process's stale pgid, stale claims get swept and requeued, and everything ends up committed exactly once (no loss, no duplicate) across both runs |
| `scenario_real_smoke.sh` | Tier 2 — real `nova-data-cli`, no stub |

## Known timing characteristics (not bugs)

Local-mode scenarios take ~2-4 minutes each: `is_candidate()` requires a
`recording.rrd` to sit untouched for 60s before it's claimable (no
rsync-provided atomicity locally), and idle-detection needs `IDLE_MINUTES`
(shortened to 1 in tests) of sustained quiet on top of that. Remote-mode
scenarios are faster since that 60s floor doesn't apply. `scenario_concurrency.sh`
measures only the first→last-commit span, not total wall time, specifically
to avoid that floor contaminating the timing assertion.

## What isn't covered

- The `systemd-run`-unavailable fallback path in `export_and_commit()` — `systemd-run`
  is present on this machine, so that branch never executes here. Worth a
  manual check on a machine without it.
- A crash *during* the atomic commit rename itself (vs. mid-export, which
  `scenario_crash_restart.sh` does cover) isn't specifically targeted — `mv -T`
  is a single rename(2) syscall, effectively instantaneous, so there's no
  practical window to land a kill inside it deterministically.
