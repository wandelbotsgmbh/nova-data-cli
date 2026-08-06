# `pipeline.sh` design notes

This documents *why* `tools/pipeline.sh` (+ `validate_batch.py`, `merge_batches.py`)
is built the way it is. For usage, see the main [README](../README.md). This
file is for whoever next has to change this script and needs the reasoning,
not just the code.

## The problem

`nova-data-cli` has no incremental/append mode — every `--output` must be a
fresh directory — and a single invocation is single-threaded and CPU-light
(~1-1.7 cores, ~50s/episode observed). Collecting ~1000 episodes and exporting
them sequentially after collection finishes wastes hours: pulling and
exporting can overlap, and multiple exports can run concurrently on idle
cores. `pipeline.sh` does both, then merges the results into one dataset.

## Why a claim/commit protocol instead of static work partitioning

An earlier, simpler design statically split the recording list across N
workers up front. That doesn't hold up once new recordings can arrive mid-run
(the live-feed case) — a static split would need recomputing on every arrival,
which is the same problem in a different shape. Instead, workers dynamically
claim work from a shared pool:

- **Claiming** is `mkdir claimed/<id>` — atomic on any POSIX filesystem
  (including NFS, unlike `open(O_EXCL)`), so no separate lock is needed to
  prevent two workers claiming the same recording.
- **"Done" is derived, not tracked.** There's no separate done-marker
  directory. A batch's claimed IDs are written into `.claimed_ids` *inside*
  its own tmp output dir *before* exporting, so the single atomic `mv` that
  commits the batch to its final path is simultaneously the durable record of
  every ID it contains. `rebuild_done()` just reads `.claimed_ids` out of
  every existing `batch_*` dir. This collapses what would otherwise be two
  separate operations (commit, then mark-done) into one atomic one — and that
  matters: an earlier version *did* do them separately (commit, then loop
  `touch done/<id>`), and a crash between the two left a committed-but-
  unmarked batch, which got silently re-exported and duplicated on retry. One
  atomic operation can't have that window.

## The race this still has, and how it's closed

`rebuild_done()` is a point-in-time snapshot. If it's read at the top of a
worker's loop, then *before* that worker's `mkdir` lands, another worker
commits and releases a claim on one of the same IDs, the snapshot is stale —
the ID looks unclaimed (it is) and undone (it isn't, but the snapshot doesn't
know that yet). The `mkdir` would then succeed and re-export something already
committed. This was reproduced directly (via temporary tracing) during
development: worker A committed `rec05/06/07` and released their claims 10ms
before worker B's claim scan reached them.

The fix is one fresh `rebuild_done()` call *after* claiming, not before:
`mv` (commit) always fully completes before `rmdir` (release) for a given ID,
so the moment a claim becomes available, any commit for that ID is already on
disk. And once a worker holds a claim, nobody else can commit that ID (commit
requires a claim to run at all) — so a single recheck immediately after
claiming can never go stale again. This is why the recheck is a *quality*
distinct from "check more often": one recheck at the right point is
permanently sufficient, not just less likely to race.

## Bisection instead of batch-wide quarantine

`nova-data-cli` exits 0 even when it silently skips or fails individual
episodes within a batch — exit code alone doesn't mean "all N claimed
recordings are in the output." `validate_batch.py` checks
`export_summary.json` against `.claimed_ids`, but that file is keyed by
*segment*, not recording (a single `.rrd` can yield multiple segments, and the
exporter has no per-segment source tag) — so a skip/fail inside a multi-ID
batch can't be attributed to a specific recording from that file alone.

Rather than quarantine (or blindly retry) the whole batch — which would
needlessly punish healthy batch-mates for one bad recording — a batch that
fails validation is bisected: split in half, each half retried independently,
recursing until it narrows to size 1. At size 1, a skip/fail is unambiguous
("the only recording in this invocation didn't produce an episode"), so the
per-ID failure counter and 3-strikes quarantine only apply there. Counting
failures at every bisection level would let one real failure rack up several
"strikes" on the way down and quarantine a perfectly good recording that
happened to share unlucky batches.

The failure counter itself is a single-byte atomic append
(`printf x >> failed/<id>`, count = file size) recorded *at claim time*, not
after a failure — so a hard kill/OOM of the worker mid-export still counts as
an attempt, rather than letting a recording that keeps crashing the worker
retry forever without ever reaching the quarantine threshold.

## Crash safety in general

- **Single-instance lock**: the supervisor holds an `flock` for its entire
  life. Because of that, on startup, any leftover `claimed/`/`.tmp/` entries
  are provably abandoned (nothing else can legitimately be running) — except
  a killed run's process group might still have orphaned children alive even
  though the lock is free again; that's checked separately via a recorded
  pgid before sweeping.
- **Process group teardown**: the supervisor `setsid`s itself once at startup
  (becoming its own group leader) so a single `kill -- -$$` in its shutdown
  trap reaches every acquire/worker/`nova-data-cli`/ffmpeg descendant, not
  just its direct children.
- **Restart is always safe**: nothing needs manual cleanup after a crash.
  Committed batches stay committed (derived done-state), abandoned claims are
  swept, in-flight work is simply redone (at most `CHUNK` recordings' worth
  per worker, since claim is held for the whole bisection tree of one claim
  round).

## Acquisition modes

`--mode remote|local` share every line of the claim/export/validate/commit/
merge machinery — the only thing that differs is how new `recording.rrd`
files show up in `$WATCH_DIR` and how "collection is finished" is detected:

- **remote**: `rsync`/`ssh` on a poll loop; idle-detected via the remote
  host's own max file mtime being unchanged for `IDLE_MINUTES`. `rsync` exit
  codes 23/24 ("partial transfer"/"some files vanished") are expected — the
  source is being actively written — and don't abort the loop. A failed or
  timed-out `ssh` idle-check is treated as *inconclusive*, not as evidence of
  idleness (a network blip must never cause an early merge) and not as fatal
  (a network blip must never abort the whole pipeline either).
- **local**: no network at all — idle-detected via a local `find` on
  `$WATCH_DIR`'s own mtimes. Since there's no rsync-provided atomicity for a
  locally-written file (rsync's `--partial` semantics meant a file only
  appears at its final name once fully transferred), local mode additionally
  requires `recording.rrd` to be untouched for 60s before treating it as
  finished writing, as a substitute completeness signal.

This is also why **switching a finished remote collection to `--mode local`**
before going fully offline is the correct move, not a workaround: remote
mode's "safe to merge" signal fundamentally requires reaching the remote host
to confirm nothing's still arriving, so it can never fire without network —
whereas local mode's signal is purely local file-mtime watching and works
fully offline, correctly, once nothing new can possibly arrive.

## Resource budgeting

Worker count and per-worker memory cap are computed at startup from
`/proc/meminfo`/`nproc`, not hardcoded, so this doesn't silently misbehave on
a different machine. `MEM_TARGET_FRACTION` bounds total worker memory as a
fraction of system RAM (default 0.55 — deliberately conservative, since the
per-worker estimate is a rough baseline, not a guarantee, and the fraction is
computed against *total* memory, not what's actually free right now). Each
worker's `nova-data-cli` runs inside a `systemd-run --scope` cgroup with a
hard `MemoryMax` and `MemorySwapMax=0` — this is what actually prevents the
failure mode observed during development, where a single oversized batch
caused an OOM-kill of an internal subprocess (`rerun`) that left a truncated,
silently-corrupt output directory. A cgroup-enforced kill is a clean, cheap,
retryable failure; letting the kernel's OOM-killer pick an arbitrary victim
under memory pressure is not. A live watchdog additionally pauses claiming
new work when `MemAvailable` drops near what one worker needs, independent of
the startup sizing — this reacts to whatever else is running on the machine
right now, not just what the pipeline itself is doing.

`nice -n 10`/`ionice -c2 -n7` on the export command is the mechanism that
keeps this from monopolizing CPU/disk even without an active watchdog for
those resources — it tells the kernel to prefer any other process, so the
pipeline only consumes spare capacity.

## Merge

`merge_batches.py` merges via `lerobot.datasets.aggregate.aggregate_datasets`,
which has two gaps this script covers:

1. Its own metadata check (`validate_all_metadata`) compares fps/robot_type/
   features, but `features_equal_for_merge` *strips* video-encoder info
   (codec/pix_fmt/resolution) before that comparison — so encoder drift
   between batches passes validation and only surfaces deep inside the
   expensive video-concatenation copy, potentially hours in. This script
   checks those fields itself, up front, before starting.
2. It has no atomic-output guarantee — a killed merge leaves a partial,
   multi-GB output directory that can't just be resumed (its own append logic
   assumes a from-scratch destination). `merge_batches.py` builds into a
   `<output>.tmp-<pid>` directory and `os.rename`s it into place only on
   success, matching the same commit pattern used for individual batches.

The merge itself runs exactly once, after the acquisition process and every
worker have drained (checked by relaunching the worker pool if a pass finds
leftover candidates — a straggler bisection retry can land right as the last
worker exits — until a pass finds nothing left; there's no cap on this, since
remaining work only ever shrinks).
