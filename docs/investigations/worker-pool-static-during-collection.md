# Symptom: many workers running, but most idle for the whole collection phase

## Observed

During a long-running `tools/pipeline.sh` run, `ps aux | grep nova-data-cli`
showed a large number of workers up, but only a handful actively logging
`[worker] committed batch_...` lines — the rest sat silent, doing nothing,
for most of the run.

## Why it happens

`role_supervisor` calls `top_up_workers` exactly once, then blocks:

```bash
top_up_workers
wait "$acquire_pid" || true
```

`top_up_workers` sizes the pool from `compute_workers()`, which is a pure
function of **system memory/cores at the instant it's called** — it has no
idea how many recordings are actually claimable yet. At the moment collection
starts, `list_candidates` may already show a nontrivial backlog (e.g. a
`--mode remote` run resuming against a host with data already sitting there),
so this first call can spawn the machine's *entire* memory-supported worker
count immediately — before there's any evidence that candidates will keep
arriving at a matching rate.

`wait "$acquire_pid"` then blocks the supervisor's own control flow until the
**entire acquisition phase finishes** (which can be hours, for a live
collection). No further `top_up_workers` call happens until then — the
drain-loop's periodic top-up (every 60s) only starts *after* `COLLECTION_DONE`
is set. So whatever pool size the very first call happened to produce is what
runs, unmonitored, for the whole collection window, regardless of how the
real backlog behaves afterward (e.g. if it trickles in far slower than
`workers × CHUNK` can consume, or if more memory frees up later as batches
commit).

The code comment directly above `top_up_workers` claims "re-evaluated on
every call... so the worker count tracks memory headroom as it opens up" —
true of the *function*, but that guarantee only holds once collection is
already done; during collection, the function is simply never called again
to exercise it.

## Effect

Most workers spawned at that first call end up polling `list_candidates`
every 30s and finding nothing (no log line on a non-final empty scan — see
`role_worker`'s `empty_scans` loop — so this is invisible unless you're
watching `ps aux`), while a handful of workers that won the claim race keep
grinding through their batches. Throughput looks fine at a glance (batches do
keep committing), but most of the machine's provisioned worker capacity sits
unused for the run.

## Fix

Two changes, both in `tools/pipeline.sh` `role_supervisor`/`top_up_workers`:

1. Keep calling `top_up_workers` on the same 60s cadence during acquisition
   too, not just after `COLLECTION_DONE`:

   ```bash
   top_up_workers
   while kill -0 "$acquire_pid" 2>/dev/null; do
     sleep 60
     top_up_workers
   done
   wait "$acquire_pid" || true
   ```

2. `top_up_workers` now caps the new-worker count against the actual
   unclaimed backlog (`list_candidates | wc -l`) on every call, not only once
   `COLLECTION_DONE` is set. This cap already existed for the post-collection
   drain phase; it was deliberately *not* applied during collection because a
   single `list_candidates` read coming back empty (e.g. `is_candidate`'s
   60s-untouched freshness gate momentarily reading zero) would otherwise
   permanently block the very first spawn under the old one-shot-call
   structure. Change (1) removes that risk: a transient zero reading now just
   costs one 60s poll, not the rest of the run, so the cap can safely apply
   throughout.

Together these mean the pool actually scales with the live backlog and
memory headroom as collection proceeds, instead of being frozen at whatever
the single startup snapshot allowed.
