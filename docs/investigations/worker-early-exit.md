# Symptom: worker exits early, permanently down one worker

## Observed

During a `tools/pipeline.sh` run exporting the `choreo2` dataset
(`pick_and_place_sim_20260805_191245`, 1001 recordings, `--mode local`,
started 2026-08-06 ~17:07), worker `w3` logged:

```
[w3] [17:17:58] [worker] committed batch_20260806_171301_w3_1147218_10206 (8 recording(s))
[w3] [17:17:59] [worker] no candidates and collection done, exiting
```

...and never restarted. From then on, only `w0`, `w1`, `w2` were running
(confirmed via `ps aux | grep nova-data-cli`), each still committing batches
every ~5 minutes.

## Why it doesn't add up

At 17:17:59:
- ~160 episodes committed (21 batches × 8, from the `w0-w3` logs)
- At most ~32 in flight (4 workers × `CHUNK=8`)
- That leaves **~800 unclaimed candidates** still sitting in
  `/home/sebi/ws/Data/raw_data/choreo2/pick_and_place_sim_20260805_191245`

`collection_done` was legitimately set at 17:07:56 (the earlier remote-mode
acquire pass had already finished its final rsync at 17:01:36, so all 1001
recordings were genuinely present on disk by the time the local-mode watcher
started). So the "collection is done" half of the exit condition is correct.

The "no candidates" half is not plausible given ~800 unclaimed recordings
should have been sitting right there for `list_candidates()` to find.

## Effect

- One of four workers permanently disappears mid-run, with no log line
  indicating an error — just the same message a worker prints on genuine,
  correct completion.
- The remaining three workers keep making progress, so nothing looks wrong
  at a glance (batches keep committing), but overall throughput drops to
  75% for the rest of the run.
- Nothing in the pipeline's own signals (logs, exit code, `.pipeline/`
  state) distinguishes this from the expected end-of-run shutdown.

## Reproduction notes

Not reproduced in isolation — several attempts to trigger the same premature
`set -e`-style abort or scan failure in `list_candidates`/`is_candidate`
(including under a synthetic 50-recording watch dir with the exact same
functions) came back clean, so the underlying trigger is genuinely
load-dependent, not a deterministic logic bug in the scan itself.

## Fix

`role_worker`'s exit condition trusted a *single* empty scan as proof there
was no more work, the moment `COLLECTION_DONE` was also true — with no
tolerance for a transient glitch in that one scan (e.g. a `find` fork failing
under the exact system load described above). `role_supervisor`'s drain loop
had the same shape and a worse consequence (a false-empty scan there would
merge before everything was actually exported, not just drop a worker).

Both now require repeated confirmation before trusting "nothing left" — 3
consecutive empty scans for a worker to exit, 2 consecutive clean passes for
the supervisor to proceed to merge — the same pattern `role_acquire`'s
idle-detection already used and this code didn't. See `tools/pipeline.sh`
`role_worker`/`role_supervisor` and `tools/AGENT.md`.
