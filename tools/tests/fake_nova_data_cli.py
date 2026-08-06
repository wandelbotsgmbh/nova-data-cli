#!/usr/bin/env python
"""Stand-in for `nova-data-cli`, used only by tools/tests/.

Reads the same --dataset/--config/--output args as the real CLI. --dataset is
a scratch dir of symlinks (one per claimed recording, named by recording_id —
same layout pipeline.sh's export_and_commit() builds for the real CLI). Each
symlinked recording dir must contain a `.behavior` file controlling what this
run does with it:

  success   (default if `.behavior` is missing) — writes one real episode
  sleep:N   — sleeps N seconds first, then behaves like `success`
  skip      — writes nothing for this id (counts as a skipped episode, exit 0)
  crash     — aborts the WHOLE invocation immediately (exit 1, no output dir
              contents at all) — mirrors the real CLI's SystemExit(1) path
              for a config/data problem that isn't a per-episode failure
  flaky:N   — behaves like `skip` for the first N invocations of this specific
              recording, then `success` from then on — distinguishes
              "transient, recovers on retry" from a permanently-bad recording
              (`skip` forever, which should end up quarantined instead).
              Attempt counts are tracked in $FAKE_CLI_STATE_DIR (default
              /tmp/fake_nova_data_cli_state), NOT next to `.behavior` — that
              path is reached through pipeline.sh's scratch symlink back into
              $WATCH_DIR, and writing there would touch the real watched
              directory's mtime on every retry, confusing local-mode's
              idle-detection (a test-harness concern, not something the real
              nova-data-cli would ever do — it only reads recordings).

Uses the real lerobot.datasets.lerobot_dataset.LeRobotDataset writer (with
use_videos=False, to skip ffmpeg — an image feature is exercised just as much
of the merge path as a video one for these tests) so the dataset this writes
is genuinely mergeable by the real `lerobot.datasets.aggregate.aggregate_datasets`,
not a hand-rolled mock of the schema.

--config's content is ignored; only its path needs to be passed through, to
match the real CLI's argv shape (pipeline.sh doesn't care what's in it).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def read_behavior(recording_dir: Path) -> str:
    f = recording_dir / ".behavior"
    return f.read_text().strip() if f.is_file() else "success"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    recording_dirs = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    behaviors = {p.name: read_behavior(p) for p in recording_dirs}

    if any(b == "crash" for b in behaviors.values()):
        print(f"fake_nova_data_cli: crash-tagged recording in batch {list(behaviors)}", file=sys.stderr)
        return 1

    successful, skipped, failed = [], [], []
    dataset = None
    for name, behavior in behaviors.items():
        if behavior.startswith("sleep:"):
            time.sleep(float(behavior.split(":", 1)[1]))
            behavior = "success"
        elif behavior.startswith("flaky:"):
            state_dir = Path(os.environ.get("FAKE_CLI_STATE_DIR", "/tmp/fake_nova_data_cli_state"))
            state_dir.mkdir(parents=True, exist_ok=True)
            counter_file = state_dir / f"{name}.attempts"
            attempts = int(counter_file.read_text()) + 1 if counter_file.is_file() else 1
            counter_file.write_text(str(attempts))
            threshold = int(behavior.split(":", 1)[1])
            behavior = "skip" if attempts <= threshold else "success"

        if behavior == "skip":
            skipped.append(name)
            continue
        if behavior != "success":
            failed.append(name)
            continue

        if dataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset.create(
                repo_id=f"test/{args.output.name}",
                fps=10,
                features={
                    "action": {"dtype": "float32", "shape": (2,), "names": None},
                    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
                },
                root=args.output,
                use_videos=False,
            )

        for _ in range(3):
            dataset.add_frame(
                {
                    "action": np.zeros(2, dtype=np.float32),
                    "observation.state": np.zeros(2, dtype=np.float32),
                    "task": "fake_task",
                }
            )
        dataset.save_episode()
        successful.append(name)

    if dataset is None:
        # Mirrors the real exporter: a batch where every segment is skipped
        # raises rather than producing a valid zero-episode dataset.
        print("fake_nova_data_cli: all claimed recordings skipped, nothing to export", file=sys.stderr)
        return 1

    dataset.finalize()

    (args.output / "export_summary.json").write_text(
        json.dumps(
            {
                "total_episodes_attempted": len(behaviors),
                "successful_episodes": len(successful),
                "skipped_episodes": len(skipped),
                "failed_episodes": len(failed),
                "successful_list": successful,
                "skipped_list": skipped,
                "failed_list": failed,
            }
        )
    )
    print(f"fake_nova_data_cli: {len(successful)} ok, {len(skipped)} skipped, {len(failed)} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
