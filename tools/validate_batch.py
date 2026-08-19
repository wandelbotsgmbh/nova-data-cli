#!/usr/bin/env python
"""Validate a pipeline.sh batch export before it's committed.

Checks export_summary.json against the batch's .claimed_ids. At batch size 1,
a skip/fail is attributable to that one recording; larger batches are only
checked in aggregate — see AGENT.md for why, and how the caller bisects.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Batch's tmp output dir")
    parser.add_argument("--claimed-ids", required=True, type=Path, help="Path to the batch's .claimed_ids file")
    args = parser.parse_args()

    claimed = [line.strip() for line in args.claimed_ids.read_text().splitlines() if line.strip()]
    if not claimed:
        print("No claimed IDs — nothing to validate", file=sys.stderr)
        return 1

    summary_path = args.output_dir / "export_summary.json"
    if not summary_path.is_file():
        print(f"No export_summary.json in {args.output_dir}", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text())
    successful = summary.get("successful_episodes", 0)
    skipped = summary.get("skipped_episodes", 0)
    failed = summary.get("failed_episodes", 0)

    if len(claimed) == 1:
        if successful > 0:
            print(f"OK: {claimed[0]} exported ({successful} episode(s))")
            return 0
        print(f"FAIL: {claimed[0]} produced no episodes (skipped={skipped}, failed={failed})", file=sys.stderr)
        return 1

    if skipped == 0 and failed == 0:
        print(f"OK: all {len(claimed)} claimed recordings exported cleanly")
        return 0

    print(
        f"AMBIGUOUS: batch of {len(claimed)} had {skipped} skipped / {failed} failed episodes, "
        "cannot attribute to a specific recording — bisect required. "
        f"Claimed: {', '.join(claimed)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
