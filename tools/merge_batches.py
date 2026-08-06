#!/usr/bin/env python
"""Merge the batch_* LeRobot dataset dirs produced by sync_loop.sh into one dataset."""

import argparse
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-root", required=True, type=Path, help="Dir containing batch_* dataset dirs")
    parser.add_argument("--output", required=True, type=Path, help="Output dir for the merged dataset")
    parser.add_argument("--repo-id", default="merged", help="Identifier for the merged dataset")
    args = parser.parse_args()

    batches = sorted(p for p in args.batches_root.glob("batch_*") if p.is_dir())
    if not batches:
        print(f"No batch_* dirs found under {args.batches_root}", file=sys.stderr)
        sys.exit(1)

    if args.output.exists():
        print(f"Output dir already exists, refusing to overwrite: {args.output}", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(batches)} batches into {args.output}")
    aggregate_datasets(
        repo_ids=[b.name for b in batches],
        aggr_repo_id=args.repo_id,
        roots=batches,
        aggr_root=args.output,
    )
    print("Merge complete.")


if __name__ == "__main__":
    main()
