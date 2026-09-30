#!/usr/bin/env python
"""Merge the batch_*/dataset dirs produced by tools/pipeline.sh into one dataset.

See AGENT.md for why this checks video-encoder compatibility itself rather
than relying on lerobot's aggregate_datasets (which ignores it).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

_VIDEO_INFO_KEYS_TO_CHECK = ("video.codec", "video.pix_format", "video.height", "video.width")

# GR00T batches (format="groot") get converted in place to v2.1 by
# tools/groot_lerobot_conversion, which preserves the original v3.0 export
# alongside as "dataset_v3.0". lerobot's aggregate_datasets only understands
# this CLI's pinned v3.0 schema, so for GR00T batches we must merge that
# preserved v3.0 copy, not the converted v2.1 "dataset" dir — then redo the
# GR00T conversion once on the merged result.
_GROOT_CONVERTER_DIR = Path(__file__).resolve().parent / "groot_lerobot_conversion"


def _dataset_dirs(batches_root: Path) -> list[Path]:
    dirs = []
    for p in sorted(batches_root.glob("batch_*")):
        v3_backup = p / "dataset_v3.0"
        plain = p / "dataset"
        if v3_backup.is_dir():
            dirs.append(v3_backup)
        elif plain.is_dir():
            dirs.append(plain)
    return dirs


def _convert_merged_to_groot(merged_dir: Path, modality_source: Path) -> None:
    """Re-run the GR00T v3.0->v2.1 conversion on the merged dataset.

    aggregate_datasets rebuilds meta/ from scratch, so modality.json must be
    seeded into the merged v3.0 dataset before conversion (it's identical
    across batches, so any one batch's copy works).
    """
    shutil.copy2(modality_source, merged_dir / "meta" / "modality.json")

    if shutil.which("uv") is None:
        raise SystemExit(
            "GR00T conversion needs `uv` on PATH. Convert the merged v3.0 dataset "
            f"manually:\n  uv run --project {_GROOT_CONVERTER_DIR} groot-convert {merged_dir}"
        )
    print(f"Re-running GR00T conversion on merged dataset {merged_dir}...")
    cmd = ["uv", "run", "--project", str(_GROOT_CONVERTER_DIR), "groot-convert", str(merged_dir)]
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(cmd, check=True, env=env)


def _load_info(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def _check_compatible(dataset_dirs: list[Path]) -> None:
    """Fail fast on fps/robot_type/video-encoder drift."""
    first_info = _load_info(dataset_dirs[0])
    first_fps = first_info.get("fps")
    first_robot_type = first_info.get("robot_type")
    first_features = first_info.get("features", {})

    problems = []
    for d in dataset_dirs[1:]:
        info = _load_info(d)
        if info.get("fps") != first_fps:
            problems.append(f"{d}: fps={info.get('fps')} != {first_fps}")
        if info.get("robot_type") != first_robot_type:
            problems.append(f"{d}: robot_type={info.get('robot_type')} != {first_robot_type}")

        features = info.get("features", {})
        for key, feat in first_features.items():
            if feat.get("dtype") != "video":
                continue
            other_feat = features.get(key, {})
            first_video_info = feat.get("info") or {}
            other_video_info = other_feat.get("info") or {}
            for video_key in _VIDEO_INFO_KEYS_TO_CHECK:
                if first_video_info.get(video_key) != other_video_info.get(video_key):
                    problems.append(
                        f"{d}: feature '{key}' {video_key}="
                        f"{other_video_info.get(video_key)!r} != {first_video_info.get(video_key)!r}"
                    )

    if problems:
        print(
            "Refusing to merge — batches disagree on schema/video encoding:\n  "
            + "\n  ".join(problems),
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-root", required=True, type=Path, help="Dir containing batch_* dirs")
    parser.add_argument("--output", required=True, type=Path, help="Output dir for the merged dataset")
    parser.add_argument("--repo-id", default="merged", help="Identifier for the merged dataset")
    args = parser.parse_args()

    dataset_dirs = _dataset_dirs(args.batches_root)
    if not dataset_dirs:
        print(f"No batch_*/dataset dirs found under {args.batches_root}", file=sys.stderr)
        sys.exit(1)

    if args.output.exists():
        print(f"Output dir already exists, refusing to overwrite: {args.output}", file=sys.stderr)
        sys.exit(1)

    _check_compatible(dataset_dirs)

    tmp_output = args.output.parent / f"{args.output.name}.tmp-{os.getpid()}"
    if tmp_output.exists():
        shutil.rmtree(tmp_output)

    is_groot = dataset_dirs[0].name == "dataset_v3.0"

    print(f"Merging {len(dataset_dirs)} batches into {args.output} (via {tmp_output})")
    try:
        aggregate_datasets(
            repo_ids=[d.parent.name for d in dataset_dirs],
            aggr_repo_id=args.repo_id,
            roots=dataset_dirs,
            aggr_root=tmp_output,
        )
        if is_groot:
            _convert_merged_to_groot(
                tmp_output, modality_source=dataset_dirs[0].parent / "dataset" / "meta" / "modality.json"
            )
    except BaseException:
        shutil.rmtree(tmp_output, ignore_errors=True)
        shutil.rmtree(f"{tmp_output}_v3.0", ignore_errors=True)
        raise

    os.rename(tmp_output, args.output)
    print("Merge complete.")


if __name__ == "__main__":
    main()
