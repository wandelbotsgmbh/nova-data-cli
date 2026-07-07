"""GR00T conversion CLI — turn a local LeRobot v3.0 export into GR00T v2.1.

This is the thin, project-specific wrapper around NVIDIA's vendored
``convert_v3_to_v2.py``. It exists because that upstream script is oriented
toward Hugging Face repo ids and rebuilds ``meta/`` from scratch — which drops
the ``modality.json`` descriptor that ``nova-export --format groot`` writes.

This wrapper:
1. Accepts a plain local dataset directory (the ``output_dir`` produced by the
   export service) instead of a ``--repo-id`` / ``--root`` pair.
2. Runs the in-place v3.0 → v2.1 conversion (the original tree is preserved
   alongside with a ``_v3.0`` suffix).
3. Forwards ``meta/modality.json`` from the preserved v3.0 tree into the
   converted v2.1 dataset, so the result is directly loadable by GR00T.

It runs in this subproject's own pinned environment (old LeRobot, Python
< 3.12) — see this directory's ``pyproject.toml`` and ``README.md``. It is
deliberately decoupled from the ``nova-export`` service env, which pins a newer,
incompatible LeRobot.

Usage:
    uv run --project export/tools/groot_lerobot_conversion \\
        groot-convert /path/to/exported/dataset
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from convert_v3_to_v2 import convert_dataset

V30_SUFFIX = "_v3.0"
MODALITY_FILENAME = "modality.json"


def _forward_modality(dataset_dir: Path) -> None:
    """Copy modality.json from the preserved v3.0 backup into the v2.1 dataset.

    The upstream converter renames the original tree to ``<name>_v3.0`` and
    writes the v2.1 dataset in its place, rebuilding ``meta/`` from scratch. The
    GR00T ``modality.json`` written by the export head therefore only survives in
    the backup, so we forward it into the converted dataset.
    """
    backup_modality = dataset_dir.parent / f"{dataset_dir.name}{V30_SUFFIX}" / "meta" / MODALITY_FILENAME
    if not backup_modality.exists():
        logging.warning(
            "No %s found in the v3.0 backup (%s); the converted dataset will lack "
            "the GR00T modality descriptor. Was this exported with --format groot?",
            MODALITY_FILENAME,
            backup_modality,
        )
        return

    dest = dataset_dir / "meta" / MODALITY_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_modality, dest)
    logging.info("Forwarded GR00T %s → %s", MODALITY_FILENAME, dest)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Convert a local LeRobot v3.0 export (as produced by "
            "`nova-export --format groot`) into GR00T-compatible LeRobot v2.1, "
            "preserving meta/modality.json."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the exported LeRobot v3.0 dataset directory to convert in place.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Passed through to the upstream converter (re-download/overwrite).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not (dataset_dir / "meta" / "info.json").exists():
        parser.error(f"Not a LeRobot dataset (no meta/info.json): {dataset_dir}")

    # The upstream converter computes root as `root / repo_id`; splitting the
    # local path into parent + name makes it operate on this exact directory.
    convert_dataset(
        repo_id=dataset_dir.name,
        root=str(dataset_dir.parent),
        force_conversion=args.force_conversion,
    )

    _forward_modality(dataset_dir)
    logging.info(
        "Done. GR00T v2.1 dataset at %s (original v3.0 preserved at %s%s).",
        dataset_dir,
        dataset_dir,
        V30_SUFFIX,
    )


if __name__ == "__main__":
    sys.exit(main())
