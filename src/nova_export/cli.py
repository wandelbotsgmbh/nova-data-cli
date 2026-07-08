"""Local CLI to export .rrd recordings to a training dataset.

Spins up a temporary in-process Rerun server backed by the local recordings,
exports, then shuts it down — no long-running service required.

Usage:
    nova-data-cli \
        --recordings-dir ./recordings \
        --dataset my_dataset \
        --config examples/lerobot_export.json \
        --output ./exports/my_dataset

Recordings are expected under <recordings-dir>/<dataset>/<recording_id>/recording.rrd
(the layout the collector writes after a recording is stopped). --dataset may also
be a direct path, to either that dataset directory (exports every recording.rrd
found under it) or a single recording directory (exports just that one) — see
docs/export-guide.md#selecting-recordings-to-export for details.

Formats (set via the config's "format" field):
- "lerobot_v3": writes a LeRobot v3.0 dataset.
- "groot": writes LeRobot v3.0 + modality.json, then automatically converts it to
  the GR00T-compatible LeRobot v2.1 layout using the isolated converter subproject
  (requires `uv` and `ffmpeg`).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from nova_export.export import ExportConfig, export_recordings
from nova_export.logging_setup import configure_cli_logging, suppress_native_stderr

# The GR00T v3.0 → v2.1 converter runs in its own separately-pinned environment
# (older LeRobot, Python < 3.12), incompatible with this service's env. We locate
# it relative to the source tree and invoke it via `uv run --project`.
_GROOT_CONVERTER_DIR = Path(__file__).resolve().parents[2] / "tools" / "groot_lerobot_conversion"


def _convert_to_groot_v21(output_dir: Path) -> None:
    """Convert an exported LeRobot v3.0 dataset in place to GR00T-compatible v2.1.

    Shells out to the isolated converter subproject (see its README) so the user
    only runs a single `nova-data-cli` command. Requires `uv` on PATH; the
    converter itself requires `ffmpeg`.
    """
    tool_dir = Path(os.environ.get("GROOT_CONVERTER_DIR", _GROOT_CONVERTER_DIR))
    if shutil.which("uv") is None:
        raise SystemExit(
            "GR00T conversion needs `uv` on PATH to run the isolated converter env.\n"
            "Install uv, then convert the v3.0 export manually:\n"
            f"  uv run --project {tool_dir} groot-convert {output_dir}"
        )
    if not (tool_dir / "pyproject.toml").is_file():
        raise SystemExit(
            f"GR00T converter not found at {tool_dir}. Set GROOT_CONVERTER_DIR to its "
            "location."
        )

    logger.info("Converting export to GR00T-compatible LeRobot v2.1 (isolated env)...")
    cmd = ["uv", "run", "--project", str(tool_dir), "groot-convert", str(output_dir)]
    # Drop VIRTUAL_ENV so the nested `uv run` doesn't warn about it not matching
    # the converter subproject's own environment.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"GR00T conversion failed (exit {e.returncode}). The unconverted v3.0 "
            f"dataset is still at {output_dir}. See the command output above."
        ) from e
    logger.success("GR00T v2.1 dataset ready → {} (v3.0 preserved at {}_v3.0)", output_dir, output_dir)


def _resolve_dataset_dir(recordings_dir: Path, dataset: str) -> Path:
    """Resolve the dataset directory from --dataset (a name or a direct path).

    Accepts either a plain dataset name (resolved under --recordings-dir) or a
    path to the dataset directory itself, so both of these work:
        --recordings-dir ./recordings --dataset pick-and-place-demo
        --dataset ./recordings/pick-and-place-demo
    """
    as_path = Path(dataset)
    if as_path.is_dir():
        return as_path
    return recordings_dir / dataset


def _discover_rrds(dataset_dir: Path, recordings_dir: Path) -> list[Path]:
    """Find optimized recording.rrd files for a dataset (mirrors the catalog scanner).

    Handles both the multi-recording layout (``<dataset>/<recording_id>/recording.rrd``)
    and a directory that is itself a single recording (``<...>/recording.rrd``).
    """
    if not dataset_dir.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {dataset_dir}\n"
            "Pass --dataset as a name under --recordings-dir "
            f"(currently {recordings_dir}), or as a direct path to the dataset directory.\n"
            "Expected layout: <recordings-dir>/<dataset>/<recording_id>/recording.rrd"
        )
    # A directory that is itself a single recording.
    if (dataset_dir / "recording.rrd").is_file():
        return [dataset_dir / "recording.rrd"]
    rrds = sorted(dataset_dir.glob("*/recording.rrd"))
    if not rrds:
        raise SystemExit(
            f"No 'recording.rrd' files found under {dataset_dir}.\n"
            "A recording.rrd is written only after a recording is stopped "
            "(chunks/ alone are not enough). Stop the recording, then retry."
        )
    return rrds


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nova-data-cli",
        description="Export local .rrd recordings to a LeRobot / GR00T dataset.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path(os.environ.get("STORAGE_DIR", "./recordings")),
        help="Root recordings directory (default: $STORAGE_DIR or ./recordings)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset to export: either a name (a subdirectory under --recordings-dir) "
            "or a direct path to the dataset directory"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an export config JSON (e.g. examples/lerobot_export.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the exported dataset (must not already exist)",
    )
    parser.add_argument(
        "--catalog-url",
        default=None,
        help="Optional: use a running Rerun catalog server instead of local .rrd files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed debug output (decoder internals, library warnings)",
    )
    args = parser.parse_args()

    configure_cli_logging(verbose=args.verbose)

    config = ExportConfig.model_validate_json(args.config.read_text())

    if args.catalog_url:
        rrd_paths = None
        dataset_name = args.dataset
        logger.info("Using external catalog: {}", args.catalog_url)
    else:
        dataset_dir = _resolve_dataset_dir(args.recordings_dir, args.dataset)
        dataset_name = dataset_dir.name
        rrd_paths = _discover_rrds(dataset_dir, args.recordings_dir)
        logger.info("Found {} recording(s) for dataset '{}'", len(rrd_paths), dataset_name)

    def on_progress(current: int, total: int) -> None:
        logger.info("Episode {}/{}", current, total)

    try:
        with suppress_native_stderr(enabled=not args.verbose):
            result = export_recordings(
                output_dir=args.output,
                config=config,
                dataset_name=dataset_name,
                catalog_url=args.catalog_url,
                rrd_paths=rrd_paths,
                progress_callback=on_progress,
            )
    except (ValueError, RuntimeError) as e:
        # Config/data problems (e.g. a source name not in the recording) — show the
        # actionable message, not a stack trace. Use -v for the full traceback.
        if args.verbose:
            raise
        logger.error("Export failed:\n{}", e)
        raise SystemExit(1) from e

    logger.success(
        "Exported {} episodes ({} frames) → {}",
        result.num_episodes,
        result.num_frames,
        result.output_dir,
    )

    if config.format == "groot":
        _convert_to_groot_v21(args.output)


if __name__ == "__main__":
    sys.exit(main())
