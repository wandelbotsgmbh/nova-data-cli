"""Export orchestration — high-level API for exporting recordings.

Entry point for the export pipeline:

1. Load recordings into Rerun catalog
2. Create EpisodeSampler to decode video and resample at target FPS
3. Create appropriate ExportHead for target format
4. Iterate episodes and write to dataset

Usage:
    from nova_export.export import export_recordings

    result = export_recordings(
        rrd_paths=[Path("recording.rrd")],
        output_dir=Path("output"),
        config=ExportConfig(fps=15, ...),
    )
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from pathlib import Path

import rerun as rr
from loguru import logger
from rerun.catalog import CatalogClient, DatasetEntry
from tqdm import tqdm

from nova_export.export.config import ExportConfig
from nova_export.export.episode_sampler import EpisodeSampler
from nova_export.export.heads.base import ExportResult


@contextlib.contextmanager
def _open_dataset(
    dataset_name: str,
    catalog_url: str | None,
    rrd_paths: list[Path] | None,
) -> Generator[DatasetEntry, None, None]:
    """Open a dataset for export.

    Either connects to an external Rerun catalog server (when `catalog_url` is set),
    or spins up a temporary in-process Rerun server backed by local `.rrd` files
    (when `rrd_paths` is given). The temporary server is shut down on exit.
    """
    if catalog_url:
        logger.info("Connecting to external catalog: {}", catalog_url)
        client = CatalogClient(catalog_url)
        dataset = client.get_dataset(name=dataset_name)
        yield dataset
    else:
        if not rrd_paths:
            raise ValueError(
                "Either catalog_url or rrd_paths must be provided to open a dataset."
            )
        logger.info(
            "Spinning up temporary local Rerun server for export ({} .rrd files)",
            len(rrd_paths),
        )
        rrd_str_paths = [str(p) for p in rrd_paths]
        with rr.server.Server(datasets={dataset_name: rrd_str_paths}) as server:
            client = server.client()
            dataset = client.get_dataset(name=dataset_name)
            yield dataset


def export_recordings(
    *,
    output_dir: Path,
    config: ExportConfig,
    dataset_name: str,
    catalog_url: str | None = None,
    rrd_paths: list[Path] | None = None,
    segment_ids: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    abort_callback: Callable[[], bool] | None = None,
) -> ExportResult:
    """Export recordings to a training dataset.

    Provide exactly one data source:
    - `catalog_url`: connect to a running Rerun catalog server, or
    - `rrd_paths`: local `.rrd` files; a temporary in-process server is spun up.

    Args:
        output_dir: Where to write the dataset.
        config: Export configuration (fps, columns, cameras, format).
        dataset_name: Name of the dataset (in the catalog, or assigned to the temp server).
        catalog_url: URL of an external Rerun catalog server.
        rrd_paths: Local `.rrd` recording files (used when catalog_url is unset).
        segment_ids: Optional subset of segment IDs to export.
        progress_callback: Optional callback(current, total) called after each episode.
        abort_callback: Optional callback that returns True if export should be aborted.

    Returns:
        ExportResult with output path and statistics.
    """
    if not catalog_url and not rrd_paths:
        raise ValueError("Provide either catalog_url or rrd_paths.")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    logger.info(
        "Starting export (v2, {}): dataset '{}' → {}",
        config.format,
        dataset_name,
        output_dir,
    )

    with _open_dataset(dataset_name, catalog_url, rrd_paths) as dataset:
        # Determine segments to export
        all_segment_ids = list(segment_ids) if segment_ids else dataset.segment_ids()
        if not all_segment_ids:
            raise ValueError("No segments found in the recordings.")

        logger.info("Found {} segments (episodes) to export", len(all_segment_ids))

        # Create the episode sampler (Layer 1: decode + resample)
        sampler = EpisodeSampler(
            config=config,
            dataset=dataset,
            segment_ids=all_segment_ids,
        )

        # Create the export head (Layer 2: format-specific writer)
        head = _create_export_head(config, output_dir)

        # Track episode results
        successful_episodes = []
        skipped_episodes = []
        failed_episodes = []
        first_episode = True
        episode_count = 0

        # Iterate over segment IDs directly to track all attempts
        for episode_id, segment_id in enumerate(tqdm(all_segment_ids, desc="Episodes")):
            # Check if abort was requested
            if abort_callback and abort_callback():
                logger.warning(
                    "Export aborted by user after {} episodes", episode_count
                )
                break

            try:
                # Process segment through sampler
                logger.info(
                    "Processing episode {} (segment {})", episode_id, segment_id[:8]
                )
                episode = sampler._process_segment(segment_id, episode_id)

                # Check if sampler returned an episode
                if episode is None:
                    reason = "Sampler returned None (likely no video frames or invalid time range)"
                    logger.warning(
                        "Episode {} ({}): {}", episode_id, segment_id[:8], reason
                    )
                    skipped_episodes.append(
                        {
                            "episode": episode_id,
                            "segment_id": segment_id[:8],
                            "reason": reason,
                        }
                    )
                    continue

                # Check if episode has samples
                if not episode.samples:
                    reason = "No samples in episode"
                    logger.warning(
                        "Episode {} ({}): {}", episode_id, segment_id[:8], reason
                    )
                    skipped_episodes.append(
                        {
                            "episode": episode_id,
                            "segment_id": segment_id[:8],
                            "reason": reason,
                        }
                    )
                    continue

                # Initialize head with features from first episode
                if first_episode:
                    features = head.infer_features(episode.samples[0])
                    head.initialize(features)
                    first_episode = False

                # Write episode
                success = head.write_episode(episode)
                if not success:
                    reason = "write_episode returned False"
                    logger.warning(
                        "Episode {} ({}): {}", episode_id, segment_id[:8], reason
                    )
                    skipped_episodes.append(
                        {
                            "episode": episode_id,
                            "segment_id": segment_id[:8],
                            "reason": reason,
                        }
                    )
                else:
                    episode_count += 1
                    logger.info(
                        "Episode {} ({}): Successfully exported ({} samples)",
                        episode_id,
                        segment_id[:8],
                        len(episode.samples),
                    )
                    successful_episodes.append(
                        {
                            "episode": episode_id,
                            "segment_id": segment_id[:8],
                            "samples": len(episode.samples),
                        }
                    )
                    if progress_callback:
                        progress_callback(episode_count, len(all_segment_ids))

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "Episode {} ({}): Failed with error: {}",
                    episode_id,
                    segment_id[:8],
                    error_msg,
                )
                failed_episodes.append(
                    {
                        "episode": episode_id,
                        "segment_id": segment_id[:8],
                        "error": error_msg,
                    }
                )

        # Finalize and return result
        if first_episode:
            # No episodes were successfully processed — head was never initialized
            reasons = []
            for skip in skipped_episodes:
                reasons.append(f"Episode {skip['episode']}: {skip['reason']}")
            for fail in failed_episodes:
                reasons.append(f"Episode {fail['episode']}: {fail['error']}")
            detail = "; ".join(reasons) if reasons else "unknown reason"
            raise RuntimeError(
                f"Export failed: all {len(all_segment_ids)} episodes were skipped or failed. {detail}"
            )

        result = head.finalize()

        # Log detailed summary
        logger.info("=" * 60)
        logger.info("Export Summary:")
        logger.info("  Total episodes attempted: {}", len(all_segment_ids))
        logger.info("  ✓ Successful: {} episodes", len(successful_episodes))
        logger.info("  ⊘ Skipped: {} episodes", len(skipped_episodes))
        logger.info("  ✗ Failed: {} episodes", len(failed_episodes))
        logger.info("  Total frames exported: {}", result.num_frames)

        if skipped_episodes:
            logger.warning("Skipped episodes details:")
            for skip in skipped_episodes:
                logger.warning(
                    "  - Episode {} ({}): {}",
                    skip["episode"],
                    skip["segment_id"],
                    skip["reason"],
                )

        if failed_episodes:
            logger.error("Failed episodes details:")
            for fail in failed_episodes:
                logger.error(
                    "  - Episode {} ({}): {}",
                    fail["episode"],
                    fail["segment_id"],
                    fail["error"],
                )

        logger.info("=" * 60)

        # Add metadata to result
        result.metadata = {
            "total_episodes_attempted": len(all_segment_ids),
            "successful_episodes": len(successful_episodes),
            "skipped_episodes": len(skipped_episodes),
            "failed_episodes": len(failed_episodes),
            "successful_list": successful_episodes,
            "skipped_list": skipped_episodes,
            "failed_list": failed_episodes,
        }

        logger.success(
            "Export complete: {} episodes, {} frames → {}",
            result.num_episodes,
            result.num_frames,
            output_dir,
        )

        return result


def _create_export_head(config: ExportConfig, output_dir: Path):
    """Create the appropriate export head for the configured format.

    Args:
        config: Export configuration.
        output_dir: Output directory.

    Returns:
        ExportHead instance.
    """
    if config.format == "lerobot_v3":
        from nova_export.export.heads.lerobot import LeRobotHead

        return LeRobotHead(config, output_dir)
    elif config.format == "groot":
        from nova_export.export.heads.groot import GrootHead

        return GrootHead(config, output_dir)
    else:
        raise ValueError(f"Unknown export format: {config.format}")
