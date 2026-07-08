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

_SCALARS_SUFFIX = ":Scalars:scalars"
_VIDEO_SUFFIX = ":VideoStream:sample"


def _sources_for_suffix(column_names: list[str], suffix: str) -> list[str]:
    """Extract the user-facing source names for columns with a given suffix.

    e.g. "/joint_positions:Scalars:scalars" -> "joint_positions".
    """
    return sorted(
        name[1 : -len(suffix)]
        for name in column_names
        if name.startswith("/") and name.endswith(suffix)
    )


def _find_missing_sources(
    config: ExportConfig,
    *,
    timelines: list[str],
    signals: list[str],
    cameras: list[str],
) -> list[str]:
    """Return human-readable lines for every configured source that is absent.

    Pure (no I/O) so it can be unit-tested. Empty list means the config's sources
    all resolve against the recording.
    """
    timeline_set, signal_set, camera_set = set(timelines), set(signals), set(cameras)
    problems: list[str] = []

    if config.index_column not in timeline_set:
        problems.append(
            f"  index_column '{config.index_column}' not found — "
            f"available timelines: {', '.join(timelines) or '(none)'}"
        )

    missing_signals = [
        s for s in (*config.action, *config.state) if s not in signal_set
    ]
    if config.trimming.source and config.trimming.source not in signal_set:
        missing_signals.append(config.trimming.source)
    if missing_signals:
        problems.append(
            f"  signal sources {sorted(set(missing_signals))} not found "
            f"(used by action / state / trimming) — "
            f"available signals: {', '.join(signals) or '(none)'}"
        )

    missing_cameras = [c.source for c in config.cameras if c.source not in camera_set]
    if missing_cameras:
        problems.append(
            f"  camera sources {missing_cameras} not found — "
            f"available cameras: {', '.join(cameras) or '(none)'}"
        )

    return problems


def _raw_segment_duration_s(index_ranges, segment_id: str, index_column: str) -> float | None:
    """Read a segment's raw index-column span from dataset manifest metadata.

    This is a pure metadata lookup — no per-row query, no video decode — so it's
    cheap enough to check for every segment before doing any real work, even on
    very large recordings. It reflects the *raw* recording span, independent of
    trimming or cameras: a plain upper-bound sanity check for `max_episode_duration_s`
    (e.g. to catch a stuck or left-running recording before it costs minutes of
    wasted video decoding).

    Args:
        index_ranges: The DataFusion DataFrame from `dataset.get_index_ranges()`.
        segment_id: Segment to look up.
        index_column: Timeline whose start/end columns to read (e.g. "canonical_time").

    Returns:
        Raw duration in seconds, or None if it can't be determined — callers
        should treat that as "unknown" and proceed rather than block on it.
    """
    from datafusion import col

    try:
        row = (
            index_ranges.filter(col("rerun_segment_id") == segment_id)
            .select(f"{index_column}:start", f"{index_column}:end")
            .to_arrow_table()
        )
        if row.num_rows == 0:
            return None
        start = row[f"{index_column}:start"][0].as_py()
        end = row[f"{index_column}:end"][0].as_py()
        return (end - start).total_seconds()
    except Exception as e:
        logger.debug(
            "Could not read raw duration for segment {}: {}", segment_id[:8], e
        )
        return None


def _validate_sources(dataset, config: ExportConfig, segment_id: str) -> None:
    """Preflight-check configured sources against the recording's schema.

    Raises ValueError listing the available timelines / signals / cameras when a
    configured name is missing, so a typo produces an actionable message up front
    instead of a cryptic per-episode DataFusion schema error.
    """
    try:
        schema = dataset.filter_segments(segment_id).schema()
        column_names = [field.name for field in schema]
        # index_columns() / column_names() are methods on the Schema object.
        index_columns = schema.index_columns
        index_columns = index_columns() if callable(index_columns) else index_columns
        timelines = sorted(str(getattr(c, "name", c)) for c in index_columns)
    except Exception as e:
        # Never let validation itself break an otherwise-valid export.
        logger.warning("Skipping source validation (could not read schema): {}", e)
        return

    problems = _find_missing_sources(
        config,
        timelines=timelines,
        signals=_sources_for_suffix(column_names, _SCALARS_SUFFIX),
        cameras=_sources_for_suffix(column_names, _VIDEO_SUFFIX),
    )
    if problems:
        raise ValueError(
            "Export config references sources that are not in the recording:\n"
            + "\n".join(problems)
            + "\n\nEdit the config so these names match the recording (names above)."
        )


def _raise_fd_limit_for(num_files: int) -> None:
    """Best-effort: raise this process's open-file limit to fit num_files.

    Loading many .rrd files into one temporary Rerun server opens roughly one
    file descriptor per file, which can exceed the OS's default per-process
    limit (e.g. 256 on macOS) well before any large-ish recording count. This
    raises only this process's soft RLIMIT_NOFILE — it doesn't touch the shell
    or any other process — so it's safe to attempt unconditionally. A no-op
    (logged at debug level) if unsupported (e.g. Windows) or if the hard limit
    is already below what's needed.
    """
    try:
        import resource
    except ImportError:
        return  # RLIMIT_NOFILE is POSIX-only; nothing to do on Windows.

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        needed = num_files + 256  # headroom for the server's own sockets/logs
        target = min(hard, max(soft, needed))
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.debug(
                "Raised open-file limit {} → {} (hard limit {}) for {} files",
                soft,
                target,
                hard,
                num_files,
            )
    except (ValueError, OSError) as e:
        logger.debug("Could not raise open-file limit: {}", e)


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
        _raise_fd_limit_for(len(rrd_paths))

        # Manage the server context manually (rather than via `with`) so this
        # try/except only wraps startup, not the caller's entire export — a
        # broad `except` around a `yield` would also catch unrelated errors
        # raised later, deep in the caller's own processing. Note: the server
        # actually starts during construction, not __enter__(), so both must
        # be inside the try.
        try:
            server_cm = rr.server.Server(datasets={dataset_name: rrd_str_paths})
            server = server_cm.__enter__()
        except ValueError as e:
            raise RuntimeError(
                f"Failed to start the local Rerun server while loading "
                f"{len(rrd_paths)} .rrd files: {e}\n\n"
                "If this is a large number of files, it may be an OS "
                "open-file-descriptor limit — try raising it "
                "(e.g. `ulimit -n 8192`) and retry. Otherwise, check that the "
                ".rrd files aren't corrupted."
            ) from e

        try:
            client = server.client()
            dataset = client.get_dataset(name=dataset_name)
            yield dataset
        finally:
            server_cm.__exit__(None, None, None)


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

        # Preflight: fail fast with the available field list if a source is wrong.
        _validate_sources(dataset, config, all_segment_ids[0])

        # Create the episode sampler (Layer 1: decode + resample)
        sampler = EpisodeSampler(
            config=config,
            dataset=dataset,
            segment_ids=all_segment_ids,
        )

        # Create the export head (Layer 2: format-specific writer)
        head = _create_export_head(config, output_dir)

        # For the max_episode_duration_s safety check: fetch the dataset's
        # per-segment raw time ranges once (a cheap manifest-metadata read),
        # rather than once per episode.
        index_ranges = None
        if config.max_episode_duration_s is not None:
            try:
                index_ranges = dataset.get_index_ranges()
            except Exception as e:
                logger.warning(
                    "Could not read dataset index ranges; "
                    "max_episode_duration_s check will be skipped: {}",
                    e,
                )

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
                # Upper-bound safety check: reject a segment whose raw recording
                # span is implausibly long (e.g. a stuck or left-running
                # recording) using cheap manifest metadata — before doing any
                # trimming or (potentially very expensive) video decode. This is
                # independent of trimming: it looks at the raw span only.
                if index_ranges is not None:
                    raw_duration_s = _raw_segment_duration_s(
                        index_ranges, segment_id, config.index_column
                    )
                    logger.debug(
                        "Episode {} ({}): raw recording span = {} (limit {}s)",
                        episode_id,
                        segment_id[:8],
                        f"{raw_duration_s:.2f}s"
                        if raw_duration_s is not None
                        else "unknown — could not determine, proceeding",
                        config.max_episode_duration_s,
                    )
                    if (
                        raw_duration_s is not None
                        and raw_duration_s > config.max_episode_duration_s
                    ):
                        reason = (
                            f"Raw recording span ~{raw_duration_s:.2f}s exceeds "
                            f"max_episode_duration_s={config.max_episode_duration_s}s "
                            "— likely a stuck/left-running recording "
                            "(skipped before processing)"
                        )
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
