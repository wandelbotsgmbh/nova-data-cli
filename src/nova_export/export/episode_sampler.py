"""Episode sampler — resamples decoded video/action/state at target FPS.

This layer handles:
- Building a fixed-rate time grid at target FPS
- Sampling video frames from decoded caches
- Querying action/state data at grid timestamps
- Combining into unified Sample objects

The sampler is format-agnostic — it yields Sample objects that export heads
can transform into any target format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import av
import numpy as np
import numpy.typing as npt
from loguru import logger
from tqdm import tqdm

from nova_export.export.camera_lag import REFERENCE_CAMERA, measure_episode_delta_ns
from nova_export.export.video_decoder import FrameCache, VideoDecoder

if TYPE_CHECKING:
    from nova_export.export.config import ExportConfig


@dataclass
class Sample:
    """A single sample at a fixed timestamp.

    Contains all data needed for a single training step:
    - Action vector (what the robot should do)
    - State vector (current robot state observation)
    - Images (camera observations)
    """

    timestamp_ns: int
    action: npt.NDArray[np.float32]
    state: npt.NDArray[np.float32]
    images: dict[str, npt.NDArray[np.uint8]]  # camera_name -> HWC RGB

    @property
    def frame_index(self) -> int | None:
        """Frame index within the episode (set by sampler)."""
        return getattr(self, "_frame_index", None)

    @frame_index.setter
    def frame_index(self, value: int) -> None:
        self._frame_index = value


@dataclass
class Episode:
    """A complete episode with metadata and samples."""

    segment_id: str
    episode_index: int
    samples: list[Sample]

    @property
    def num_frames(self) -> int:
        return len(self.samples)

    @property
    def duration_s(self) -> float:
        if not self.samples:
            return 0.0
        return (self.samples[-1].timestamp_ns - self.samples[0].timestamp_ns) / 1e9


def make_time_grid(start_ns: int, end_ns: int, fps: int) -> npt.NDArray[np.int64]:
    """Generate a fixed-rate time grid.

    Args:
        start_ns: Start timestamp in nanoseconds.
        end_ns: End timestamp in nanoseconds (inclusive).
        fps: Target frames per second.

    Returns:
        Array of timestamps at fixed intervals.
    """
    step_ns = int(1e9 / fps)
    return np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)


class EpisodeSampler:
    """Samples episodes at fixed FPS from Rerun recordings.

    Pipeline:
    1. For each segment, decode all video streams into frame caches
    2. Find the valid time range where all streams have data
    3. Build a time grid at target FPS
    4. For each grid timestamp:
       - Sample nearest video frame from each cache
       - Query action/state via fill_latest_at
       - Combine into a Sample
    """

    def __init__(
        self,
        config: ExportConfig,
        dataset: Any,  # rr.server.Dataset
        segment_ids: list[str] | None = None,
    ):
        """Initialize the sampler.

        Args:
            config: Export configuration with fps, columns, cameras.
            dataset: Rerun catalog dataset.
            segment_ids: Optional subset of segments to process.
        """
        self.config = config
        self.dataset = dataset
        self.segment_ids = segment_ids or dataset.segment_ids()

        # segment_id -> measured camera-lag alignment result (or skip
        # reason), written out by write_camera_lag_report() after export.
        self.camera_lag_report: dict[str, dict[str, Any]] = {}

        self._video_decoders: dict[str, VideoDecoder] = {}
        self._target_sizes: dict[str, tuple[int, int] | None] = {}
        for cam in config.cameras:
            self._video_decoders[cam.source] = VideoDecoder()
            # Store target resize dimensions (width, height) if specified
            if cam.width is not None and cam.height is not None:
                self._target_sizes[cam.source] = (cam.width, cam.height)
            else:
                self._target_sizes[cam.source] = None

        # Log resize targets
        resize_info = [
            f"{name}→{size[0]}x{size[1]}"
            for name, size in self._target_sizes.items()
            if size is not None
        ]

        logger.info(
            "EpisodeSampler initialized: {} segments, {} cameras, target {}fps{}",
            len(self.segment_ids),
            len(config.cameras),
            config.fps,
            f", resize: {resize_info}" if resize_info else "",
        )

    def iterate_episodes(self) -> Iterator[Episode]:
        """Iterate over all episodes, yielding decoded samples.

        Yields:
            Episode objects containing all samples at target FPS.
        """
        for ep_idx, segment_id in enumerate(self.segment_ids):
            logger.info("Processing episode {} (segment {})", ep_idx, segment_id[:8])

            try:
                episode = self._process_segment(segment_id, ep_idx)
                if episode is not None:
                    yield episode
            except Exception as e:
                logger.error("Failed to process segment {}: {}", segment_id[:8], e)
                continue

    def _process_segment(self, segment_id: str, episode_index: int) -> Episode | None:
        """Process a single segment into an episode.

        Args:
            segment_id: Segment ID to process.
            episode_index: Index of this episode in the dataset.

        Returns:
            Episode with all samples, or None if processing failed.
        """
        # Step 1: Decode all video streams
        frame_caches: dict[str, FrameCache] = {}
        for cam in self.config.cameras:
            entity_path = self.config._normalize_path(cam.source)
            cache = self._video_decoders[cam.source].decode_segment(
                dataset=self.dataset,
                segment_id=segment_id,
                video_entity=entity_path,
                index_column=self.config.index_column,
            )
            if cache.num_frames == 0:
                logger.warning(
                    "No video frames for {} in segment {}", cam.source, segment_id[:8]
                )
                return None
            frame_caches[cam.source] = cache

        # Step 1.5: Measure this episode's camera/state alignment delta,
        # *before* trimming or grid-building. Trimming's own bounds are
        # unaffected (still computed from joint_positions, not video), but
        # every camera frame lookup for the rest of this episode must use
        # the aligned timestamp, not just near the onset point. See
        # docs/investigations/camera-joint-sync-lag.md.
        camera_lag_delta_ns = self._measure_camera_lag(segment_id, frame_caches)
        if camera_lag_delta_ns is None:
            return None

        # Step 2: Find valid time range (intersection of all streams)
        valid_range = self._find_valid_range(segment_id, frame_caches)
        if valid_range is None:
            return None

        valid_start_ns, valid_end_ns = valid_range

        # Step 3: Build time grid at target FPS
        time_grid = make_time_grid(valid_start_ns, valid_end_ns, self.config.fps)
        logger.info(
            "Episode {}: {} samples at {}fps over {:.2f}s",
            episode_index,
            len(time_grid),
            self.config.fps,
            (valid_end_ns - valid_start_ns) / 1e9,
        )

        # Step 4: Query action/state for all timestamps
        action_state_data = self._query_action_state(segment_id, time_grid)
        if action_state_data is None:
            return None

        # Step 5: Combine into samples
        samples = self._build_samples(
            time_grid, frame_caches, action_state_data, camera_lag_delta_ns
        )

        return Episode(
            segment_id=segment_id,
            episode_index=episode_index,
            samples=samples,
        )

    def _find_valid_range(
        self,
        segment_id: str,
        frame_caches: dict[str, FrameCache],
    ) -> tuple[int, int] | None:
        """Find the time range where all streams have data.

        Args:
            segment_id: Segment ID.
            frame_caches: Decoded frame caches for each camera.

        Returns:
            (start_ns, end_ns) tuple, or None if no valid overlap.
        """
        from datafusion import col

        # Start with video cache bounds. With no cameras, start unbounded and let
        # action availability (and trimming) define the range.
        if frame_caches:
            valid_start_ns = max(cache.start_ns for cache in frame_caches.values())
            valid_end_ns = min(cache.end_ns for cache in frame_caches.values())
        else:
            valid_start_ns = -(2**63)
            valid_end_ns = 2**63 - 1

        # Narrow to action availability. The first action source drives the
        # sampling cadence; additional action sources are filled latest-at.
        view = self.dataset.filter_segments(segment_id)
        action_col = self.config.action_columns()[0]

        try:
            action_reader = view.reader(index=self.config.index_column).filter(
                col(action_col).is_not_null()
            )
            action_table = action_reader.select(
                self.config.index_column
            ).to_arrow_table()

            if action_table.num_rows == 0:
                logger.warning("No action data in segment {}", segment_id[:8])
                return None

            action_times = self._extract_timestamps_ns(
                action_table[self.config.index_column]
            )
            action_start_ns = int(np.min(action_times))
            action_end_ns = int(np.max(action_times))

            valid_start_ns = max(valid_start_ns, action_start_ns)
            valid_end_ns = min(valid_end_ns, action_end_ns)

        except Exception as e:
            logger.error("Failed to query action times: {}", e)
            raise RuntimeError(f"Action entity not found in recording: {e}") from e

        if valid_start_ns >= valid_end_ns:
            logger.warning("No overlapping time range in segment {}", segment_id[:8])
            return None

        # Apply trimming to narrow bounds
        trim_start, trim_end = self._apply_trimming(
            segment_id, valid_start_ns, valid_end_ns
        )
        valid_start_ns = max(valid_start_ns, trim_start)
        valid_end_ns = min(valid_end_ns, trim_end)

        if valid_start_ns >= valid_end_ns:
            logger.warning(
                "Trimming removed all data in segment {} (mode={})",
                segment_id[:8],
                self.config.trimming.mode,
            )
            return None

        return (valid_start_ns, valid_end_ns)

    def _apply_trimming(
        self,
        segment_id: str,
        raw_start_ns: int,
        raw_end_ns: int,
    ) -> tuple[int, int]:
        """Apply trimming config to narrow episode bounds.

        Returns (trim_start_ns, trim_end_ns). The caller takes the intersection
        with raw bounds, so this can only narrow the range.
        """
        from datafusion import col

        cfg = self.config.trimming

        if cfg.mode == "all_present":
            return (raw_start_ns, raw_end_ns)

        col_name = self.config.trimming_column()
        view = self.dataset.filter_segments(segment_id)

        if cfg.mode == "signal_presence":
            reader = view.reader(index=self.config.index_column).filter(
                col(col_name).is_not_null()
            )
            table = reader.select(self.config.index_column).to_arrow_table()

            if table.num_rows == 0:
                logger.warning(
                    "Trimming source '{}' has no data in segment {}, skipping trim",
                    cfg.source,
                    segment_id[:8],
                )
                return (raw_start_ns, raw_end_ns)

            times = self._extract_timestamps_ns(table[self.config.index_column])
            return (int(np.min(times)), int(np.max(times)))

        elif cfg.mode == "signal_change":
            reader = view.reader(index=self.config.index_column).filter(
                col(col_name).is_not_null()
            )
            table = reader.select(self.config.index_column, col_name).to_arrow_table()

            if table.num_rows < 2:
                logger.warning(
                    "Trimming source '{}' has <2 samples in segment {}, skipping trim",
                    cfg.source,
                    segment_id[:8],
                )
                return (raw_start_ns, raw_end_ns)

            times = self._extract_timestamps_ns(table[self.config.index_column])

            # Extract values — handle both scalar and list (vector) columns
            raw_col = table[col_name]
            if hasattr(raw_col.type, "value_type"):
                # List-type column (vector signal like joint positions)
                values = np.array(raw_col.to_pylist(), dtype=np.float64)
            else:
                # Scalar column
                values = raw_col.to_numpy().astype(np.float64)

            # Compute diffs and detect activity
            if values.ndim == 1:
                diffs = np.abs(np.diff(values))
            else:
                diffs = np.max(np.abs(np.diff(values, axis=0)), axis=1)

            active_mask = diffs > cfg.threshold
            active_indices = np.where(active_mask)[0]

            if len(active_indices) == 0:
                logger.warning(
                    "No activity detected in '{}' (threshold={}), skipping trim",
                    cfg.source,
                    cfg.threshold,
                )
                return (raw_start_ns, raw_end_ns)

            # First activity → trim start
            # Last activity + 1 (the sample after last change) + tail buffer → trim end
            trim_start = int(times[active_indices[0]])
            last_active_idx = (
                active_indices[-1] + 1
            )  # index into times (diff shifts by 1)
            trim_end = int(times[last_active_idx]) + cfg.tail_ms * 1_000_000

            return (trim_start, trim_end)

        return (raw_start_ns, raw_end_ns)

    def _signal_change_onset(self, segment_id: str) -> int | None:
        """Raw timestamp of the first sample-to-sample change exceeding
        `trimming.threshold` on the trimming source column (L-inf norm for
        vector signals) — the same "first real motion" detector
        `_apply_trimming`'s `signal_change` mode uses internally, but
        *without* the tail_ms backoff it applies to produce a trim boundary.

        Deliberately duplicated (not routed through `_apply_trimming`) so
        this can be used standalone as the state-side onset reference for
        per-episode camera-lag alignment, computed before trimming runs, and
        so trimming's own logic stays untouched. Only meaningful when
        `trimming.mode == "signal_change"` — callers are expected to check
        that first.
        """
        from datafusion import col

        cfg = self.config.trimming
        col_name = self.config.trimming_column()
        view = self.dataset.filter_segments(segment_id)

        reader = view.reader(index=self.config.index_column).filter(
            col(col_name).is_not_null()
        )
        table = reader.select(self.config.index_column, col_name).to_arrow_table()

        if table.num_rows < 2:
            return None

        times = self._extract_timestamps_ns(table[self.config.index_column])

        raw_col = table[col_name]
        if hasattr(raw_col.type, "value_type"):
            values = np.array(raw_col.to_pylist(), dtype=np.float64)
        else:
            values = raw_col.to_numpy().astype(np.float64)

        if values.ndim == 1:
            diffs = np.abs(np.diff(values))
        else:
            diffs = np.max(np.abs(np.diff(values, axis=0)), axis=1)

        active_indices = np.where(diffs > cfg.threshold)[0]
        if len(active_indices) == 0:
            return None

        return int(times[active_indices[0]])

    def _measure_camera_lag(
        self,
        segment_id: str,
        frame_caches: dict[str, FrameCache],
    ) -> int | None:
        """Measure this episode's camera/state alignment delta (see
        `camera_lag.py`), log it, and record it in `camera_lag_report`.

        Returns the delta in ns to add to every camera frame lookup for this
        episode (0 if alignment isn't applicable — trimming isn't in
        `signal_change` mode, or the reference camera isn't configured), or
        None if the episode must be skipped entirely (no clear onset found,
        or the measured delta is implausible). Never returns a value meaning
        "export unshifted" in a failure case — callers must skip on None.
        """
        if self.config.trimming.mode != "signal_change":
            # No established "first real motion" signal to anchor on.
            return 0

        if REFERENCE_CAMERA not in frame_caches:
            logger.warning(
                "Segment {}: camera-lag alignment needs '{}' in config.cameras; "
                "exporting this episode's camera frames unshifted",
                segment_id[:8],
                REFERENCE_CAMERA,
            )
            return 0

        state_onset_ns = self._signal_change_onset(segment_id)
        if state_onset_ns is None:
            reason = "no state-side motion onset found for camera-lag alignment"
            logger.warning("Segment {}: {} — skipping episode", segment_id[:8], reason)
            self.camera_lag_report[segment_id] = {"status": "skipped", "reason": reason}
            return None

        delta_ns, error = measure_episode_delta_ns(
            frame_caches[REFERENCE_CAMERA], state_onset_ns
        )
        if error is not None:
            logger.warning(
                "Segment {}: camera-lag alignment failed ({}) — skipping episode",
                segment_id[:8],
                error,
            )
            self.camera_lag_report[segment_id] = {
                "status": "skipped",
                "reason": error,
                "state_onset_ns": state_onset_ns,
            }
            return None

        logger.info(
            "Segment {}: camera-lag delta = {:.1f}ms (state onset {}, camera onset {})",
            segment_id[:8],
            delta_ns / 1e6,
            state_onset_ns,
            state_onset_ns + delta_ns,
        )
        self.camera_lag_report[segment_id] = {
            "status": "ok",
            "delta_ns": delta_ns,
            "delta_ms": round(delta_ns / 1e6, 1),
            "state_onset_ns": state_onset_ns,
        }
        return delta_ns

    def write_camera_lag_report(self, output_dir: Path) -> None:
        """Write per-episode camera-lag measurements to
        `<output_dir>/camera_lag_report.json` (segment_id -> measured delta,
        or skip reason) so they're inspectable after the export finishes,
        not just console/loguru noise."""
        if not self.camera_lag_report:
            return
        report_path = output_dir / "camera_lag_report.json"
        report_path.write_text(json.dumps(self.camera_lag_report, indent=2))
        logger.info("Camera-lag report written to {}", report_path)

    def _query_action_state(
        self,
        segment_id: str,
        time_grid: npt.NDArray[np.int64],
    ) -> dict[int, dict[str, npt.NDArray[np.float32]]] | None:
        """Query action and state data at grid timestamps using fill_latest_at.

        Args:
            segment_id: Segment ID.
            time_grid: Array of target timestamps.

        Returns:
            Dict mapping timestamp_ns -> {"action": array, "state": array}.
        """
        from datafusion import col

        view = self.dataset.filter_segments(segment_id)

        # Build index values as a reader result (what Rerun expects)
        # Query at action timestamps, then use fill_latest_at for state
        base_reader = view.reader(index=self.config.index_column)

        action_columns = self.config.action_columns()
        index_values_reader = base_reader.filter(
            col(action_columns[0]).is_not_null()
        ).select("rerun_segment_id", self.config.index_column)

        # Select columns
        columns = [self.config.index_column, *action_columns]
        columns.extend(self.config.state_columns())

        # Query with fill_latest_at
        reader = view.reader(
            index=self.config.index_column,
            using_index_values=index_values_reader,
            fill_latest_at=True,
        )

        try:
            result_table = reader.select(*columns).to_arrow_table()
        except Exception as e:
            logger.error("Failed to query action/state: {}", e)
            raise RuntimeError(f"Failed to query action/state data: {e}") from e

        if result_table.num_rows == 0:
            logger.warning(
                "No action/state data returned for segment {}", segment_id[:8]
            )
            return None

        # Build lookup table from query results
        query_timestamps = self._extract_timestamps_ns(
            result_table[self.config.index_column]
        )
        action_cols = [
            result_table[col].combine_chunks() for col in self.config.action_columns()
        ]
        state_cols = [
            result_table[col].combine_chunks() for col in self.config.state_columns()
        ]

        # Build raw data dict from query timestamps
        raw_data: dict[int, dict[str, npt.NDArray[np.float32]]] = {}
        for i, ts in enumerate(query_timestamps):
            # Concatenate all action sources in order. Skip the row if any action
            # source is missing, so the action vector keeps a constant dimension.
            action_parts = []
            for action_col in action_cols:
                action_part = self._extract_scalars(action_col, i)
                if action_part is None:
                    break
                action_parts.append(action_part)
            if len(action_parts) != len(action_cols):
                continue
            action = (
                action_parts[0]
                if len(action_parts) == 1
                else np.concatenate(action_parts)
            )

            state_parts = []
            for state_col in state_cols:
                state_part = self._extract_scalars(state_col, i)
                if state_part is not None:
                    state_parts.append(state_part)

            state = (
                np.concatenate(state_parts)
                if state_parts
                else np.array([], dtype=np.float32)
            )
            raw_data[int(ts)] = {"action": action, "state": state}

        # Resample to time_grid using nearest-neighbor lookup
        sorted_query_ts = np.array(sorted(raw_data.keys()), dtype=np.int64)
        data: dict[int, dict[str, npt.NDArray[np.float32]]] = {}

        for target_ts in time_grid:
            # Find nearest query timestamp
            idx = np.searchsorted(sorted_query_ts, target_ts)
            if idx == 0:
                nearest_ts = sorted_query_ts[0]
            elif idx >= len(sorted_query_ts):
                nearest_ts = sorted_query_ts[-1]
            else:
                # Pick closest
                if (target_ts - sorted_query_ts[idx - 1]) <= (
                    sorted_query_ts[idx] - target_ts
                ):
                    nearest_ts = sorted_query_ts[idx - 1]
                else:
                    nearest_ts = sorted_query_ts[idx]

            data[int(target_ts)] = raw_data[int(nearest_ts)]

        return data

    def _build_samples(
        self,
        time_grid: npt.NDArray[np.int64],
        frame_caches: dict[str, FrameCache],
        action_state_data: dict[int, dict[str, npt.NDArray[np.float32]]],
        camera_lag_delta_ns: int = 0,
    ) -> list[Sample]:
        """Combine video frames and action/state into Sample objects.

        Args:
            time_grid: Target timestamps.
            frame_caches: Decoded frame caches.
            action_state_data: Action/state data at each timestamp.
            camera_lag_delta_ns: This episode's measured camera-lag delta
                (see `_measure_camera_lag`). Added to every camera frame
                lookup timestamp so the frame pulled for grid-time `t` is
                the one whose visual content actually matches `t`, not
                `t - delta`. Zero if alignment wasn't measured/applicable.

        Returns:
            List of Sample objects.
        """
        samples: list[Sample] = []

        for frame_idx, ts in enumerate(
            tqdm(time_grid, desc="Building samples", leave=False)
        ):
            ts_int = int(ts)

            # Get action/state
            if ts_int not in action_state_data:
                logger.debug("Missing action/state at timestamp {}", ts_int)
                continue

            action = action_state_data[ts_int]["action"]
            state = action_state_data[ts_int]["state"]

            # Get video frames
            images: dict[str, npt.NDArray[np.uint8]] = {}
            all_images_ok = True

            for cam_name, cache in frame_caches.items():
                frame = cache.get_frame_at(ts_int + camera_lag_delta_ns)
                if frame is None:
                    logger.debug(
                        "Missing frame for {} at timestamp {}", cam_name, ts_int
                    )
                    all_images_ok = False
                    break

                # Resize frame if target size specified in config.
                # Use PyAV (libswscale) rather than OpenCV so we don't load a
                # second ffmpeg/libavdevice and clash with av's (macOS objc warning).
                target_size = self._target_sizes.get(cam_name)
                if target_size is not None:
                    vf = av.VideoFrame.from_ndarray(
                        np.ascontiguousarray(frame), format="rgb24"
                    )
                    vf = vf.reformat(width=target_size[0], height=target_size[1])
                    frame = vf.to_ndarray(format="rgb24")

                images[cam_name] = frame

            if not all_images_ok:
                continue

            sample = Sample(
                timestamp_ns=ts_int,
                action=action,
                state=state,
                images=images,
            )
            sample.frame_index = frame_idx
            samples.append(sample)

        return samples

    def _extract_timestamps_ns(self, ts_column: Any) -> npt.NDArray[np.int64]:
        """Extract timestamps from PyArrow column as nanosecond integers."""
        timestamps = []
        for chunk in ts_column.chunks:
            for val in chunk:
                ts = val.as_py()
                if isinstance(ts, (int, float)):
                    timestamps.append(int(ts))
                elif hasattr(ts, "value"):
                    timestamps.append(int(ts.value))
                else:
                    timestamps.append(int(ts))
        return np.array(timestamps, dtype=np.int64)

    def _extract_scalars(
        self, col: Any, row_idx: int
    ) -> npt.NDArray[np.float32] | None:
        """Extract scalar array from PyArrow column at given row."""
        try:
            val = col[row_idx]
            if val.is_valid:
                arr = val.as_py()
                if isinstance(arr, list):
                    return np.array(arr, dtype=np.float32)
                return np.array([arr], dtype=np.float32)
        except Exception:
            pass
        return None
