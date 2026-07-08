"""Tests for the v2 export pipeline.

Tests cover:
- VideoDecoder: Sequential H.264 decoding
- FrameCache: Timestamp-based frame lookup
- EpisodeSampler: Fixed-rate resampling
- LeRobotHead: Feature inference and writing
- End-to-end export orchestration
- Real integration test with actual RRD data (marked slow)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest
from pydantic import ValidationError

from nova_export.export.config import CameraSource, ExportConfig
from nova_export.export.episode_sampler import (
    Episode,
    EpisodeSampler,
    Sample,
    make_time_grid,
)
from nova_export.export.exporter import export_recordings
from nova_export.export.heads.base import ExportResult
from nova_export.export.heads.lerobot import LeRobotHead
from nova_export.export.video_decoder import (
    FrameCache,
    VideoDecoder,
    _avcc_to_annex_b,
    _is_annex_b,
)

# =============================================================================
# Test Data Paths
# =============================================================================

# Path to the 1-episode-test recording
TEST_RRD_PATH = (
    Path(__file__).parent.parent.parent
    / "recordings"
    / "1-episode-test"
    / "fbca7c1dfd4a"
    / "recording.rrd"
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def create_test_h264_packets(
    width: int = 64,
    height: int = 64,
    num_frames: int = 10,
    fps: int = 30,
) -> list[tuple[bytes, int]]:
    """Create real H.264 packets for testing using PyAV.

    Returns:
        List of (packet_bytes, timestamp_ns) tuples.
    """
    packets: list[tuple[bytes, int]] = []

    # Create encoder
    codec = av.codec.Codec("libx264", "w")
    encoder = codec.create()
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, fps)
    encoder.gop_size = 5  # Keyframe every 5 frames for testing
    encoder.max_b_frames = 0
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency"}
    encoder.open()

    ns_per_frame = int(1e9 / fps)

    for i in range(num_frames):
        # Create a test frame with a gradient pattern
        y_plane = np.full((height, width), fill_value=(i * 25) % 256, dtype=np.uint8)
        u_plane = np.full((height // 2, width // 2), fill_value=128, dtype=np.uint8)
        v_plane = np.full((height // 2, width // 2), fill_value=128, dtype=np.uint8)

        frame = av.VideoFrame(width=width, height=height, format="yuv420p")
        frame.planes[0].update(y_plane.tobytes())
        frame.planes[1].update(u_plane.tobytes())
        frame.planes[2].update(v_plane.tobytes())
        frame.pts = i
        frame.time_base = encoder.time_base

        for packet in encoder.encode(frame):
            timestamp_ns = i * ns_per_frame
            packets.append((bytes(packet), timestamp_ns))

    # Flush encoder
    for packet in encoder.encode(None):
        timestamp_ns = (num_frames - 1) * ns_per_frame
        packets.append((bytes(packet), timestamp_ns))

    return packets


def create_test_frame(
    height: int = 64,
    width: int = 64,
    value: int = 128,
) -> npt.NDArray[np.uint8]:
    """Create a test RGB frame."""
    frame = np.full((height, width, 3), fill_value=value, dtype=np.uint8)
    return frame


def create_test_sample(
    timestamp_ns: int = 0,
    action_dim: int = 7,
    state_dim: int = 14,
    image_shape: tuple[int, int, int] = (64, 64, 3),
    camera_names: list[str] | None = None,
) -> Sample:
    """Create a test Sample object."""
    if camera_names is None:
        camera_names = ["wrist"]

    images = {
        name: np.random.randint(0, 255, image_shape, dtype=np.uint8)
        for name in camera_names
    }

    return Sample(
        timestamp_ns=timestamp_ns,
        action=np.random.randn(action_dim).astype(np.float32),
        state=np.random.randn(state_dim).astype(np.float32),
        images=images,
    )


def create_test_episode(
    segment_id: str = "test-segment",
    episode_index: int = 0,
    num_samples: int = 10,
    fps: int = 15,
) -> Episode:
    """Create a test Episode object."""
    ns_per_sample = int(1e9 / fps)
    samples = [
        create_test_sample(timestamp_ns=i * ns_per_sample) for i in range(num_samples)
    ]
    for i, s in enumerate(samples):
        s.frame_index = i

    return Episode(
        segment_id=segment_id,
        episode_index=episode_index,
        samples=samples,
    )


# =============================================================================
# FrameCache Tests
# =============================================================================


class TestFrameCache:
    """Tests for FrameCache timestamp-based lookup."""

    def test_empty_cache(self):
        """Empty cache returns None for lookups."""
        cache = FrameCache()
        assert cache.num_frames == 0
        assert cache.get_frame_at(1000) is None
        assert cache.get_frame_index(1000) == -1

    def test_single_frame(self):
        """Single frame cache returns that frame for any timestamp."""
        frame = create_test_frame(value=100)
        cache = FrameCache(
            frames=[frame],
            timestamps_ns=np.array([1000], dtype=np.int64),
        )

        assert cache.num_frames == 1
        assert cache.start_ns == 1000
        assert cache.end_ns == 1000

        # Any timestamp should return the single frame
        result = cache.get_frame_at(0)
        assert result is not None
        assert np.array_equal(result, frame)

        result = cache.get_frame_at(5000)
        assert np.array_equal(result, frame)

    def test_nearest_frame_lookup(self):
        """Lookup returns nearest frame by timestamp."""
        frames = [create_test_frame(value=i * 50) for i in range(5)]
        timestamps = np.array([0, 1000, 2000, 3000, 4000], dtype=np.int64)
        cache = FrameCache(frames=frames, timestamps_ns=timestamps)

        assert cache.num_frames == 5
        assert cache.start_ns == 0
        assert cache.end_ns == 4000
        assert cache.duration_s == 4e-6  # 4000 ns = 4 microseconds

        # Exact matches
        assert np.array_equal(cache.get_frame_at(0), frames[0])
        assert np.array_equal(cache.get_frame_at(2000), frames[2])
        assert np.array_equal(cache.get_frame_at(4000), frames[4])

        # Nearest to lower
        assert np.array_equal(cache.get_frame_at(400), frames[0])

        # Nearest to upper
        assert np.array_equal(cache.get_frame_at(600), frames[1])

        # Midpoint goes to lower
        assert np.array_equal(cache.get_frame_at(500), frames[0])

        # Before start
        assert np.array_equal(cache.get_frame_at(-1000), frames[0])

        # After end
        assert np.array_equal(cache.get_frame_at(10000), frames[4])

    def test_frame_index_lookup(self):
        """get_frame_index returns correct indices."""
        frames = [create_test_frame() for _ in range(3)]
        timestamps = np.array([100, 200, 300], dtype=np.int64)
        cache = FrameCache(frames=frames, timestamps_ns=timestamps)

        assert cache.get_frame_index(100) == 0
        assert cache.get_frame_index(200) == 1
        assert cache.get_frame_index(300) == 2

        # Nearest (equidistant rounds to earlier frame with <=)
        assert cache.get_frame_index(140) == 0
        assert cache.get_frame_index(160) == 1
        assert (
            cache.get_frame_index(250) == 1
        )  # Equidistant (50 from 200 and 300) -> earlier
        assert cache.get_frame_index(251) == 2  # Closer to 300


# =============================================================================
# VideoDecoder Tests
# =============================================================================


class TestVideoDecoder:
    """Tests for VideoDecoder H.264 decoding."""

    def test_annex_b_detection(self):
        """Test Annex B start code detection."""
        # 3-byte start code
        assert _is_annex_b(b"\x00\x00\x01\x67...")

        # 4-byte start code
        assert _is_annex_b(b"\x00\x00\x00\x01\x67...")

        # AVCC format (length-prefixed)
        assert not _is_annex_b(b"\x00\x00\x00\x10\x67...")

        # Random data
        assert not _is_annex_b(b"\x12\x34\x56\x78")

    def test_avcc_to_annex_b_conversion(self):
        """Test AVCC to Annex B conversion."""
        # Simple AVCC: 4-byte length (5) + 5 bytes of NALU
        avcc = b"\x00\x00\x00\x05\x67\x42\x00\x1e\x00"

        annex_b = _avcc_to_annex_b(avcc)

        # Should have 4-byte start code + original NALU
        assert annex_b.startswith(b"\x00\x00\x00\x01")
        assert annex_b[4:] == b"\x67\x42\x00\x1e\x00"

    def test_decode_real_packets(self):
        """Test decoding real H.264 packets created by PyAV."""
        packets = create_test_h264_packets(width=64, height=64, num_frames=5, fps=30)
        assert len(packets) > 0, "Should have created some packets"

        # Decode manually to verify our decoder logic works
        decoder = av.CodecContext.create("h264", "r")
        frames_decoded = []

        for packet_bytes, _ in packets:
            for frame in decoder.decode(av.Packet(packet_bytes)):
                frames_decoded.append(frame.to_ndarray(format="rgb24"))

        # Flush
        for frame in decoder.decode(None):
            frames_decoded.append(frame.to_ndarray(format="rgb24"))

        assert len(frames_decoded) == 5, f"Expected 5 frames, got {len(frames_decoded)}"
        assert frames_decoded[0].shape == (64, 64, 3)


# =============================================================================
# make_time_grid Tests
# =============================================================================


class TestMakeTimeGrid:
    """Tests for the time grid generation function."""

    def test_basic_grid(self):
        """Generate a simple time grid."""
        grid = make_time_grid(start_ns=0, end_ns=1_000_000_000, fps=10)

        # 10 fps over 1 second = 11 samples (0, 100ms, 200ms, ..., 1000ms)
        assert len(grid) == 11
        assert grid[0] == 0
        assert grid[-1] == 1_000_000_000

        # Check spacing
        diffs = np.diff(grid)
        assert np.all(diffs == 100_000_000)  # 100ms in ns

    def test_15fps_grid(self):
        """Generate 15fps grid (common target FPS)."""
        grid = make_time_grid(start_ns=0, end_ns=1_000_000_000, fps=15)

        # 15 fps = 66.67ms per frame
        step_ns = int(1e9 / 15)
        expected_count = 1_000_000_000 // step_ns + 1
        assert len(grid) == expected_count

    def test_non_zero_start(self):
        """Grid can start at arbitrary timestamp."""
        start = 5_000_000_000  # 5 seconds
        end = 6_000_000_000  # 6 seconds

        grid = make_time_grid(start_ns=start, end_ns=end, fps=10)

        assert grid[0] == start
        assert grid[-1] <= end


# =============================================================================
# Sample and Episode Tests
# =============================================================================


class TestSampleAndEpisode:
    """Tests for Sample and Episode data structures."""

    def test_sample_creation(self):
        """Test Sample creation and properties."""
        sample = create_test_sample(
            timestamp_ns=1000,
            action_dim=7,
            state_dim=14,
            camera_names=["wrist", "head"],
        )

        assert sample.timestamp_ns == 1000
        assert sample.action.shape == (7,)
        assert sample.state.shape == (14,)
        assert len(sample.images) == 2
        assert "wrist" in sample.images
        assert "head" in sample.images

    def test_sample_frame_index(self):
        """Test Sample frame_index property."""
        sample = create_test_sample()

        assert sample.frame_index is None

        sample.frame_index = 42
        assert sample.frame_index == 42

    def test_episode_creation(self):
        """Test Episode creation and properties."""
        episode = create_test_episode(
            segment_id="abc123",
            episode_index=5,
            num_samples=20,
            fps=15,
        )

        assert episode.segment_id == "abc123"
        assert episode.episode_index == 5
        assert episode.num_frames == 20
        assert len(episode.samples) == 20

        # Duration should be (num_samples - 1) / fps
        expected_duration = 19 / 15  # ~1.27 seconds
        assert abs(episode.duration_s - expected_duration) < 0.01

    def test_empty_episode(self):
        """Test empty Episode properties."""
        episode = Episode(
            segment_id="empty",
            episode_index=0,
            samples=[],
        )

        assert episode.num_frames == 0
        assert episode.duration_s == 0.0


# =============================================================================
# LeRobotHead Tests
# =============================================================================


class TestLeRobotHead:
    """Tests for LeRobot export head."""

    def test_format_name(self):
        """Test format name property."""
        config = ExportConfig(fps=15)
        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")
            assert head.format_name == "lerobot_v3"

    def test_infer_features(self):
        """Test feature inference from a sample."""
        config = ExportConfig(
            fps=15,
            cameras=[CameraSource(source="wrist")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")

            sample = create_test_sample(
                action_dim=7,
                state_dim=14,
                image_shape=(480, 640, 3),
                camera_names=["wrist"],
            )

            features = head.infer_features(sample)

            assert "action" in features
            assert features["action"]["shape"] == (7,)
            assert features["action"]["dtype"] == "float32"

            assert "observation.state" in features
            assert features["observation.state"]["shape"] == (14,)

            assert "observation.images.wrist" in features
            assert features["observation.images.wrist"]["shape"] == (480, 640, 3)
            assert features["observation.images.wrist"]["dtype"] == "video"

    def test_infer_features_no_state(self):
        """Test feature inference with empty state."""
        config = ExportConfig(fps=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")

            sample = Sample(
                timestamp_ns=0,
                action=np.zeros(7, dtype=np.float32),
                state=np.array([], dtype=np.float32),  # Empty state
                images={},
            )

            features = head.infer_features(sample)

            assert "action" in features
            assert "observation.state" not in features

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_initialize(self, mock_dataset_cls):
        """Test dataset initialization."""
        mock_dataset = MagicMock()
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(
            fps=15,
            dataset_id="test/dataset",
            cameras=[CameraSource(source="wrist")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            head = LeRobotHead(config, output_dir)

            features = {
                "action": {"dtype": "float32", "shape": (7,)},
                "observation.images.wrist": {"dtype": "video", "shape": (64, 64, 3)},
            }

            head.initialize(features)

            mock_dataset_cls.create.assert_called_once()
            call_kwargs = mock_dataset_cls.create.call_args.kwargs
            assert call_kwargs["repo_id"] == "test/dataset"
            assert call_kwargs["fps"] == 15
            assert call_kwargs["features"] == features
            assert call_kwargs["root"] == output_dir
            assert call_kwargs["use_videos"] is True
            assert call_kwargs["rgb_encoder"].vcodec == "h264"

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_write_episode(self, mock_dataset_cls):
        """Test episode writing."""
        mock_dataset = MagicMock()
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(fps=15, task_description="test_task")

        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")
            head.initialize({"action": {"dtype": "float32", "shape": (7,)}})

            episode = create_test_episode(num_samples=5)
            success = head.write_episode(episode)

            assert success
            assert mock_dataset.add_frame.call_count == 5
            mock_dataset.save_episode.assert_called_once()

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_write_empty_episode(self, mock_dataset_cls):
        """Test writing empty episode returns False."""
        mock_dataset = MagicMock()
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(fps=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            head = LeRobotHead(config, Path(tmpdir) / "output")
            head.initialize({"action": {"dtype": "float32", "shape": (7,)}})

            empty_episode = Episode(segment_id="empty", episode_index=0, samples=[])
            success = head.write_episode(empty_episode)

            assert not success
            mock_dataset.add_frame.assert_not_called()

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_finalize(self, mock_dataset_cls):
        """Test finalization returns correct result."""
        mock_dataset = MagicMock()
        mock_dataset.num_episodes = 3
        mock_dataset.num_frames = 45
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(fps=15, task_description="test_task")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            head = LeRobotHead(config, output_dir)
            head.initialize({"action": {"dtype": "float32", "shape": (7,)}})

            result = head.finalize()

            mock_dataset.finalize.assert_called_once()

            assert isinstance(result, ExportResult)
            assert result.output_dir == output_dir
            assert result.num_episodes == 3
            assert result.num_frames == 45
            assert result.format == "lerobot_v3"


# =============================================================================
# EpisodeSampler Tests (with mocked Rerun)
# =============================================================================


class TestEpisodeSamplerHelpers:
    """Tests for EpisodeSampler helper methods."""

    def test_build_samples_combines_data(self):
        """Test that _build_samples correctly combines video and action/state."""
        config = ExportConfig(
            fps=10,
            cameras=[CameraSource(source="wrist")],
        )

        # Create mock frame cache
        frames = [create_test_frame(value=i * 25) for i in range(5)]
        timestamps = np.array(
            [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000], dtype=np.int64
        )
        cache = FrameCache(frames=frames, timestamps_ns=timestamps)

        # Create mock action/state data
        action_state_data = {
            int(ts): {
                "action": np.array([float(i)], dtype=np.float32),
                "state": np.array([float(i) * 2], dtype=np.float32),
            }
            for i, ts in enumerate(timestamps)
        }

        # Mock the sampler (without Rerun)
        with patch.object(EpisodeSampler, "__init__", lambda self, **kwargs: None):
            sampler = EpisodeSampler.__new__(EpisodeSampler)
            sampler.config = config
            sampler._target_sizes = {"wrist": None}

            # Test _build_samples
            time_grid = timestamps
            frame_caches = {"wrist": cache}

            samples = sampler._build_samples(time_grid, frame_caches, action_state_data)

            assert len(samples) == 5

            for i, sample in enumerate(samples):
                assert sample.timestamp_ns == timestamps[i]
                assert sample.frame_index == i
                assert np.array_equal(sample.images["wrist"], frames[i])
                assert sample.action[0] == float(i)
                assert sample.state[0] == float(i) * 2


# =============================================================================
# Source validation (helpful error when a config source is missing)
# =============================================================================


class TestSourceValidation:
    """The preflight validator reports missing sources with what's available."""

    COLUMNS = [
        "/actions_target:Scalars:scalars",
        "/joint_positions:Scalars:scalars",
        "/gripper:Scalars:scalars",
        "/cam_flange:VideoStream:sample",
        "/cam_top:VideoStream:sample",
        "canonical_time",
    ]
    TIMELINES = ["canonical_time", "camera_time"]

    def test_extracts_source_names_by_suffix(self):
        from nova_export.export.exporter import _sources_for_suffix

        assert _sources_for_suffix(self.COLUMNS, ":Scalars:scalars") == [
            "actions_target",
            "gripper",
            "joint_positions",
        ]
        assert _sources_for_suffix(self.COLUMNS, ":VideoStream:sample") == [
            "cam_flange",
            "cam_top",
        ]

    def _signals(self):
        from nova_export.export.exporter import _sources_for_suffix

        return _sources_for_suffix(self.COLUMNS, ":Scalars:scalars")

    def _cameras(self):
        from nova_export.export.exporter import _sources_for_suffix

        return _sources_for_suffix(self.COLUMNS, ":VideoStream:sample")

    def test_valid_config_has_no_problems(self):
        from nova_export.export.exporter import _find_missing_sources

        config = ExportConfig(
            index_column="canonical_time",
            action=["actions_target"],
            state=["joint_positions", "gripper"],
            cameras=[CameraSource(source="cam_flange")],
        )
        assert (
            _find_missing_sources(
                config,
                timelines=self.TIMELINES,
                signals=self._signals(),
                cameras=self._cameras(),
            )
            == []
        )

    def test_missing_camera_lists_available(self):
        from nova_export.export.exporter import _find_missing_sources

        config = ExportConfig(
            index_column="canonical_time",
            action=["actions_target"],
            cameras=[CameraSource(source="cam_cobot_flange")],
        )
        problems = "\n".join(
            _find_missing_sources(
                config,
                timelines=self.TIMELINES,
                signals=self._signals(),
                cameras=self._cameras(),
            )
        )
        assert "cam_cobot_flange" in problems  # the bad name
        assert "cam_flange" in problems and "cam_top" in problems  # available ones

    def test_missing_signal_and_timeline_reported(self):
        from nova_export.export.exporter import _find_missing_sources

        config = ExportConfig(
            index_column="wallclock",  # not a real timeline
            action=["teleop"],  # not a real signal
            cameras=[CameraSource(source="cam_flange")],
        )
        problems = _find_missing_sources(
            config,
            timelines=self.TIMELINES,
            signals=self._signals(),
            cameras=self._cameras(),
        )
        joined = "\n".join(problems)
        assert "wallclock" in joined and "canonical_time" in joined  # timeline miss
        assert "teleop" in joined and "actions_target" in joined  # signal miss


# =============================================================================
# Max episode duration (raw-span safety check, independent of trimming)
# =============================================================================


class TestMaxEpisodeDuration:
    """`max_episode_duration_s` rejects segments by raw recording span.

    This is a plain upper-bound sanity check read from cheap dataset-manifest
    metadata (`dataset.get_index_ranges()`) — no query, no video decode, and
    unrelated to trimming. It exists to catch stuck/left-running recordings
    before they cost minutes of wasted video decoding.
    """

    def test_unset_by_default(self):
        assert ExportConfig().max_episode_duration_s is None

    def test_must_be_positive(self):
        with pytest.raises(ValidationError):
            ExportConfig(max_episode_duration_s=0)
        with pytest.raises(ValidationError):
            ExportConfig(max_episode_duration_s=-5)

    @staticmethod
    def _mock_index_ranges(start, end):
        """Build a mock DataFusion DataFrame matching get_index_ranges()'s chain."""
        table = pa.table({"canonical_time:start": [start], "canonical_time:end": [end]})
        mock = MagicMock()
        mock.filter.return_value.select.return_value.to_arrow_table.return_value = table
        return mock

    def test_reads_duration_from_index_ranges(self):
        from nova_export.export.exporter import _raw_segment_duration_s

        start = datetime(2026, 1, 1, 10, 0, 0)
        end = start + timedelta(seconds=42.5)
        index_ranges = self._mock_index_ranges(start, end)

        duration = _raw_segment_duration_s(index_ranges, "seg-1", "canonical_time")
        assert duration == pytest.approx(42.5)

    def test_segment_not_found_returns_none(self):
        from nova_export.export.exporter import _raw_segment_duration_s

        mock = MagicMock()
        mock.filter.return_value.select.return_value.to_arrow_table.return_value = (
            pa.table({"canonical_time:start": [], "canonical_time:end": []})
        )
        assert _raw_segment_duration_s(mock, "missing-seg", "canonical_time") is None

    def test_query_failure_returns_none(self):
        from nova_export.export.exporter import _raw_segment_duration_s

        mock = MagicMock()
        mock.filter.side_effect = RuntimeError("boom")
        assert _raw_segment_duration_s(mock, "seg-1", "canonical_time") is None

    def test_within_limit_is_kept(self):
        """End-to-end: a segment under the limit is processed, not skipped."""
        config = ExportConfig(
            fps=15,
            action=["actions_target"],
            cameras=[CameraSource(source="wrist")],
            max_episode_duration_s=60.0,
        )
        start = datetime(2026, 1, 1, 10, 0, 0)
        end = start + timedelta(seconds=10.0)  # well under the limit
        mock_dataset = MagicMock()
        mock_dataset.get_index_ranges.return_value = self._mock_index_ranges(
            start, end
        )
        mock_dataset.segment_ids.return_value = ["seg-1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            with (
                patch("nova_export.export.exporter.CatalogClient") as mock_client_cls,
                patch("nova_export.export.exporter._validate_sources"),
                patch(
                    "nova_export.export.episode_sampler.EpisodeSampler._process_segment",
                    return_value=create_test_episode(num_samples=5, fps=15),
                ) as mock_process,
                patch(
                    "lerobot.datasets.lerobot_dataset.LeRobotDataset"
                ) as mock_lerobot_cls,
            ):
                mock_client_cls.return_value.get_dataset.return_value = mock_dataset
                mock_lerobot_cls.create.return_value = MagicMock(
                    num_episodes=1, num_frames=5
                )

                result = export_recordings(
                    output_dir=output_dir,
                    config=config,
                    dataset_name="d",
                    catalog_url="http://fake",
                )

            mock_process.assert_called_once()
            assert result.num_episodes == 1

    def test_exceeding_limit_is_skipped_before_processing(self):
        """End-to-end: a segment over the limit never reaches _process_segment."""
        config = ExportConfig(
            fps=15,
            action=["actions_target"],
            cameras=[CameraSource(source="wrist")],
            max_episode_duration_s=60.0,
        )
        start = datetime(2026, 1, 1, 10, 0, 0)
        end = start + timedelta(seconds=4488.6)  # way over the limit
        mock_dataset = MagicMock()
        mock_dataset.get_index_ranges.return_value = self._mock_index_ranges(
            start, end
        )
        mock_dataset.segment_ids.return_value = ["seg-1"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            with (
                patch(
                    "nova_export.export.exporter.CatalogClient"
                ) as mock_client_cls,
                patch("nova_export.export.exporter._validate_sources"),
                patch(
                    "nova_export.export.episode_sampler.EpisodeSampler._process_segment"
                ) as mock_process,
            ):
                mock_client_cls.return_value.get_dataset.return_value = mock_dataset
                with pytest.raises(RuntimeError, match="exceeds max_episode_duration_s"):
                    export_recordings(
                        output_dir=output_dir,
                        config=config,
                        dataset_name="d",
                        catalog_url="http://fake",
                    )
                mock_process.assert_not_called()


# =============================================================================
# Camera Resize (configured via export config JSON)
# =============================================================================


class TestCameraResizeFromConfig:
    """Verify per-camera image resizing driven by the export config JSON.

    A CameraSource with `width`/`height` set resizes exported frames to those
    dimensions (via PyAV reformat); a camera without them keeps native size.
    """

    CONFIG_JSON = """
    {
        "format": "lerobot_v3",
        "fps": 10,
        "action": ["teleop"],
        "cameras": [
            { "source": "wrist", "width": 32, "height": 48 },
            { "source": "top" }
        ]
    }
    """

    def test_config_json_parses_camera_dimensions(self):
        """width/height in the JSON are parsed onto the CameraSource."""
        config = ExportConfig.model_validate_json(self.CONFIG_JSON)

        by_name = {c.source: c for c in config.cameras}
        assert (by_name["wrist"].width, by_name["wrist"].height) == (32, 48)
        assert by_name["top"].width is None and by_name["top"].height is None

    def test_sampler_builds_target_sizes_from_config(self):
        """The sampler turns configured dimensions into resize targets."""
        config = ExportConfig.model_validate_json(self.CONFIG_JSON)

        sampler = EpisodeSampler(config=config, dataset=MagicMock(), segment_ids=["seg"])

        # (width, height) target for the configured camera; None for the other.
        assert sampler._target_sizes["wrist"] == (32, 48)
        assert sampler._target_sizes["top"] is None

    def test_frames_resized_to_configured_dimensions(self):
        """Exported frames match the configured size; others keep native size."""
        config = ExportConfig.model_validate_json(self.CONFIG_JSON)
        sampler = EpisodeSampler(config=config, dataset=MagicMock(), segment_ids=["seg"])

        # Native frames are 64x64; "wrist" should be resized, "top" left alone.
        timestamps = np.array([0, 100_000_000], dtype=np.int64)
        native = [create_test_frame(height=64, width=64, value=90) for _ in timestamps]
        frame_caches = {
            "wrist": FrameCache(frames=list(native), timestamps_ns=timestamps),
            "top": FrameCache(frames=list(native), timestamps_ns=timestamps),
        }
        action_state_data = {
            int(ts): {
                "action": np.array([float(i)], dtype=np.float32),
                "state": np.array([], dtype=np.float32),
            }
            for i, ts in enumerate(timestamps)
        }

        samples = sampler._build_samples(timestamps, frame_caches, action_state_data)

        assert len(samples) == len(timestamps)
        for sample in samples:
            # numpy shape is (H, W, C); config is width=32, height=48.
            assert sample.images["wrist"].shape == (48, 32, 3)
            assert sample.images["top"].shape == (64, 64, 3)


# =============================================================================
# Integration Test (mocked LeRobot)
# =============================================================================


class TestIntegrationMocked:
    """Integration tests using mocked components."""

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_full_pipeline_mocked(self, mock_dataset_cls):
        """Test full pipeline with mocked Rerun and LeRobot."""
        mock_dataset = MagicMock()
        mock_dataset.num_episodes = 2
        mock_dataset.num_frames = 20
        mock_dataset_cls.create.return_value = mock_dataset

        config = ExportConfig(
            fps=15,
            cameras=[CameraSource(source="wrist")],
            task_description="pick_and_place",
        )

        # Create test episodes
        episodes = [
            create_test_episode(segment_id=f"seg-{i}", episode_index=i, num_samples=10)
            for i in range(2)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            # Create and initialize head
            head = LeRobotHead(config, output_dir)
            features = head.infer_features(episodes[0].samples[0])
            head.initialize(features)

            # Write episodes
            for episode in episodes:
                success = head.write_episode(episode)
                assert success

            # Finalize
            result = head.finalize()

            assert result.num_episodes == 2
            assert result.num_frames == 20


# =============================================================================
# Real Integration Test (with actual RRD data)
# =============================================================================


@pytest.mark.slow
class TestRealIntegration:
    """Integration tests using real RRD recording data.

    These tests require the test RRD file to exist.
    Run with: pytest -v --run-slow
    """

    @pytest.fixture
    def rrd_path(self) -> Path:
        """Get path to test RRD, skip if not available."""
        if not TEST_RRD_PATH.exists():
            pytest.skip(f"Test RRD not found: {TEST_RRD_PATH}")
        return TEST_RRD_PATH

    def test_video_decoder_with_real_rrd(self, rrd_path: Path):
        """Test VideoDecoder with real RRD recording."""
        import rerun as rr

        dataset_name = "test_decode"

        with rr.server.Server(datasets={dataset_name: [str(rrd_path)]}) as server:
            client = server.client()
            dataset = client.get_dataset(name=dataset_name)

            segment_ids = dataset.segment_ids()
            assert len(segment_ids) >= 1, "Expected at least 1 segment"

            segment_id = segment_ids[0]

            # Test decoding one camera
            decoder = VideoDecoder(codec="h264")
            cache = decoder.decode_segment(
                dataset=dataset,
                segment_id=segment_id,
                video_entity="/cam_flange",
                index_column="canonical_time",
            )

            # Should have decoded frames
            assert cache.num_frames > 0, "Expected decoded frames"
            assert cache.duration_s > 0, "Expected non-zero duration"

            # Frames should be valid images
            frame = cache.get_frame_at(cache.start_ns)
            assert frame is not None
            assert frame.ndim == 3  # HWC
            assert frame.shape[2] == 3  # RGB
            assert frame.dtype == np.uint8

    def test_episode_sampler_with_real_rrd(self, rrd_path: Path):
        """Test EpisodeSampler with real RRD recording."""
        import rerun as rr

        dataset_name = "test_sampler"

        config = ExportConfig(
            fps=15,
            cameras=[CameraSource(source="cam_flange")],
            action="commanded_joint_positions",
            state=["joint_positions"],
        )

        with rr.server.Server(datasets={dataset_name: [str(rrd_path)]}) as server:
            client = server.client()
            dataset = client.get_dataset(name=dataset_name)

            segment_ids = dataset.segment_ids()

            sampler = EpisodeSampler(
                config=config,
                dataset=dataset,
                segment_ids=segment_ids,
            )

            # Get first episode
            episodes = list(sampler.iterate_episodes())

            assert len(episodes) >= 1, "Expected at least 1 episode"

            episode = episodes[0]
            assert episode.num_frames > 0, "Expected samples in episode"
            assert episode.duration_s > 0, "Expected non-zero duration"

            # Check sample structure
            sample = episode.samples[0]
            assert sample.timestamp_ns > 0
            assert sample.action is not None
            assert sample.action.shape[0] > 0  # Has action dimensions
            assert "cam_flange" in sample.images

            # Image should be valid
            img = sample.images["cam_flange"]
            assert img.ndim == 3
            assert img.shape[2] == 3  # RGB

    @patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
    def test_full_export_with_real_rrd(self, mock_dataset_cls, rrd_path: Path):
        """Test full export pipeline with real RRD and mocked LeRobot."""
        import rerun as rr

        mock_dataset = MagicMock()
        mock_dataset.num_episodes = 1
        mock_dataset.num_frames = 0  # Will be counted
        mock_dataset_cls.create.return_value = mock_dataset

        dataset_name = "test_export"

        config = ExportConfig(
            fps=15,
            cameras=[CameraSource(source="cam_flange")],
            action="commanded_joint_positions",
            state=["joint_positions"],
            task_description="Put the red cube into the box.",
        )

        with rr.server.Server(datasets={dataset_name: [str(rrd_path)]}) as server:
            client = server.client()
            dataset = client.get_dataset(name=dataset_name)

            segment_ids = dataset.segment_ids()

            sampler = EpisodeSampler(
                config=config,
                dataset=dataset,
                segment_ids=segment_ids,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / "output"
                head = LeRobotHead(config, output_dir)

                # Process episodes
                first_episode = True
                total_frames = 0

                for episode in sampler.iterate_episodes():
                    if first_episode and episode.samples:
                        features = head.infer_features(episode.samples[0])
                        head.initialize(features)
                        first_episode = False

                    success = head.write_episode(episode)
                    if success:
                        total_frames += episode.num_frames

                # Verify LeRobot interactions
                assert mock_dataset.add_frame.call_count == total_frames
                assert mock_dataset.save_episode.call_count >= 1

                # Should have processed frames
                assert total_frames > 0, "Expected some frames to be exported"


def pytest_configure(config):
    """Register the 'slow' marker."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run with --run-slow)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is provided."""
    if config.getoption("--run-slow", default=False):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add --run-slow command line option."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
