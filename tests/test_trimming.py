"""Tests for episode trimming configuration and logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from pydantic import ValidationError

from nova_export.export.config import ExportConfig, TrimmingConfig

# =============================================================================
# TrimmingConfig Validation Tests
# =============================================================================


class TestTrimmingConfigValidation:
    """Tests for TrimmingConfig Pydantic validation."""

    def test_default_mode(self):
        """Default mode is all_present with no source required."""
        cfg = TrimmingConfig()
        assert cfg.mode == "all_present"
        assert cfg.source is None
        assert cfg.threshold == 0.01
        assert cfg.tail_ms == 500

    def test_valid_all_present(self):
        """all_present mode doesn't require source."""
        cfg = TrimmingConfig(mode="all_present")
        assert cfg.source is None

    def test_valid_signal_presence(self):
        """signal_presence works with a source."""
        cfg = TrimmingConfig(mode="signal_presence", source="gripper")
        assert cfg.mode == "signal_presence"
        assert cfg.source == "gripper"

    def test_valid_signal_change(self):
        """signal_change works with all params."""
        cfg = TrimmingConfig(
            mode="signal_change",
            source="joint_states",
            threshold=0.05,
            tail_ms=1000,
        )
        assert cfg.mode == "signal_change"
        assert cfg.source == "joint_states"
        assert cfg.threshold == 0.05
        assert cfg.tail_ms == 1000

    def test_invalid_mode_rejected(self):
        """Invalid mode string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrimmingConfig(mode="garbage")
        errors = exc_info.value.errors()
        assert errors[0]["type"] == "literal_error"

    def test_signal_presence_requires_source(self):
        """signal_presence without source raises ValidationError."""
        with pytest.raises(ValidationError, match="'source' is required"):
            TrimmingConfig(mode="signal_presence")

    def test_signal_change_requires_source(self):
        """signal_change without source raises ValidationError."""
        with pytest.raises(ValidationError, match="'source' is required"):
            TrimmingConfig(mode="signal_change")

    def test_source_must_be_safe_token(self):
        """Source with spaces or special chars is rejected."""
        with pytest.raises(ValidationError, match="must contain only"):
            TrimmingConfig(mode="signal_presence", source="my source/invalid")

    def test_source_allows_dashes_underscores(self):
        """Source with dashes and underscores is valid."""
        cfg = TrimmingConfig(mode="signal_presence", source="joint_states-left")
        assert cfg.source == "joint_states-left"

    def test_threshold_must_be_positive(self):
        """Threshold <= 0 is rejected."""
        with pytest.raises(ValidationError):
            TrimmingConfig(mode="signal_change", source="x", threshold=0.0)
        with pytest.raises(ValidationError):
            TrimmingConfig(mode="signal_change", source="x", threshold=-1.0)

    def test_tail_ms_must_be_non_negative(self):
        """tail_ms < 0 is rejected."""
        with pytest.raises(ValidationError):
            TrimmingConfig(mode="signal_change", source="x", tail_ms=-100)

    def test_tail_ms_zero_is_valid(self):
        """tail_ms = 0 is valid (no buffer)."""
        cfg = TrimmingConfig(mode="signal_change", source="x", tail_ms=0)
        assert cfg.tail_ms == 0


class TestExportConfigTrimming:
    """Tests for trimming integration in ExportConfig."""

    def test_default_trimming(self):
        """ExportConfig has default TrimmingConfig."""
        config = ExportConfig(fps=15)
        assert config.trimming.mode == "all_present"

    def test_trimming_column_none_for_all_present(self):
        """trimming_column() returns None when no source."""
        config = ExportConfig(fps=15)
        assert config.trimming_column() is None

    def test_trimming_column_with_source(self):
        """trimming_column() returns fully-qualified column name."""
        config = ExportConfig(
            fps=15,
            trimming=TrimmingConfig(mode="signal_presence", source="gripper"),
        )
        assert config.trimming_column() == "/gripper:Scalars:scalars"

    def test_trimming_from_json(self):
        """TrimmingConfig round-trips through JSON."""
        config = ExportConfig(
            fps=15,
            trimming=TrimmingConfig(
                mode="signal_change", source="joint_states", threshold=0.05, tail_ms=200
            ),
        )
        json_str = config.model_dump_json()
        restored = ExportConfig.model_validate_json(json_str)
        assert restored.trimming.mode == "signal_change"
        assert restored.trimming.source == "joint_states"
        assert restored.trimming.threshold == 0.05
        assert restored.trimming.tail_ms == 200


# =============================================================================
# _apply_trimming Tests (mocked DataFusion)
# =============================================================================


def _make_arrow_timestamp_column(timestamps_ns: list[int]) -> pa.ChunkedArray:
    """Create a PyArrow chunked array of timestamps."""
    return pa.chunked_array([pa.array(timestamps_ns, type=pa.int64())])


def _make_arrow_scalar_column(values: list[float]) -> pa.ChunkedArray:
    """Create a PyArrow chunked array of scalar values."""
    return pa.chunked_array([pa.array(values, type=pa.float64())])


def _make_arrow_list_column(values: list[list[float]]) -> pa.ChunkedArray:
    """Create a PyArrow chunked array of list (vector) values."""
    return pa.chunked_array([pa.array(values, type=pa.list_(pa.float64()))])


def _build_sampler_with_config(trimming: TrimmingConfig) -> MagicMock:
    """Create a mock EpisodeSampler with the given trimming config."""
    from nova_export.export.episode_sampler import EpisodeSampler

    with patch.object(EpisodeSampler, "__init__", lambda self, **kw: None):
        sampler = EpisodeSampler()

    sampler.config = ExportConfig(fps=15, trimming=trimming)
    sampler.dataset = MagicMock()
    return sampler


def _setup_mock_query(sampler, timestamps_ns, col_name, col_data):
    """Wire up the mock dataset to return a table with given data."""
    table = pa.table(
        {
            sampler.config.index_column: timestamps_ns,
            col_name: col_data,
        }
    )
    mock_view = MagicMock()
    sampler.dataset.filter_segments.return_value = mock_view

    mock_reader = MagicMock()
    mock_view.reader.return_value = mock_reader
    mock_reader.filter.return_value = mock_reader
    mock_reader.select.return_value = mock_reader
    mock_reader.to_arrow_table.return_value = table


class TestApplyTrimmingAllPresent:
    """Tests for _apply_trimming with mode=all_present."""

    def test_noop(self):
        """all_present mode returns raw bounds unchanged."""
        sampler = _build_sampler_with_config(TrimmingConfig())
        result = sampler._apply_trimming("seg123", 1000, 9000)
        assert result == (1000, 9000)


class TestApplyTrimmingSignalPresence:
    """Tests for _apply_trimming with mode=signal_presence."""

    def test_narrows_to_source_bounds(self):
        """Trims to first/last sample of the named source."""
        trimming = TrimmingConfig(mode="signal_presence", source="gripper")
        sampler = _build_sampler_with_config(trimming)

        col_name = "/gripper:Scalars:scalars"
        timestamps = [2000, 3000, 4000, 5000, 6000]

        _setup_mock_query(
            sampler,
            timestamps,
            col_name,
            [1.0, 0.5, 0.3, 0.8, 1.0],
        )

        result = sampler._apply_trimming("seg123", 1000, 9000)
        assert result == (2000, 6000)

    def test_no_data_skips_trim(self):
        """If source has no samples, returns raw bounds (no narrowing)."""
        trimming = TrimmingConfig(mode="signal_presence", source="gripper")
        sampler = _build_sampler_with_config(trimming)

        # Return empty table
        empty_table = pa.table(
            {sampler.config.index_column: pa.array([], type=pa.int64())}
        )
        mock_view = MagicMock()
        sampler.dataset.filter_segments.return_value = mock_view
        mock_reader = MagicMock()
        mock_view.reader.return_value = mock_reader
        mock_reader.filter.return_value = mock_reader
        mock_reader.select.return_value = mock_reader
        mock_reader.to_arrow_table.return_value = empty_table

        result = sampler._apply_trimming("seg123", 1000, 9000)
        assert result == (1000, 9000)


class TestApplyTrimmingSignalChange:
    """Tests for _apply_trimming with mode=signal_change."""

    def test_scalar_signal_basic(self):
        """Trims to first/last change for scalar signal."""
        trimming = TrimmingConfig(
            mode="signal_change", source="gripper", threshold=0.01, tail_ms=0
        )
        sampler = _build_sampler_with_config(trimming)

        col_name = "/gripper:Scalars:scalars"
        # Signal: flat, then changes, then flat again
        # Timestamps: 0, 100, 200, 300, 400, 500, 600 ms (in ns)
        timestamps = [int(i * 100e6) for i in range(7)]
        values = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]

        _setup_mock_query(sampler, timestamps, col_name, values)

        result = sampler._apply_trimming("seg123", 0, int(600e6))
        # First diff > threshold at index 1 (0.0 -> 0.5): trim_start = timestamps[1] = 100ms
        # Last diff > threshold at index 2 (0.5 -> 1.0): trim_end = timestamps[3] + 0ms tail
        assert result[0] == int(100e6)
        assert result[1] == int(300e6)

    def test_scalar_signal_with_tail(self):
        """tail_ms adds buffer after last change."""
        trimming = TrimmingConfig(
            mode="signal_change", source="gripper", threshold=0.01, tail_ms=200
        )
        sampler = _build_sampler_with_config(trimming)

        col_name = "/gripper:Scalars:scalars"
        timestamps = [int(i * 100e6) for i in range(7)]
        values = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]

        _setup_mock_query(sampler, timestamps, col_name, values)

        result = sampler._apply_trimming("seg123", 0, int(600e6))
        # Same start as above: 100ms
        # End: timestamps[3] + 200ms tail = 300ms + 200ms = 500ms
        assert result[0] == int(100e6)
        assert result[1] == int(300e6) + 200 * 1_000_000

    def test_vector_signal_linf_norm(self):
        """Vector signal uses L∞ norm for change detection."""
        trimming = TrimmingConfig(
            mode="signal_change", source="joint_states", threshold=0.05, tail_ms=0
        )
        sampler = _build_sampler_with_config(trimming)

        col_name = "/joint_states:Scalars:scalars"
        timestamps = [int(i * 100e6) for i in range(5)]
        # 3-DOF joint positions: flat, then one joint moves significantly
        values = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # no change
            [0.0, 0.1, 0.0],  # joint 2 moves by 0.1 > threshold
            [0.0, 0.1, 0.0],  # no change
            [0.0, 0.1, 0.0],  # no change
        ]

        _setup_mock_query(sampler, timestamps, col_name, values)

        result = sampler._apply_trimming("seg123", 0, int(400e6))
        # Change at diff index 1 (rows 1->2): trim_start = timestamps[1] = 100ms
        # Last change at diff index 1: trim_end = timestamps[2] = 200ms (no tail)
        assert result[0] == int(100e6)
        assert result[1] == int(200e6)

    def test_no_activity_skips_trim(self):
        """If signal is flat (no diffs exceed threshold), returns raw bounds."""
        trimming = TrimmingConfig(
            mode="signal_change", source="gripper", threshold=1.0, tail_ms=0
        )
        sampler = _build_sampler_with_config(trimming)

        col_name = "/gripper:Scalars:scalars"
        timestamps = [int(i * 100e6) for i in range(5)]
        values = [0.5, 0.5, 0.5, 0.5, 0.5]  # constant

        _setup_mock_query(sampler, timestamps, col_name, values)

        result = sampler._apply_trimming("seg123", 0, int(400e6))
        assert result == (0, int(400e6))

    def test_too_few_samples_skips_trim(self):
        """If source has <2 samples, returns raw bounds."""
        trimming = TrimmingConfig(
            mode="signal_change", source="gripper", threshold=0.01, tail_ms=0
        )
        sampler = _build_sampler_with_config(trimming)

        col_name = "/gripper:Scalars:scalars"
        # Only 1 sample
        table = pa.table(
            {
                sampler.config.index_column: [int(100e6)],
                col_name: [0.5],
            }
        )
        mock_view = MagicMock()
        sampler.dataset.filter_segments.return_value = mock_view
        mock_reader = MagicMock()
        mock_view.reader.return_value = mock_reader
        mock_reader.filter.return_value = mock_reader
        mock_reader.select.return_value = mock_reader
        mock_reader.to_arrow_table.return_value = table

        result = sampler._apply_trimming("seg123", 0, int(400e6))
        assert result == (0, int(400e6))

    def test_below_threshold_ignored(self):
        """Small changes below threshold don't count as activity."""
        trimming = TrimmingConfig(
            mode="signal_change", source="gripper", threshold=0.1, tail_ms=0
        )
        sampler = _build_sampler_with_config(trimming)

        col_name = "/gripper:Scalars:scalars"
        timestamps = [int(i * 100e6) for i in range(6)]
        # Small jitter (0.01) then real change (0.5) then flat
        values = [0.0, 0.01, 0.02, 0.5, 0.5, 0.5]

        _setup_mock_query(sampler, timestamps, col_name, values)

        result = sampler._apply_trimming("seg123", 0, int(500e6))
        # Only diff index 2 (0.02 -> 0.5 = 0.48) exceeds threshold 0.1
        assert result[0] == int(200e6)
        assert result[1] == int(300e6)
