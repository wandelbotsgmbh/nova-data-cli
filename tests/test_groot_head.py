"""Tests for the GR00T export head.

GR00T datasets are LeRobot datasets plus a ``meta/modality.json`` descriptor.
These tests focus on the GR00T-specific behavior — the modality descriptor and
the format wiring — and reuse the LeRobot writer (mocked) for the underlying
dataset.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nova_export.export.config import CameraSource, ExportConfig
from nova_export.export.exporter import _create_export_head
from nova_export.export.heads import GrootHead, LeRobotHead
from nova_export.export.heads.base import ExportResult


def _make_head(config: ExportConfig, output_dir: Path) -> GrootHead:
    """Build a GrootHead with a mocked, already-initialized LeRobot dataset."""
    head = GrootHead(config, output_dir)
    dataset = MagicMock()
    dataset.num_episodes = 2
    dataset.num_frames = 30
    head._dataset = dataset
    return head


def test_is_lerobot_subclass():
    """GR00T reuses the LeRobot writer."""
    assert issubclass(GrootHead, LeRobotHead)


def test_format_name():
    config = ExportConfig(format="groot")
    with tempfile.TemporaryDirectory() as tmpdir:
        head = GrootHead(config, Path(tmpdir) / "output")
        assert head.format_name == "groot"


def test_dispatch_creates_groot_head():
    """The exporter dispatch returns a GrootHead for format='groot'."""
    config = ExportConfig(format="groot", action=["teleop"])
    with tempfile.TemporaryDirectory() as tmpdir:
        head = _create_export_head(config, Path(tmpdir) / "output")
        assert isinstance(head, GrootHead)


def test_finalize_writes_modality_json():
    """finalize() writes meta/modality.json and returns a groot result."""
    config = ExportConfig(
        format="groot",
        action=["actions_target"],
        state=["joint_positions"],
        cameras=[CameraSource(source="cam_flange"), CameraSource(source="cam_top")],
        task_description="pick-and-place",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        head = _make_head(config, output_dir)
        head._features = {
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.images.cam_flange": {"dtype": "video", "shape": (240, 320, 3)},
            "observation.images.cam_top": {"dtype": "video", "shape": (240, 320, 3)},
        }

        result = head.finalize()

        # LeRobot dataset was finalized.
        head._dataset.finalize.assert_called_once()

        assert isinstance(result, ExportResult)
        assert result.format == "groot"
        assert result.num_episodes == 2
        assert result.num_frames == 30

        modality_path = output_dir / "meta" / "modality.json"
        assert modality_path.exists()

        modality = json.loads(modality_path.read_text())

        # Single source → keyed by source name, full range.
        assert modality["state"] == {"joint_positions": {"start": 0, "end": 7}}
        assert modality["action"] == {"actions_target": {"start": 0, "end": 7}}

        # One video entry per camera, mapped to its LeRobot column.
        assert modality["video"] == {
            "cam_flange": {"original_key": "observation.images.cam_flange"},
            "cam_top": {"original_key": "observation.images.cam_top"},
        }

        # Task annotation is sourced from LeRobot's task index.
        assert modality["annotation"] == {
            "human.task_description": {"original_key": "task_index"}
        }

        # Result metadata carries the modality descriptor.
        assert result.metadata is not None
        assert result.metadata["modality"] == modality


def test_modality_multi_source_uses_generic_key():
    """With multiple concatenated sources, a generic contiguous key is used."""
    config = ExportConfig(
        format="groot",
        action=["arm_joints", "gripper"],
        state=["joint_positions", "gripper"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        head = _make_head(config, output_dir)
        head._features = {
            "action": {"dtype": "float32", "shape": (8,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        }

        head.finalize()

        modality = json.loads((output_dir / "meta" / "modality.json").read_text())
        assert modality["state"] == {"state": {"start": 0, "end": 8}}
        assert modality["action"] == {"action": {"start": 0, "end": 8}}
        assert "video" not in modality


def test_modality_without_state():
    """Action-only datasets omit the state modality."""
    config = ExportConfig(format="groot", action=["teleop"])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        head = _make_head(config, output_dir)
        head._features = {
            "action": {"dtype": "float32", "shape": (6,), "names": None},
        }

        head.finalize()

        modality = json.loads((output_dir / "meta" / "modality.json").read_text())
        assert "state" not in modality
        assert modality["action"] == {"teleop": {"start": 0, "end": 6}}


def test_build_modality_requires_features():
    """Building the modality descriptor before inference is an error."""
    config = ExportConfig(format="groot", action=["teleop"])
    with tempfile.TemporaryDirectory() as tmpdir:
        head = GrootHead(config, Path(tmpdir) / "output")
        with pytest.raises(RuntimeError):
            head._build_modality_config()
