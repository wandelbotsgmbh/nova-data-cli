"""Raw multi-modality export head — per-episode, per-camera, unstitched video files.

Unlike LeRobotHead/GrootHead (one shared dataset object, episodes concatenated
into shared Parquet/MP4 shards for training-loop random access), this head
writes each episode as its own self-contained folder:

    <output_dir>/<recording_id>/<camera>/<modality>.mp4
    <output_dir>/<recording_id>/actions.parquet
    <output_dir>/<recording_id>/meta.json

Intended for handoff to per-episode video tooling (e.g. Cosmos Transfer),
which wants one video file per camera/modality, addressable by the episode's
own recording ID, not baked into a cross-episode training dataset shard.
`<camera>`/`<modality>` are derived from the configured source name using the
same suffix convention nova-data-collection's camera pipeline already uses
(`cam_top`, `cam_top_depth`, `cam_top_canny`, `cam_top_segmentation`, ...) —
no new config fields needed.

Every modality (color, depth, canny, segmentation) arrives as an ordinary
Rerun VideoStream entity (see this repo's export-guide.md), so
EpisodeSampler/VideoDecoder decode all of them identically already —
`Sample.images` is already a generic {camera_name: RGB array} dict per frame.
This head only adds the write-out side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import av
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nova_export.export.heads.base import ExportHead, ExportResult

if TYPE_CHECKING:
    from nova_export.export.config import ExportConfig
    from nova_export.export.episode_sampler import Episode, Sample

# Suffix -> modality name. No suffix match = "rgb". Mirrors the entity-naming
# convention nova-data-collection's camera channel pipeline already uses.
_MODALITY_SUFFIXES: dict[str, str] = {
    "_depth": "depth",
    "_canny": "canny",
    "_segmentation": "segmentation",
}


def _split_camera_modality(source_name: str) -> tuple[str, str]:
    """Derive (physical_camera, modality) from a configured camera source name."""
    for suffix, modality in _MODALITY_SUFFIXES.items():
        if source_name.endswith(suffix):
            return source_name[: -len(suffix)], modality
    return source_name, "rgb"


def _encode_video(
    frames: list[npt.NDArray[np.uint8]], fps: int, path: Path
) -> None:
    """Encode a sequence of HWC RGB frames to an H.264 mp4 file.

    All modalities land here already lossy at the collector (see
    nova-data-collection's camera.py stream handlers) -- re-encoding
    lossless on top of an already-lossy source can't recover anything, it
    only bloats the file. crf=18/yuv420p matches the collector's own lossy
    setting, measured to add only small additional degradation on top of an
    already-lossy source (canny mean edge-pixel IoU 0.97-0.997 across a full
    episode on all 3 cameras, no visible dashing; depth MAE 0.24/255;
    segmentation MAE 0.85/255)."""
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}

        for frame_arr in frames:
            vf = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frame_arr), format="rgb24"
            )
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


class RawMultimodalHead(ExportHead):
    """Export head that writes one unstitched folder of files per episode."""

    @property
    def format_name(self) -> str:
        return "raw_multimodal"

    def infer_features(self, sample: Sample) -> dict[str, Any]:
        # No shared dataset schema to infer — every episode is self-contained.
        return {}

    def initialize(self, features: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_episode(self, episode: Episode) -> bool:
        if not episode.samples:
            logger.warning("Skipping empty episode {}", episode.episode_index)
            return False

        episode_dir = self.output_dir / episode.segment_id

        camera_names = list(episode.samples[0].images.keys())
        for camera_name in camera_names:
            physical_camera, modality = _split_camera_modality(camera_name)
            frames = [s.images[camera_name] for s in episode.samples]
            video_path = episode_dir / physical_camera / f"{modality}.mp4"
            _encode_video(frames, fps=self.config.fps, path=video_path)

        self._write_actions(episode, episode_dir)
        self._write_meta(episode, episode_dir, camera_names)

        self._update_counts(episode)
        return True

    def finalize(self) -> ExportResult:
        logger.success(
            "Raw multimodal export finalized: {} episodes, {} frames → {}",
            self._num_episodes,
            self._num_frames,
            self.output_dir,
        )
        return ExportResult(
            output_dir=self.output_dir,
            num_episodes=self._num_episodes,
            num_frames=self._num_frames,
            format=self.format_name,
            metadata={"fps": self.config.fps},
        )

    def _write_actions(self, episode: Episode, episode_dir: Path) -> None:
        table = pa.table(
            {
                "timestamp_ns": pa.array(
                    [s.timestamp_ns for s in episode.samples], type=pa.int64()
                ),
                "frame_index": pa.array(
                    [s.frame_index for s in episode.samples], type=pa.int32()
                ),
                "action": pa.array(
                    [s.action.tolist() for s in episode.samples], type=pa.list_(pa.float32())
                ),
                "state": pa.array(
                    [s.state.tolist() for s in episode.samples], type=pa.list_(pa.float32())
                ),
            }
        )
        episode_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, episode_dir / "actions.parquet")

    def _write_meta(
        self, episode: Episode, episode_dir: Path, camera_names: list[str]
    ) -> None:
        cameras: dict[str, list[str]] = {}
        for camera_name in camera_names:
            physical_camera, modality = _split_camera_modality(camera_name)
            cameras.setdefault(physical_camera, []).append(modality)

        meta = {
            "recording_id": episode.segment_id,
            "episode_index": episode.episode_index,
            "fps": self.config.fps,
            "num_frames": episode.num_frames,
            "duration_s": episode.duration_s,
            "task": episode.task if episode.task is not None else self.config.task_description,
            "cameras": cameras,
            "extra_metadata": episode.extra_metadata or {},
        }
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
