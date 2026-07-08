"""LeRobot export head — writes episodes to LeRobot v3 dataset format.

Handles:
- Feature schema from Sample objects
- LeRobot dataset creation
- Frame serialization (action, state, images)
- Video encoding (via LeRobot's internals)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from tqdm import tqdm

from nova_export.export.heads.base import ExportHead, ExportResult

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from nova_export.export.config import ExportConfig
    from nova_export.export.episode_sampler import Episode, Sample


class LeRobotHead(ExportHead):
    """Export head for LeRobot v3 dataset format.

    Transforms Episode/Sample objects into LeRobot's Parquet + MP4 format.
    """

    def __init__(self, config: ExportConfig, output_dir: Path):
        """Initialize the LeRobot export head.

        Args:
            config: Export configuration.
            output_dir: Directory to write the dataset.
        """
        super().__init__(config, output_dir)
        self._dataset: LeRobotDataset | None = None
        self._features: dict[str, Any] | None = None

    @property
    def format_name(self) -> str:
        return "lerobot_v3"

    def infer_features(self, sample: Sample) -> dict[str, Any]:
        """Infer feature schema from a sample.

        Args:
            sample: Representative sample to infer shapes from.

        Returns:
            LeRobot feature specification dict.
        """
        features: dict[str, Any] = {}

        # Action feature
        action_dim = len(sample.action)
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        }

        # State feature
        if len(sample.state) > 0:
            state_dim = len(sample.state)
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": None,
            }

        # Image features
        use_videos = bool(self.config.cameras)
        for cam_name, img_array in sample.images.items():
            h, w, c = img_array.shape
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video" if use_videos else "image",
                "shape": (h, w, c),
                "names": ["height", "width", "channels"],
            }

        logger.info(
            "Inferred features: action={}, state={}, images={}",
            features["action"]["shape"],
            features.get("observation.state", {}).get("shape", "(none)"),
            [
                f"{k.split('.')[-1]}:{v['shape']}"
                for k, v in features.items()
                if "images" in k
            ],
        )

        return features

    def initialize(self, features: dict[str, Any]) -> None:
        """Initialize the LeRobot dataset with inferred features.

        Args:
            features: Feature schema from infer_features().
        """
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._features = features
        use_videos = any(f.get("dtype") == "video" for f in features.values())

        logger.info(
            "Creating LeRobot dataset: repo_id={}, fps={}, use_videos={}",
            self.config.dataset_id,
            self.config.fps,
            use_videos,
        )

        self._dataset = LeRobotDataset.create(
            repo_id=self.config.dataset_id,
            fps=self.config.fps,
            features=features,
            root=self.output_dir,
            use_videos=use_videos,
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        )

    def write_episode(self, episode: Episode) -> bool:
        """Write an episode to the dataset.

        Args:
            episode: Episode to write.

        Returns:
            True if successfully written.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not initialized. Call initialize() first.")

        if not episode.samples:
            logger.warning("Skipping empty episode {}", episode.episode_index)
            return False

        logger.info(
            "Writing episode {}: {} frames, {:.2f}s duration",
            episode.episode_index,
            episode.num_frames,
            episode.duration_s,
        )

        try:
            for sample in tqdm(episode.samples, desc="Frames", leave=False):
                frame = self._sample_to_frame(sample)
                self._dataset.add_frame(frame)

            self._dataset.save_episode()
            self._update_counts(episode)
            return True

        except Exception as e:
            logger.error("Error writing episode {}: {}", episode.episode_index, e)
            return False

    def finalize(self) -> ExportResult:
        """Finalize the dataset and return results.

        Returns:
            ExportResult with output path and statistics.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not initialized.")

        logger.info("Finalizing LeRobot dataset...")
        self._dataset.finalize()

        logger.success(
            "Dataset finalized: {} episodes, {} frames → {}",
            self._dataset.num_episodes,
            self._dataset.num_frames,
            self.output_dir,
        )

        return ExportResult(
            output_dir=self.output_dir,
            num_episodes=self._dataset.num_episodes,
            num_frames=self._dataset.num_frames,
            format=self.format_name,
            metadata={
                "fps": self.config.fps,
                "task": self.config.task_description,
                "features": list(self._features.keys()) if self._features else [],
            },
        )

    def _sample_to_frame(self, sample: Sample) -> dict[str, Any]:
        """Convert a Sample to a LeRobot frame dict.

        Args:
            sample: Sample to convert.

        Returns:
            Frame dict for LeRobotDataset.add_frame().
        """
        frame: dict[str, Any] = {}

        # Action
        frame["action"] = sample.action

        # State
        if len(sample.state) > 0:
            frame["observation.state"] = sample.state

        # Task
        frame["task"] = self.config.task_description

        # Images
        for cam_name, img_array in sample.images.items():
            frame[f"observation.images.{cam_name}"] = img_array

        return frame
