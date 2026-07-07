"""Abstract base class for export heads.

Export heads are format-specific writers that transform Episode/Sample objects
into target dataset formats. This enables a clean separation:

- EpisodeSampler: Handles resampling and produces format-agnostic Samples
- ExportHead: Handles format-specific serialization

To add a new format:
1. Subclass ExportHead
2. Implement infer_features(), write_episode(), and finalize()
3. Register in the heads __init__.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nova_export.export.config import ExportConfig
    from nova_export.export.episode_sampler import Episode, Sample


@dataclass
class ExportResult:
    """Result of an export operation."""

    output_dir: Path
    num_episodes: int
    num_frames: int
    format: str
    metadata: dict[str, Any] | None = None


class ExportHead(ABC):
    """Abstract base class for format-specific export heads.

    Subclasses implement the details of writing to specific formats
    (LeRobot, Groot, HuggingFace, etc.).
    """

    def __init__(self, config: ExportConfig, output_dir: Path):
        """Initialize the export head.

        Args:
            config: Export configuration.
            output_dir: Directory to write output files.
        """
        self.config = config
        self.output_dir = output_dir
        self._num_episodes = 0
        self._num_frames = 0

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name (e.g., 'lerobot_v3', 'groot')."""
        ...

    @abstractmethod
    def infer_features(self, sample: Sample) -> dict[str, Any]:
        """Infer the feature schema from a sample.

        Called once before writing to determine shapes and dtypes.

        Args:
            sample: A representative sample to infer schema from.

        Returns:
            Format-specific feature dictionary.
        """
        ...

    @abstractmethod
    def initialize(self, features: dict[str, Any]) -> None:
        """Initialize the dataset with inferred features.

        Called once after infer_features() before writing episodes.

        Args:
            features: Feature schema from infer_features().
        """
        ...

    @abstractmethod
    def write_episode(self, episode: Episode) -> bool:
        """Write a single episode to the dataset.

        Args:
            episode: Episode to write.

        Returns:
            True if successfully written, False if skipped.
        """
        ...

    @abstractmethod
    def finalize(self) -> ExportResult:
        """Finalize the dataset and return export result.

        Called after all episodes have been written. Handles:
        - Writing metadata
        - Computing statistics
        - Encoding videos (if deferred)

        Returns:
            ExportResult with output path and statistics.
        """
        ...

    def _update_counts(self, episode: Episode) -> None:
        """Update internal counters after writing an episode."""
        self._num_episodes += 1
        self._num_frames += episode.num_frames
