"""Export configuration models.

Describes what data to export from recordings in a format-agnostic way.
The same config drives all exporters (LeRobot, Groot, etc.).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from typing import Self

    from nova_export.export.types import VideoSpec

ExportFormat = Literal["lerobot_v3", "groot"]

# Rerun DataFusion column suffix for scalar arrays logged with rr.Scalars
_SCALARS_SUFFIX = ":Scalars:scalars"
# Rerun DataFusion column suffix for video streams
_VIDEO_SUFFIX = ":VideoStream:sample"
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_safe_token(value: str, field_name: str) -> str:
    if not _SAFE_TOKEN_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, '-' or '_', with no spaces"
        )
    return value


class TrimmingConfig(BaseModel):
    """Controls how episode time bounds are trimmed during export."""

    mode: Literal["all_present", "signal_presence", "signal_change"] = Field(
        default="all_present",
        description=(
            "all_present: intersection of all configured streams (default). "
            "signal_presence: trim to first/last sample of a named source. "
            "signal_change: trim to first/last detected change in a signal."
        ),
    )
    source: str | None = Field(
        default=None,
        description="Source name to use for trimming (required for signal_presence and signal_change)",
    )
    threshold: float = Field(
        default=0.01,
        gt=0,
        description="Minimum change magnitude to count as activity (signal_change only). L∞ norm for vectors.",
    )
    tail_ms: int = Field(
        default=500,
        ge=0,
        description="Buffer in ms after last detected change before marking end-of-episode (signal_change only)",
    )

    @model_validator(mode="after")
    def _validate_source_required(self) -> Self:
        if self.mode in ("signal_presence", "signal_change") and not self.source:
            raise ValueError(f"'source' is required for trimming mode '{self.mode}'")
        return self

    @field_validator("source")
    @classmethod
    def _validate_source_token(cls, v: str | None) -> str | None:
        if v is not None:
            _validate_safe_token(v, "trimming.source")
        return v


class CameraSource(BaseModel):
    """A camera stream to include as an observation image."""

    source: str = Field(description="Recording source name (e.g. 'wrist')")
    width: int | None = Field(
        default=None,
        strict=True,
        description="Output width in pixels. If set, frames are resized. Defaults to original.",
    )
    height: int | None = Field(
        default=None,
        strict=True,
        description="Output height in pixels. If set, frames are resized. Defaults to original.",
    )

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        return _validate_safe_token(v, "source")

    @field_validator("width", "height")
    @classmethod
    def _validate_dimensions(cls, v: int | None, info) -> int | None:
        if v is None:
            return v
        if v <= 1:
            raise ValueError(f"{info.field_name} must be greater than 1")
        return v


class ExportConfig(BaseModel):
    """Format-agnostic configuration for exporting recordings to a training dataset."""

    version: int = 1
    format: ExportFormat = Field(
        default="lerobot_v3", description="Target dataset format"
    )
    fps: int = Field(
        default=15,
        strict=True,
        gt=1,
        le=250,
        description="Target fixed frame rate",
    )
    index_column: str = Field(
        default="canonical_time",
        description="Rerun timeline used to build the fixed-FPS time grid",
    )

    action: list[str] = Field(
        default_factory=list,
        description="Recording source names for actions, concatenated in order into a single action vector (e.g. ['teleop'] or ['arm_joints', 'gripper'])",
    )

    state: list[str] = Field(
        default_factory=list,
        description="Recording source names for state observations, concatenated in order",
    )

    cameras: list[CameraSource] = Field(
        default_factory=list,
        description="Camera sources to include as observation images",
    )

    trimming: TrimmingConfig = Field(
        default_factory=TrimmingConfig,
        description="Episode trimming configuration (controls start/end bounds)",
    )

    max_episode_duration_s: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Reject a segment outright if its raw recording span exceeds this "
            "many seconds — an upper-bound safety check for stuck or "
            "left-running recordings, independent of trimming. Checked cheaply "
            "via dataset metadata before any video decode. Unset means no limit."
        ),
    )

    task_description: str = Field(
        default="task",
        description="Task label written to the dataset",
    )

    dataset_id: str = Field(
        default="nova/dataset",
        description="Dataset identifier (repo_id for LeRobot, dataset name for Groot)",
    )

    @field_validator("index_column")
    @classmethod
    def _validate_index_column(cls, v: str) -> str:
        return _validate_safe_token(v, "index_column")

    @field_validator("action", "state", mode="before")
    @classmethod
    def _coerce_source_list(cls, v: object) -> object:
        """Accept a single source name (str) or a list; normalize to a list.

        Keeps backward compatibility with configs that set `action` as a string.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return v

    @field_validator("action", "state")
    @classmethod
    def _validate_sources(cls, values: list[str], info) -> list[str]:
        for value in values:
            _validate_safe_token(value, info.field_name)
        return values

    # --- Internal helpers (used by the export pipeline, not user-facing) ---

    def _normalize_path(self, name: str) -> str:
        """Ensure entity path has leading slash for DataFusion queries."""
        return name if name.startswith("/") else f"/{name}"

    def action_columns(self) -> list[str]:
        """Fully-qualified DataFusion column names for each action source, in order."""
        return [self._normalize_path(s) + _SCALARS_SUFFIX for s in self.action]

    def state_columns(self) -> list[str]:
        """Fully-qualified DataFusion column names for each state source, in order."""
        return [self._normalize_path(s) + _SCALARS_SUFFIX for s in self.state]

    def video_sample_columns(self) -> list[str]:
        """Fully-qualified DataFusion column names for video streams."""
        return [self._normalize_path(c.source) + _VIDEO_SUFFIX for c in self.cameras]

    def trimming_column(self) -> str | None:
        """Fully-qualified DataFusion column name for the trimming source, or None."""
        if self.trimming.source:
            return self._normalize_path(self.trimming.source) + _SCALARS_SUFFIX
        return None

    def get_filter_list(self) -> list[str]:
        """Entity paths to include when querying the rerun catalog."""
        contents: list[str] = []
        for s in self.action:
            path = self._normalize_path(s)
            if path not in contents:
                contents.append(path)
        for s in self.state:
            path = self._normalize_path(s)
            if path not in contents:
                contents.append(path)
        for c in self.cameras:
            path = self._normalize_path(c.source)
            if path not in contents:
                contents.append(path)
        return contents

    def to_video_specs(self) -> list[VideoSpec]:
        """Convert camera sources to the internal VideoSpec format."""
        return [
            {
                "key": c.source,
                "path": self._normalize_path(c.source),
                "width": c.width,
                "height": c.height,
            }
            for c in self.cameras
        ]
