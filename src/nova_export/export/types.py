"""Internal data types for the export pipeline."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class FeatureSpec(TypedDict):
    """Typed feature specification for LeRobot datasets."""

    dtype: str
    shape: tuple[int, ...]
    names: list[str] | None


class VideoSpec(TypedDict):
    """Specification for a video stream in the dataset."""

    key: str
    path: str
    width: NotRequired[int | None]
    height: NotRequired[int | None]
