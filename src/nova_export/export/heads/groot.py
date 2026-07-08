"""GR00T export head — writes episodes in NVIDIA Isaac GR00T format.

The GR00T (Isaac-GR00T) dataset format is the LeRobot dataset layout plus a
`meta/modality.json` descriptor. The modality file tells GR00T how to slice the
flat `observation.state` / `action` vectors into named modality groups and which
video / annotation columns to use.

This head therefore reuses the LeRobot writer for the heavy lifting (Parquet +
MP4 + LeRobot metadata) and only adds the `modality.json` descriptor during
finalization.

Version note: this writes a LeRobot **v3.0** dataset (what `lerobot>=0.6`
produces) plus `modality.json`. GR00T ingests LeRobot **v2.1**, so the export is
a two-step flow — run this head, then convert with the standalone, separately
pinned tool in `tools/groot_lerobot_conversion/` (it preserves the
`modality.json` written here). See that directory's README.

Reference: https://github.com/NVIDIA/Isaac-GR00T (LeRobot-compatible datasets).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from nova_export.export.heads.lerobot import LeRobotHead

if TYPE_CHECKING:
    from nova_export.export.heads.base import ExportResult

_IMAGE_PREFIX = "observation.images."


class GrootHead(LeRobotHead):
    """Export head for the NVIDIA Isaac GR00T dataset format.

    Produces a LeRobot dataset (via :class:`LeRobotHead`) and augments it with a
    ``meta/modality.json`` file describing the state, action, video, and
    annotation modalities that GR00T expects.
    """

    @property
    def format_name(self) -> str:
        return "groot"

    def finalize(self) -> ExportResult:
        """Finalize the LeRobot dataset and write GR00T's modality descriptor.

        Returns:
            ExportResult with output path, statistics, and modality metadata.
        """
        # Write the LeRobot dataset first. Because ``format_name`` is overridden,
        # the returned result already carries format="groot".
        result = super().finalize()

        modality = self._build_modality_config()
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        modality_path = meta_dir / "modality.json"
        modality_path.write_text(json.dumps(modality, indent=2) + "\n")

        logger.success("Wrote GR00T modality descriptor → {}", modality_path)

        if result.metadata is not None:
            result.metadata["modality"] = modality
        return result

    def _build_modality_config(self) -> dict[str, Any]:
        """Build the ``modality.json`` contents from the inferred feature schema.

        The flat ``observation.state`` and ``action`` columns are exposed as a
        single contiguous modality group each. When a single source feeds a
        modality the group is named after that source; otherwise a generic
        ``state`` / ``action`` key spanning the full vector is used (per-source
        splitting is not derivable from the concatenated vector alone).

        Returns:
            Modality descriptor dict ready to be serialized to JSON.
        """
        if self._features is None:
            raise RuntimeError("Features not inferred; call infer_features() first.")

        modality: dict[str, Any] = {}

        # State modality (optional — some datasets are action-only).
        state_feature = self._features.get("observation.state")
        if state_feature is not None:
            state_dim = int(state_feature["shape"][0])
            state_key = self.config.state[0] if len(self.config.state) == 1 else "state"
            modality["state"] = {state_key: {"start": 0, "end": state_dim}}

        # Action modality (always present).
        action_dim = int(self._features["action"]["shape"][0])
        action_key = (
            self.config.action[0] if len(self.config.action) == 1 else "action"
        )
        modality["action"] = {action_key: {"start": 0, "end": action_dim}}

        # Video modality — one entry per camera, mapping to its LeRobot column.
        video: dict[str, Any] = {}
        for feature_key in self._features:
            if feature_key.startswith(_IMAGE_PREFIX):
                cam_name = feature_key[len(_IMAGE_PREFIX) :]
                video[cam_name] = {"original_key": feature_key}
        if video:
            modality["video"] = video

        # Annotation modality — task description, sourced from LeRobot's task index.
        modality["annotation"] = {
            "human.task_description": {"original_key": "task_index"}
        }

        logger.info(
            "GR00T modality: state={}, action={}, video={}",
            list(modality.get("state", {}).keys()) or "(none)",
            list(modality["action"].keys()),
            list(video.keys()) or "(none)",
        )

        return modality
