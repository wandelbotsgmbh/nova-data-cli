"""Export heads — format-specific dataset writers."""

from nova_export.export.heads.base import ExportHead


def __getattr__(name: str):
    if name == "LeRobotHead":
        from nova_export.export.heads.lerobot import LeRobotHead

        return LeRobotHead
    if name == "GrootHead":
        from nova_export.export.heads.groot import GrootHead

        return GrootHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ExportHead", "GrootHead", "LeRobotHead"]
