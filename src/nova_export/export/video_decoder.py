"""Simple sequential H.264 video decoder.

Decodes all video packets in order (no keyframe hunting), builds a frame cache
with timestamp-based indexing for efficient sampling at arbitrary timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import av
import av.logging
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from loguru import logger

if TYPE_CHECKING:
    pass


# Suppress swscaler "No accelerated colorspace conversion" warnings
av.logging.set_level(av.logging.ERROR)


@dataclass
class FrameCache:
    """In-memory cache of decoded video frames with timestamp indexing.

    Provides O(1) lookup of the nearest frame to any target timestamp.
    """

    frames: list[npt.NDArray[np.uint8]] = field(default_factory=list)
    timestamps_ns: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def start_ns(self) -> int:
        """Timestamp of the first frame in nanoseconds."""
        return int(self.timestamps_ns[0]) if len(self.timestamps_ns) > 0 else 0

    @property
    def end_ns(self) -> int:
        """Timestamp of the last frame in nanoseconds."""
        return int(self.timestamps_ns[-1]) if len(self.timestamps_ns) > 0 else 0

    @property
    def duration_s(self) -> float:
        """Duration of the video in seconds."""
        return (self.end_ns - self.start_ns) / 1e9

    def get_frame_at(self, target_ns: int) -> npt.NDArray[np.uint8] | None:
        """Get the frame nearest to the target timestamp.

        Args:
            target_ns: Target timestamp in nanoseconds.

        Returns:
            The nearest frame as HWC uint8 RGB array, or None if cache is empty.
        """
        if len(self.timestamps_ns) == 0:
            return None

        # Binary search for nearest timestamp
        idx = np.searchsorted(self.timestamps_ns, target_ns)

        # Choose closest of the two candidates
        if idx == 0:
            return self.frames[0]
        if idx >= len(self.timestamps_ns):
            return self.frames[-1]

        # Compare distances
        if (target_ns - self.timestamps_ns[idx - 1]) <= (
            self.timestamps_ns[idx] - target_ns
        ):
            return self.frames[idx - 1]
        return self.frames[idx]

    def get_frame_index(self, target_ns: int) -> int:
        """Get the index of the frame nearest to the target timestamp."""
        if len(self.timestamps_ns) == 0:
            return -1

        idx = np.searchsorted(self.timestamps_ns, target_ns)

        if idx == 0:
            return 0
        if idx >= len(self.timestamps_ns):
            return len(self.timestamps_ns) - 1

        if (target_ns - self.timestamps_ns[idx - 1]) <= (
            self.timestamps_ns[idx] - target_ns
        ):
            return idx - 1
        return idx


def _is_annex_b(data: bytes) -> bool:
    """Check if data starts with Annex B start code."""
    return data[:3] == b"\x00\x00\x01" or data[:4] == b"\x00\x00\x00\x01"


def _avcc_to_annex_b(data: bytes) -> bytes:
    """Convert AVCC format (length-prefixed NALUs) to Annex B (start code prefixed)."""
    result = bytearray()
    pos = 0
    while pos < len(data) - 4:
        # Read 4-byte length prefix
        nalu_len = int.from_bytes(data[pos : pos + 4], "big")
        pos += 4
        if pos + nalu_len > len(data):
            break
        # Add start code + NALU
        result.extend(b"\x00\x00\x00\x01")
        result.extend(data[pos : pos + nalu_len])
        pos += nalu_len
    return bytes(result)


def _flatten_blob(combined_array: Any, row_idx: int) -> bytes:
    """Extract bytes from a nested PyArrow blob structure.

    Rerun stores video samples as list<list<uint8>>, so we need to
    flatten to get the raw bytes for a given row.
    """
    row = combined_array[row_idx]
    # Handle nested list structure: list<list<uint8>>
    if hasattr(row, "values") and hasattr(row.values, "to_pylist"):
        nested = row.values.to_pylist()
        if nested and isinstance(nested[0], list):
            flat = []
            for chunk in nested:
                flat.extend(chunk)
            return bytes(flat)
        return bytes(nested)
    elif hasattr(row, "as_py"):
        py_val = row.as_py()
        if isinstance(py_val, list):
            if py_val and isinstance(py_val[0], list):
                flat = []
                for chunk in py_val:
                    flat.extend(chunk)
                return bytes(flat)
            return bytes(py_val)
        return bytes(py_val) if py_val else b""
    return bytes(row) if row else b""


class VideoDecoder:
    """Sequential H.264 video decoder.

    Decodes all packets from a Rerun VideoStream in order, building a frame cache
    with timestamp indexing. Much simpler than GOP-aware random-access decoding.

    Usage:
        decoder = VideoDecoder()
        cache = decoder.decode_segment(dataset, segment_id, camera_entity)
        frame = cache.get_frame_at(target_timestamp_ns)
    """

    def __init__(self, codec: str = "h264"):
        """Initialize the decoder.

        Args:
            codec: Video codec name (currently only 'h264' supported).
        """
        self.codec = codec

    def decode_segment(
        self,
        dataset: Any,
        segment_id: str,
        video_entity: str,
        index_column: str = "canonical_time",
    ) -> FrameCache:
        """Decode all video frames from a segment into a cache.

        Args:
            dataset: Rerun catalog dataset.
            segment_id: Segment ID to decode.
            video_entity: Entity path for the video stream (e.g., "/wrist").
            index_column: Rerun timeline column for timestamps.

        Returns:
            FrameCache with all decoded frames and their timestamps.
        """
        video_column = f"{video_entity}:VideoStream:sample"

        # Query all video packets in timestamp order
        view = dataset.filter_segments(segment_id)
        reader = view.reader(index=index_column)

        # Select timestamp and video sample columns
        table: pa.Table = reader.select(index_column, video_column).to_arrow_table()

        if table.num_rows == 0:
            logger.warning(
                "No video data found for {} in segment {}", video_entity, segment_id[:8]
            )
            return FrameCache()

        # Extract timestamps
        timestamps = self._extract_timestamps_ns(table[index_column])

        # Extract and decode packets sequentially
        video_col = table[video_column].combine_chunks()

        frames: list[npt.NDArray[np.uint8]] = []
        frame_timestamps: list[int] = []

        ctx = av.CodecContext.create(self.codec, "r")

        for i in range(table.num_rows):
            packet_bytes = _flatten_blob(video_col, i)
            if not packet_bytes:
                continue

            # Convert AVCC to Annex B if needed
            if not _is_annex_b(packet_bytes):
                packet_bytes = _avcc_to_annex_b(packet_bytes)

            try:
                for frame in ctx.decode(av.Packet(packet_bytes)):
                    rgb_frame = frame.to_ndarray(format="rgb24")
                    frames.append(rgb_frame)
                    frame_timestamps.append(timestamps[i])
            except av.error.InvalidDataError as e:
                logger.debug("Decode error at packet {}: {}", i, e)
                continue
            except Exception as e:
                logger.warning("Unexpected decode error at packet {}: {}", i, e)
                continue

        # Flush decoder
        try:
            for frame in ctx.decode(None):
                rgb_frame = frame.to_ndarray(format="rgb24")
                frames.append(rgb_frame)
                # Use last known timestamp for flushed frames
                if frame_timestamps:
                    frame_timestamps.append(frame_timestamps[-1])
        except Exception:
            pass

        logger.info(
            "Decoded {} frames from {} packets for {} in segment {}",
            len(frames),
            table.num_rows,
            video_entity,
            segment_id[:8],
        )

        return FrameCache(
            frames=frames,
            timestamps_ns=np.array(frame_timestamps, dtype=np.int64),
        )

    def _extract_timestamps_ns(
        self, ts_column: pa.ChunkedArray
    ) -> npt.NDArray[np.int64]:
        """Extract timestamps from PyArrow column as nanosecond integers."""
        timestamps = []
        for chunk in ts_column.chunks:
            for val in chunk:
                ts = val.as_py()
                if isinstance(ts, (int, float)):
                    timestamps.append(int(ts))
                elif hasattr(ts, "value"):
                    # numpy.datetime64
                    timestamps.append(int(ts.value))
                else:
                    timestamps.append(int(ts))
        return np.array(timestamps, dtype=np.int64)
