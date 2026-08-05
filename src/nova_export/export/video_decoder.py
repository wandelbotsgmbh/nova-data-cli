"""Simple sequential H.264 video decoder.

Decodes video packets in order (no keyframe hunting). Two modes:

- `load_packets()` + `decode_at()`: the memory-efficient path used by the
  export pipeline. Packets (compressed) are loaded first so time bounds are
  known before decoding, then the stream is decoded once and only the frames
  nearest to the requested sample timestamps are kept (already resized).
- `decode_segment()`: decodes *all* frames into a FrameCache. Simple, but holds
  every decoded frame in memory — only suitable for short segments/tests.
"""

from __future__ import annotations

from collections.abc import Iterator
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


def resize_rgb(
    frame: npt.NDArray[np.uint8], width: int, height: int
) -> npt.NDArray[np.uint8]:
    """Resize an HWC RGB frame with PyAV (libswscale).

    Used instead of OpenCV so we don't load a second ffmpeg/libavdevice and
    clash with av's (macOS objc warning).
    """
    vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
    vf = vf.reformat(width=width, height=height)
    return vf.to_ndarray(format="rgb24")


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
        idx = self.get_frame_index(target_ns)
        return self.frames[idx] if idx >= 0 else None

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


@dataclass
class PacketSeries:
    """Compressed video packets of one stream in one segment, pre-decode.

    Holds the raw (still encoded) packet table, so it is cheap to keep around:
    it lets the caller know the stream's time bounds and packet count before
    paying for any decoding.
    """

    table: pa.Table
    timestamps_ns: npt.NDArray[np.int64]
    video_column: str
    entity: str

    @property
    def num_packets(self) -> int:
        return len(self.timestamps_ns)

    @property
    def start_ns(self) -> int:
        """Timestamp of the first packet in nanoseconds."""
        return int(self.timestamps_ns[0]) if len(self.timestamps_ns) > 0 else 0

    @property
    def end_ns(self) -> int:
        """Timestamp of the last packet in nanoseconds."""
        return int(self.timestamps_ns[-1]) if len(self.timestamps_ns) > 0 else 0


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


def _flatten_blob_slow(combined_array: Any, row_idx: int) -> bytes:
    """Per-value fallback for `_flatten_blob` (handles exotic layouts)."""
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


def _flatten_blob(combined_array: Any, row_idx: int) -> bytes:
    """Extract bytes from a nested PyArrow blob structure.

    Rerun stores video samples as list<list<uint8>>, so we need to flatten to
    get the raw bytes for a given row. The fast path stays in Arrow/NumPy;
    building per-byte Python ints (the fallback) is orders of magnitude slower.
    """
    try:
        row = combined_array[row_idx]
        if hasattr(row, "is_valid") and not row.is_valid:
            return b""
        values = row.values  # pa.Array for list-typed scalars
        while pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            values = values.flatten()
        return np.asarray(values, dtype=np.uint8).tobytes()
    except (AttributeError, TypeError, pa.ArrowInvalid):
        return _flatten_blob_slow(combined_array, row_idx)


def extract_timestamps_ns(ts_column: Any) -> npt.NDArray[np.int64]:
    """Extract timestamps from a PyArrow column as nanosecond integers."""
    try:
        arr = ts_column.to_numpy()
        if arr.dtype.kind == "M":  # datetime64
            return arr.astype("datetime64[ns]").astype(np.int64)
        if arr.dtype.kind in "iuf":
            return arr.astype(np.int64)
    except (pa.ArrowInvalid, TypeError, ValueError):
        pass

    # Fallback: per-value conversion for exotic value types.
    timestamps = []
    chunks = ts_column.chunks if hasattr(ts_column, "chunks") else [ts_column]
    for chunk in chunks:
        for val in chunk:
            ts = val.as_py()
            if isinstance(ts, (int, float)):
                timestamps.append(int(ts))
            elif hasattr(ts, "value"):
                # numpy.datetime64 / pandas Timestamp
                timestamps.append(int(ts.value))
            else:
                timestamps.append(int(ts))
    return np.array(timestamps, dtype=np.int64)


class VideoDecoder:
    """Sequential H.264 video decoder.

    Decodes packets from a Rerun VideoStream in order. Much simpler than
    GOP-aware random-access decoding.

    Usage (memory-efficient export path):
        decoder = VideoDecoder()
        packets = decoder.load_packets(dataset, segment_id, camera_entity)
        # ... derive sample timestamps from packets.start_ns / end_ns ...
        cache = decoder.decode_at(packets, sample_timestamps_ns)

    Usage (decode everything):
        cache = decoder.decode_segment(dataset, segment_id, camera_entity)
        frame = cache.get_frame_at(target_timestamp_ns)
    """

    def __init__(self, codec: str = "h264"):
        """Initialize the decoder.

        Args:
            codec: Video codec name (currently only 'h264' supported).
        """
        self.codec = codec

    def load_packets(
        self,
        dataset: Any,
        segment_id: str,
        video_entity: str,
        index_column: str = "canonical_time",
    ) -> PacketSeries:
        """Load a segment's compressed video packets without decoding.

        Args:
            dataset: Rerun catalog dataset.
            segment_id: Segment ID to load.
            video_entity: Entity path for the video stream (e.g., "/wrist").
            index_column: Rerun timeline column for timestamps.

        Returns:
            PacketSeries with the packet table and per-packet timestamps.
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
            return PacketSeries(
                table=table,
                timestamps_ns=np.array([], dtype=np.int64),
                video_column=video_column,
                entity=video_entity,
            )

        timestamps = extract_timestamps_ns(table[index_column])
        return PacketSeries(
            table=table,
            timestamps_ns=timestamps,
            video_column=video_column,
            entity=video_entity,
        )

    def _iter_frames(
        self, packets: PacketSeries
    ) -> Iterator[tuple[npt.NDArray[np.uint8], int]]:
        """Decode a packet series sequentially, yielding (RGB frame, timestamp_ns).

        Frames decoded from packet i carry that packet's timestamp; frames
        emitted by the final decoder flush carry the last seen timestamp.
        """
        video_col = packets.table[packets.video_column].combine_chunks()
        timestamps = packets.timestamps_ns

        ctx = av.CodecContext.create(self.codec, "r")
        last_ts: int | None = None

        for i in range(packets.num_packets):
            packet_bytes = _flatten_blob(video_col, i)
            if not packet_bytes:
                continue

            # Convert AVCC to Annex B if needed
            if not _is_annex_b(packet_bytes):
                packet_bytes = _avcc_to_annex_b(packet_bytes)

            try:
                for frame in ctx.decode(av.Packet(packet_bytes)):
                    last_ts = int(timestamps[i])
                    yield frame.to_ndarray(format="rgb24"), last_ts
            except av.error.InvalidDataError as e:
                logger.debug("Decode error at packet {}: {}", i, e)
                continue
            except Exception as e:
                logger.warning("Unexpected decode error at packet {}: {}", i, e)
                continue

        # Flush decoder
        try:
            for frame in ctx.decode(None):
                if last_ts is not None:
                    yield frame.to_ndarray(format="rgb24"), last_ts
        except Exception:
            pass

    def decode_at(
        self,
        packets: PacketSeries,
        target_timestamps_ns: npt.NDArray[np.int64],
        target_size: tuple[int, int] | None = None,
    ) -> FrameCache:
        """Decode a segment, keeping only the frames nearest each target timestamp.

        The stream is decoded sequentially exactly once, but instead of caching
        every decoded frame, each target timestamp is resolved to its nearest
        frame on the fly (ties to the earlier frame, clamped at both ends —
        the same selection rule as ``FrameCache.get_frame_at``). Only selected
        frames are retained, already resized to ``target_size``, so peak memory
        is one raw decoded frame plus the selected output frames. Decoding
        stops early once every target is resolved.

        Args:
            packets: Packet series from :meth:`load_packets`.
            target_timestamps_ns: Sorted sample timestamps (e.g. the FPS grid).
            target_size: Optional (width, height) to resize selected frames to.

        Returns:
            FrameCache aligned to the targets: ``frames[i]`` is the frame for
            ``target_timestamps_ns[i]`` (repeated targets share one array).
            Empty cache if no frame could be decoded.
        """
        targets = np.asarray(target_timestamps_ns, dtype=np.int64)
        num_targets = len(targets)
        frames_out: list[npt.NDArray[np.uint8]] = []

        prev_frame: npt.NDArray[np.uint8] | None = None
        prev_ts = 0
        prev_out: npt.NDArray[np.uint8] | None = None  # processed prev_frame

        def emit_prev() -> npt.NDArray[np.uint8]:
            # Resize lazily and once per selected source frame.
            nonlocal prev_out
            if prev_out is None:
                assert prev_frame is not None
                if target_size is not None:
                    prev_out = resize_rgb(prev_frame, *target_size)
                else:
                    prev_out = prev_frame
            return prev_out

        decoded_frames = 0
        for frame, ts in self._iter_frames(packets):
            decoded_frames += 1
            if prev_frame is None:
                prev_frame, prev_ts = frame, ts
                continue

            # All targets at or before the midpoint of (prev, current) are
            # nearest to prev (ties go to the earlier frame).
            while len(frames_out) < num_targets and (
                targets[len(frames_out)] - prev_ts
            ) <= (ts - targets[len(frames_out)]):
                frames_out.append(emit_prev())

            if len(frames_out) >= num_targets:
                # Every target resolved — skip decoding the rest of the stream.
                break

            prev_frame, prev_ts, prev_out = frame, ts, None

        if prev_frame is None:
            logger.warning(
                "No frames decoded from {} packets for {}",
                packets.num_packets,
                packets.entity,
            )
            return FrameCache()

        # Remaining targets are at/after the last frame: clamp to it.
        while len(frames_out) < num_targets:
            frames_out.append(emit_prev())

        logger.info(
            "Decoded {} frames from {} packets for {}, kept {} grid frames",
            decoded_frames,
            packets.num_packets,
            packets.entity,
            num_targets,
        )

        return FrameCache(frames=frames_out, timestamps_ns=targets.copy())

    def decode_segment(
        self,
        dataset: Any,
        segment_id: str,
        video_entity: str,
        index_column: str = "canonical_time",
    ) -> FrameCache:
        """Decode *all* video frames from a segment into a cache.

        Note: holds every decoded frame in memory at full resolution — use
        :meth:`load_packets` + :meth:`decode_at` for the export pipeline.

        Args:
            dataset: Rerun catalog dataset.
            segment_id: Segment ID to decode.
            video_entity: Entity path for the video stream (e.g., "/wrist").
            index_column: Rerun timeline column for timestamps.

        Returns:
            FrameCache with all decoded frames and their timestamps.
        """
        packets = self.load_packets(dataset, segment_id, video_entity, index_column)
        if packets.num_packets == 0:
            return FrameCache()

        frames: list[npt.NDArray[np.uint8]] = []
        frame_timestamps: list[int] = []
        for frame, ts in self._iter_frames(packets):
            frames.append(frame)
            frame_timestamps.append(ts)

        logger.info(
            "Decoded {} frames from {} packets for {} in segment {}",
            len(frames),
            packets.num_packets,
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
        return extract_timestamps_ns(ts_column)
