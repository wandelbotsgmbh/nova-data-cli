"""Per-episode camera/state timestamp alignment.

Lag varies per episode (see docs/investigations/camera-joint-sync-lag.md), so
a single global constant is wrong for many episodes. Instead: find "first
real motion" on the state side (same signal `signal_change` trimming uses)
and on REFERENCE_CAMERA, use the gap as this episode's shift, apply to all
camera streams. One delta for all cameras is valid because canonical_time is
already a shared, anchor-corrected timeline — the stream-start-time
differences between cameras are already normalized out of it.

FRAME_DIFF_THRESHOLD/DEBOUNCE_FRAMES were derived from 5 episodes (raw
datasets 5-9): idle mean-abs-pixel-diff ~0.03-0.05 with noise spikes to ~0.6;
3 consecutive frames above 0.05 cleanly separates real onset from noise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nova_export.export.video_decoder import FrameCache, frame_diffs

# Was cam_side: too weak a signal (often <5x idle floor), gave 600ms+ outliers
# on dataset 10. cam_wrist rejected: idle noise (0.06-0.12) already exceeds
# FRAME_DIFF_THRESHOLD, so its detector mostly fires on noise. cam_top: low
# idle floor + strong signal (25-100x) — median 91ms vs cam_side's 228ms, no
# outliers, across 5 dataset-10 episodes.
REFERENCE_CAMERA = "cam_top"

# Mean-abs-pixel-diff (0-255 scale) a frame pair must exceed, for
# DEBOUNCE_FRAMES consecutive pairs, to count as "camera shows motion".
# Empirically derived — see module docstring.
FRAME_DIFF_THRESHOLD = 0.05
DEBOUNCE_FRAMES = 3

# Search window after the state-side onset. Episodes have a ~2.5-3s idle
# lookback before real motion (EPISODE_LOOKBACK_S); 1.5s past state onset is
# generous for the lag itself without reaching into later motion phases
# (approach/grasp/place), which must not be mistaken for the initial onset.
SEARCH_WINDOW_NS = 1_500_000_000

# Sanity bounds on the measured delta: camera lags, never leads (0), and the
# investigation's two measurement rounds landed at ~630-860ms and ~150-350ms
# respectively — 900ms covers both with margin.
MIN_DELTA_NS = 0
MAX_DELTA_NS = 900_000_000


def find_camera_onset(
    mid_ts: npt.NDArray[np.int64],
    diffs: npt.NDArray[np.float64],
    search_start_ns: int,
    search_end_ns: int,
    threshold: float = FRAME_DIFF_THRESHOLD,
    debounce_frames: int = DEBOUNCE_FRAMES,
) -> int | None:
    """First timestamp where `diffs` exceeds `threshold` for `debounce_frames`
    consecutive samples, searching only within [search_start_ns, search_end_ns]
    (lag is one-directional — camera never leads — so we never search before
    the state-side onset).

    Returns the timestamp of the first frame of that qualifying run, or None
    if no run of that length is found in the window.
    """
    in_window = np.where((mid_ts >= search_start_ns) & (mid_ts <= search_end_ns))[0]
    run = 0
    for i in in_window:
        if diffs[i] > threshold:
            run += 1
            if run >= debounce_frames:
                return int(mid_ts[i - debounce_frames + 1])
        else:
            run = 0
    return None


def measure_episode_delta_ns(
    cache: FrameCache,
    state_onset_ns: int,
    *,
    threshold: float = FRAME_DIFF_THRESHOLD,
    debounce_frames: int = DEBOUNCE_FRAMES,
    search_window_ns: int = SEARCH_WINDOW_NS,
    min_delta_ns: int = MIN_DELTA_NS,
    max_delta_ns: int = MAX_DELTA_NS,
) -> tuple[int | None, str | None]:
    """Measure one episode's camera-lag delta from a decoded reference-camera
    cache and the raw state-side onset timestamp.

    Returns (delta_ns, error): exactly one is None. `error` is a human
    readable reason to skip the episode entirely — either no onset was found
    in the search window, or the implied delta is outside the plausible
    range (camera leading, or an implausibly large lag). Callers must not
    export the episode unshifted in either case.
    """
    mid_ts, diffs = frame_diffs(cache)
    if len(diffs) == 0:
        return None, "reference camera has <2 decoded frames"

    onset_ns = find_camera_onset(
        mid_ts,
        diffs,
        state_onset_ns,
        state_onset_ns + search_window_ns,
        threshold=threshold,
        debounce_frames=debounce_frames,
    )
    if onset_ns is None:
        return None, (
            f"no camera-motion onset found on {REFERENCE_CAMERA!r} within "
            f"{search_window_ns / 1e6:.0f}ms of state onset"
        )

    delta_ns = onset_ns - state_onset_ns
    if not (min_delta_ns <= delta_ns <= max_delta_ns):
        return None, (
            f"measured delta {delta_ns / 1e6:.1f}ms outside plausible range "
            f"[{min_delta_ns / 1e6:.0f}, {max_delta_ns / 1e6:.0f}]ms"
        )

    return delta_ns, None
