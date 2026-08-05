"""Tests for per-episode camera/state onset alignment (camera_lag.py).

Synthetic self-check: fabricate a frame cache that is idle-noise before an
injected onset and clearly "moving" (large frame-to-frame diff) after it,
and assert the detector recovers the injected delay within tolerance.
"""

from __future__ import annotations

import numpy as np

from nova_export.export.camera_lag import measure_episode_delta_ns
from nova_export.export.video_decoder import FrameCache

_FPS = 30
_STEP_NS = int(1e9 / _FPS)
_FRAME_SHAPE = (4, 4, 3)


def _synthetic_cache(onset_ns: int, n_frames: int = 300) -> FrameCache:
    """Frames with tiny noise before `onset_ns`, large frame-to-frame jumps
    from `onset_ns` onward (mimics idle vs. real motion on cam_side)."""
    rng = np.random.default_rng(0)
    timestamps = np.arange(n_frames, dtype=np.int64) * _STEP_NS
    frames = []
    level = np.zeros(_FRAME_SHAPE, dtype=np.float32)
    for ts in timestamps:
        level = level + (
            rng.uniform(-0.08, 0.08, size=_FRAME_SHAPE)
            if ts < onset_ns
            else rng.uniform(40, 60, size=_FRAME_SHAPE)
        )
        frames.append(np.clip(level, 0, 255).astype(np.uint8))
    return FrameCache(frames=frames, timestamps_ns=timestamps)


def test_recovers_injected_delay():
    injected_delay_ns = 300_000_000  # 300ms
    cache = _synthetic_cache(onset_ns=injected_delay_ns)

    delta_ns, error = measure_episode_delta_ns(cache, state_onset_ns=0)

    assert error is None
    assert delta_ns is not None
    # Tolerance: a couple frame periods at 30fps (~33ms/frame).
    assert abs(delta_ns - injected_delay_ns) <= 3 * _STEP_NS


def test_no_onset_in_window_is_skipped():
    # Onset well beyond the (default 1.5s) search window -> not found.
    cache = _synthetic_cache(onset_ns=5_000_000_000, n_frames=600)
    delta_ns, error = measure_episode_delta_ns(cache, state_onset_ns=0)
    assert delta_ns is None
    assert error is not None


def test_delta_outside_plausible_range_is_rejected():
    # A detectable onset, but its delta (950ms) is outside the 0-900ms bound.
    cache = _synthetic_cache(onset_ns=950_000_000, n_frames=400)
    delta_ns, error = measure_episode_delta_ns(cache, state_onset_ns=0)
    assert delta_ns is None
    assert error is not None


def test_near_zero_delay_is_accepted():
    # Camera motion coincides almost exactly with state onset -- a valid
    # (if small) delta, not a rejection. Search never looks before state
    # onset, so this also covers the "camera can't appear to lead" case.
    cache = _synthetic_cache(onset_ns=0, n_frames=300)
    delta_ns, error = measure_episode_delta_ns(cache, state_onset_ns=0)
    assert error is None
    assert delta_ns is not None
    assert 0 <= delta_ns <= 3 * _STEP_NS


if __name__ == "__main__":
    test_recovers_injected_delay()
    test_no_onset_in_window_is_skipped()
    test_delta_outside_plausible_range_is_rejected()
    test_near_zero_delay_is_accepted()
    print("ok")
