"""Testcases for online bounce-near-boundary tracking (IMPLEMENTATION.md §5.2).

The tracker turns independent per-frame ball boxes into a trajectory, fires
exactly once per physical bounce (vertical velocity flip at a y-maximum in
image coords), and the boundary gate keeps only bounces near the court edge.
"""

import sys
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

from ball_tracker import BallTracker, bounce_near_boundary

# Court numbering: 1=TR, 2=TL, 3=BR, 4=BL.
CORNERS = [(100, 10), (10, 10), (100, 100), (10, 100)]


def feed(tracker, ys, x=50, start=0, conf=0.9, corners=None):
    """Push a synthetic vertical trajectory; return all emitted bounces."""
    out = []
    for i, y in enumerate(ys):
        bounce, _ = tracker.update([(x, y, conf)], start + i, corners=corners)
        if bounce is not None:
            out.append(bounce)
    return out


def feed_xy(tracker, xys, start=0, conf=0.9, corners=None):
    """Push (x, y) samples; return all emitted exits."""
    out = []
    for i, (x, y) in enumerate(xys):
        _, exit_event = tracker.update([(x, y, conf)], start + i, corners=corners)
        if exit_event is not None:
            out.append(exit_event)
    return out


def test_fall_then_rise_fires_once_at_peak():
    tracker = BallTracker()
    ys = [10, 20, 35, 55, 80, 110, 145, 150, 140, 120, 95, 65, 40, 20]
    bounces = feed(tracker, ys)
    assert len(bounces) == 1
    # Smoothing lags the peak by ~1 frame; raw sample at the smoothed peak.
    assert bounces[0]["position"][1] == pytest.approx(140, abs=3)
    assert bounces[0]["frame_index"] == 8


def test_flat_rolling_ball_stays_silent():
    tracker = BallTracker()
    bounces = feed(tracker, [100] * 20)
    assert bounces == []


def test_rising_without_prior_fall_stays_silent():
    tracker = BallTracker()
    bounces = feed(tracker, [150, 140, 120, 95, 65, 40, 20, 10])
    assert bounces == []


def test_jitter_below_prominence_stays_silent():
    tracker = BallTracker()
    ys = [100, 102, 99, 103, 100, 98, 101, 99, 100, 102, 100, 99] * 2
    assert feed(tracker, ys) == []


def test_short_occlusion_gap_is_bridged():
    tracker = BallTracker()
    bounces = []
    ys = [10, 25, 45, 70, 100, 135, 150, 140, 115, 85, 55, 30]
    for i, y in enumerate(ys):
        if i in (5, 6):  # two missed frames around the peak
            bounce, _ = tracker.update([], i)
        else:
            bounce, _ = tracker.update([(50, y, 0.9)], i)
        if bounce is not None:
            bounces.append(bounce)
    assert len(bounces) == 1


def test_cooldown_suppresses_double_fire():
    tracker = BallTracker(cooldown_frames=30)
    ys = [10, 30, 60, 90, 120, 150, 140, 110, 80, 100, 130, 150, 130, 100, 70, 40]
    bounces = feed(tracker, ys)
    assert len(bounces) == 1


def test_gate_keeps_near_boundary_bounce():
    inside, dist_in = bounce_near_boundary((12, 50), CORNERS)
    assert inside is True
    assert dist_in > 0  # positive = inside polygon
    outside, dist_out = bounce_near_boundary((104, 50), CORNERS)
    assert outside is True
    assert dist_out < 0


def test_gate_drops_mid_court_bounce():
    inside, _ = bounce_near_boundary((55, 55), CORNERS)
    assert inside is False


def test_exit_fires_once_on_inside_to_outside():
    tracker = BallTracker()
    xys = [(50, 50)] * 5 + [(80, 50), (110, 50), (140, 50)]
    exits = feed_xy(tracker, xys, corners=CORNERS)
    assert len(exits) == 1
    assert exits[0]["position"] == [110, 50]
    assert exits[0]["frame_index"] == 6


def test_exit_needs_prior_inside_run():
    tracker = BallTracker()
    # Track starts outside and stays there — never established inside.
    exits = feed_xy(tracker, [(150, 50)] * 8, corners=CORNERS)
    assert exits == []


def test_exit_cooldown_suppresses_line_flicker():
    tracker = BallTracker(exit_cooldown_frames=30)
    xys = [(50, 50)] * 4 + [(110, 50)] + [(50, 50)] * 2 + [(110, 50)] * 3
    exits = feed_xy(tracker, xys, corners=CORNERS)
    assert len(exits) == 1


def test_exit_skipped_without_corners():
    tracker = BallTracker()
    xys = [(50, 50)] * 5 + [(140, 50)] * 3
    exits = feed_xy(tracker, xys)
    assert exits == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
