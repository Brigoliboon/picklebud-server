"""Online ball trajectory tracking + bounce detection for the sidecar.

Turns independent per-frame ball boxes into one continuous trajectory and
fires exactly once per physical bounce: a vertical-velocity flip (+ to - in
image coords, where y grows downward) at a local y-maximum, confirmed over
several samples and guarded by prominence + cooldown so detector jitter and
smoothing ripples stay silent. Short occlusions are coasted through.

The boundary gate (`bounce_near_boundary`) keeps only bounces near the court
edge, using the calibrated court polygon — a bounce far from any line is not
dispute-relevant.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


def bounce_near_boundary(
    point: tuple[float, float],
    corners: list[tuple[int, int]],
    margin_ratio: float = 0.06,
) -> tuple[bool, float]:
    """Check a bounce point against the court polygon.

    Returns (near, signed_distance_px): positive distance = inside the court.
    Near covers both sides of the line — a bounce just in and just out are
    equally dispute-worthy. Margin scales with court size on screen.
    """
    # Corners arrive numbered 1=TR, 2=TL, 3=BR, 4=BL (row-major, not a
    # perimeter loop) — re-walk as TR → TL → BL → BR for a valid polygon.
    poly = np.array([corners[0], corners[1], corners[3], corners[2]], dtype=np.float32)
    dist = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), True)
    top = math.dist(corners[0], corners[1])
    bottom = math.dist(corners[2], corners[3])
    margin = margin_ratio * max(top, bottom, 1.0)
    return abs(dist) <= margin, dist


class BallTracker:
    """Associates ball detections frame-to-frame and detects bounces."""

    def __init__(
        self,
        max_jump_px: float = 200.0,
        max_missed: int = 5,
        smooth_window: int = 3,
        confirm_up: int = 2,
        min_down: int = 2,
        min_prominence_px: float = 6.0,
        cooldown_frames: int = 30,
        min_inside_frames: int = 3,
        exit_cooldown_frames: int = 30,
        eps_px: float = 1.0,
        history: int = 120,
    ) -> None:
        self.max_jump_px = max_jump_px
        self.max_missed = max_missed
        self.smooth_window = smooth_window
        self.confirm_up = confirm_up
        self.min_down = min_down
        self.min_prominence_px = min_prominence_px
        self.cooldown_frames = cooldown_frames
        self.min_inside_frames = min_inside_frames
        self.exit_cooldown_frames = exit_cooldown_frames
        self.eps_px = eps_px
        self._samples: deque[tuple[int, int, int, float]] = deque(maxlen=history)
        self._active = False
        self._missed = 0
        self._peak_y = 0.0
        self._peak_idx = 0
        self._downs = 0
        self._ups = 0
        self._last_bounce_idx = -10**9
        self._inside_run = 0
        self._last_exit_idx = -10**9

    def update(
        self,
        centers: list[tuple[int, int, float]],
        frame_index: int,
        timestamp: float = 0.0,
        corners: list[tuple[int, int]] | None = None,
    ) -> tuple[dict | None, dict | None]:
        """Feed one frame's ball centers [(x, y, conf)].

        Returns (bounce, exit): bounce fires on a confirmed flip near the
        boundary (caller gates by polygon); exit fires on an inside→outside
        court crossing when `corners` is given. Either may be None.
        """
        bounce: dict | None = None
        accepted: tuple[int, int] | None = None
        if centers:
            best = max(centers, key=lambda c: c[2])
            if self._active:
                gated = [
                    c
                    for c in centers
                    if math.dist((c[0], c[1]), (self._last_x, self._last_y))
                    <= self.max_jump_px
                ]
                if gated:
                    best = max(gated, key=lambda c: c[2])
                    self._missed = 0
                    bounce = self._add_sample(best[0], best[1], frame_index, timestamp)
                    accepted = (best[0], best[1])
                else:
                    self._missed += 1
                    if self._missed > self.max_missed:
                        self._reset_track()
                        self._start(best[0], best[1], frame_index, timestamp)
                        accepted = (best[0], best[1])
            else:
                self._start(best[0], best[1], frame_index, timestamp)
                accepted = (best[0], best[1])
        elif self._active:
            self._missed += 1
            if self._missed > self.max_missed:
                self._reset_track()
        exit_event = self._check_exit(accepted, frame_index, timestamp, corners)
        return bounce, exit_event

    def _check_exit(
        self,
        accepted: tuple[int, int] | None,
        frame_index: int,
        timestamp: float,
        corners: list[tuple[int, int]] | None,
    ) -> dict | None:
        """Fire once when an established inside-track crosses outside."""
        if accepted is None or corners is None or len(corners) != 4:
            return None
        try:
            poly = np.array(
                [corners[0], corners[1], corners[3], corners[2]], dtype=np.float32
            )
            inside = cv2.pointPolygonTest(poly, (float(accepted[0]), float(accepted[1])), True) > 0
        except Exception:
            return None
        if inside:
            self._inside_run += 1
            return None
        if self._inside_run >= self.min_inside_frames and (
            frame_index - self._last_exit_idx
        ) >= self.exit_cooldown_frames:
            self._last_exit_idx = frame_index
            self._inside_run = 0
            return {
                "position": [accepted[0], accepted[1]],
                "frame_index": frame_index,
                "timestamp": round(timestamp, 3),
            }
        return None

    def _start(self, x: int, y: int, frame_index: int, timestamp: float) -> None:
        self._active = True
        self._missed = 0
        self._samples.append((x, y, frame_index, timestamp))
        self._last_x, self._last_y = x, y
        self._peak_y = float(y)
        self._peak_idx = frame_index
        self._downs = 0
        self._ups = 0

    def _reset_track(self) -> None:
        self._active = False
        self._missed = 0
        self._samples.clear()
        self._downs = 0
        self._ups = 0
        self._inside_run = 0

    def _add_sample(self, x: int, y: int, frame_index: int, timestamp: float) -> dict | None:
        self._samples.append((x, y, frame_index, timestamp))
        self._last_x, self._last_y = x, y
        window = list(self._samples)[-self.smooth_window :]
        s = sum(p[1] for p in window) / len(window)
        if s > self._peak_y + self.eps_px:
            self._peak_y = s
            self._peak_idx = frame_index
            self._downs += 1
            self._ups = 0
            return None
        if s < self._peak_y - self.eps_px:
            self._ups += 1
        if (
            self._ups >= self.confirm_up
            and self._downs >= self.min_down
            and (self._peak_y - s) >= self.min_prominence_px
            and (frame_index - self._last_bounce_idx) >= self.cooldown_frames
        ):
            px, py, pidx, pt = self._peak_sample()
            self._last_bounce_idx = frame_index
            self._peak_y = s
            self._peak_idx = frame_index
            self._downs = 0
            self._ups = 0
            return {"position": [px, py], "frame_index": pidx, "timestamp": round(pt, 3)}
        return None

    def _peak_sample(self) -> tuple[int, int, int, float]:
        for x, y, idx, t in reversed(self._samples):
            if idx == self._peak_idx:
                return x, y, idx, t
        x, y, idx, t = self._samples[-1]
        return x, y, idx, t
