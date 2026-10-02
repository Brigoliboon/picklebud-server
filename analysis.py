"""Frame analysis: court corners + ball tracking on a decoded frame.

Centralizes the ONNX inference stack for the sidecar. Given an OpenCV BGR
frame it runs the court-corner detector and the ball tracker, draws the
resulting boxes/labels onto the frame, and serializes the detections into the
JSON payload shape the frontend consumes (see IMPLEMENTATION.md §5.6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ball_tracker import BallTracker, bounce_near_boundary
from detector import Detection, YoloDetector

MODEL_DIR = Path(__file__).resolve().parent / "assets" / "model"

_COURT_COLOR = (0, 255, 0)      # BGR
_BALL_COLOR = (0, 0, 255)       # BGR
_BOUNCE_COLOR = (0, 255, 255)   # BGR yellow ring on a fresh bounce
_COURT_MARKER_RADIUS = 6
_BOUNCE_MARKER_TTL_FRAMES = 5


class AnalysisEngine:
    """Court detection runs once at setup and is cached; ball tracking runs
    on every frame afterward."""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        self.court = YoloDetector(
            model_dir / "court_detection-v1.onnx", conf_threshold, nms_threshold
        )
        self.ball = YoloDetector(
            model_dir / "ball_tracker-v2.onnx", conf_threshold, nms_threshold
        )
        self._court_detections: list[Detection] = []
        self.court_ready = False
        self._confirmed_corners: list[tuple[int, int]] | None = None
        self.bounce = BallTracker()
        self._last_bounce: dict | None = None

    @property
    def calibrated(self) -> bool:
        return self._confirmed_corners is not None

    # Court edges under the 1=TR, 2=TL, 3=BR, 4=BL numbering, as index
    # pairs into the corner list: top 1-2, bottom 3-4, right 1-3, left 2-4.
    COURT_EDGES = ((0, 1), (2, 3), (0, 2), (1, 3))

    @staticmethod
    def order_corners(points: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
        """Number 4 corners 1=TR, 2=TL, 3=BR, 4=BL (row-major).

        Class labels (C1..C4) carry no geometric order guarantee, and referees
        may drag handles anywhere — so normalize geometrically: the two
        highest points are the top pair (right→1, left→2), the two lowest the
        bottom pair (right→3, left→4). Edges are then top 1-2, bottom 3-4,
        right 1-3, left 2-4, which never forms an X.
        """
        pts = [(int(p[0]), int(p[1])) for p in points]
        if len(pts) != 4:
            raise ValueError("exactly 4 corner points required")
        by_y = sorted(pts, key=lambda p: (p[1], p[0]))
        top = sorted(by_y[:2], key=lambda p: -p[0])
        bottom = sorted(by_y[2:], key=lambda p: -p[0])
        return [top[0], top[1], bottom[0], bottom[1]]

    # Back-compat alias.
    order_perimeter = order_corners

    def ordered_corner_points(self) -> list[tuple[int, int]]:
        """Detected corners numbered 1=TL, 2=TR, 3=BL, 4=BR, as (x, y)."""
        pts = [(d.cx, d.cy) for d in self._court_detections]
        if len(pts) != 4:
            return pts
        try:
            return self.order_corners(pts)
        except ValueError:
            return pts

    def set_confirmed_corners(self, points: Sequence[Sequence[int]]) -> None:
        self._confirmed_corners = self.order_corners(points)
        self.court_ready = True

    def clear_confirmation(self) -> None:
        self._confirmed_corners = None

    def reset_ball_track(self) -> None:
        """Drop trajectory state — a new source means a new rally."""
        self.bounce = BallTracker()
        self._last_bounce = None

    def detect_court(self, frame_bgr: np.ndarray) -> bool:
        """Run court detection once; caches corners as soon as any are found.

        Proposes corners only — never auto-confirms. The referee must
        review/adjust the 4 points and POST /setup/confirm-court before the
        live (ball-tracking) stream is allowed to start.
        """
        self._court_detections = self.court.detect(frame_bgr)
        self.court_ready = len(self._court_detections) > 0 or self.calibrated
        return len(self._court_detections) > 0

    def analyze(
        self, frame_bgr: np.ndarray, timestamp: float = 0.0, frame_index: int = 0
    ) -> tuple[np.ndarray, list[dict], dict | None, dict | None]:
        """Return (annotated_frame, detections, bounce_info, exit_info).

        Ball tracking every frame; cached court corners drawn/returned too.
        bounce_info is set exactly once per physical bounce near the boundary;
        exit_info exactly once when the tracked ball crosses out of the court.
        """
        ball_dets = self.ball.detect(frame_bgr)
        centers = [(d.cx, d.cy, d.score) for d in ball_dets]
        bounce_candidate, exit_candidate = self.bounce.update(
            centers, frame_index, timestamp, self._confirmed_corners
        )

        bounce_info: dict | None = None
        if bounce_candidate is not None and self._confirmed_corners is not None:
            near, dist = bounce_near_boundary(
                bounce_candidate["position"], self._confirmed_corners
            )
            if near:
                bounce_info = {
                    "position": bounce_candidate["position"],
                    "frame_index": bounce_candidate["frame_index"],
                    "timestamp": bounce_candidate["timestamp"],
                    "boundary_distance_px": round(dist, 1),
                }
                self._last_bounce = bounce_info
        exit_info: dict | None = exit_candidate

        annotated = frame_bgr.copy()
        if self._confirmed_corners is not None:
            detections = [
                {
                    "label": f"C{i + 1}",
                    "class_id": i,
                    "confidence": 1.0,
                    "box": [x - 4, y - 4, x + 4, y + 4],
                    "center": [x, y],
                    "is_court": True,
                }
                for i, (x, y) in enumerate(self._confirmed_corners)
            ]
            corners = self._confirmed_corners
            for i, j in self.COURT_EDGES:
                cv2.line(annotated, corners[i], corners[j], _COURT_COLOR, 2)
            for x, y in corners:
                cv2.circle(annotated, (x, y), _COURT_MARKER_RADIUS, _COURT_COLOR, -1)
        else:
            detections = [self._to_dict(d, self.court.class_names) for d in self._court_detections]
            if self._court_detections:
                self._draw(annotated, self._court_detections, color=_COURT_COLOR,
                           labels=self.court.class_names)
        detections += [self._to_dict(d, self.ball.class_names) for d in ball_dets]
        self._draw(annotated, ball_dets, color=_BALL_COLOR,
                   labels=self.ball.class_names)
        if (
            self._last_bounce is not None
            and frame_index - self._last_bounce["frame_index"] <= _BOUNCE_MARKER_TTL_FRAMES
        ):
            x, y = self._last_bounce["position"]
            cv2.circle(annotated, (int(x), int(y)), 14, _BOUNCE_COLOR, 2)
        return annotated, detections, bounce_info, exit_info

    @staticmethod
    def _to_dict(d: Detection, class_names: Sequence[str]) -> dict:
        label = class_names[d.class_id] if d.class_id < len(class_names) else str(d.class_id)
        return {
            "label": label,
            "class_id": d.class_id,
            "confidence": round(d.score, 4),
            # Box in original-frame pixel coordinates.
            "box": [d.x1, d.y1, d.x2, d.y2],
            "center": [d.cx, d.cy],
            "is_court": label.startswith("C") and len(label) == 2,
        }

    @staticmethod
    def _draw(frame, detections, color, labels: Sequence[str], markers=True) -> None:
        for d in detections:
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
            if markers:
                cv2.circle(frame, (d.cx, d.cy), 4, color, -1)
            label = labels[d.class_id] if d.class_id < len(labels) else str(d.class_id)
            text = f"{label} {d.score:.2f}"
            cv2.putText(frame, text, (d.x1, max(d.y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
