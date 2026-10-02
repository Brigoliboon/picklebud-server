"""Constant-velocity Kalman filter for 2D ball tracking.

State vector: [x, y, vx, vy]. Designed to be lifted into the production
`models/ball_tracker.py` later — the POC viewer uses it to smooth noisy ONNX
detections and to coast (predict) through frames where the detector reports no
ball.

Tuning knobs:
  process_noise      (Q scale)   — how much we trust the motion model; higher =
                                   more aggressive smoothing/prediction.
  measurement_noise  (R scale)   — how much we trust the detector; higher =
                                   more smoothing, slower reaction to real moves.
  dt                 per step     — time delta used in the motion model.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    def __init__(self, dt: float = 1.0, process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1):
        self.dt = float(dt)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        self.Q = process_noise * np.eye(4) * np.array([0.5 * self.dt**2, 0.5 * self.dt**2, self.dt, self.dt])**2
        self.R = measurement_noise * np.eye(2)
        self.P = np.eye(4) * 100.0
        self.x: np.ndarray | None = None
        self.last_measurement: tuple[float, float] | None = None
        self.last_prediction: tuple[float, float] | None = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def init(self, x: float, y: float) -> tuple[float, float]:
        self.x = np.array([float(x), float(y), 0.0, 0.0])
        self.last_measurement = (float(x), float(y))
        self.last_prediction = (float(x), float(y))
        return (float(x), float(y))

    def predict(self) -> tuple[float, float] | None:
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.last_prediction = (float(self.x[0]), float(self.x[1]))
        return self.last_prediction

    def update(self, mx: float, my: float) -> tuple[float, float]:
        if self.x is None:
            return self.init(mx, my)
        z = np.array([float(mx), float(my)])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.last_measurement = (float(mx), float(my))
        self.last_prediction = (float(self.x[0]), float(self.x[1]))
        return self.last_prediction

    def track(self, mx: float | None, my: float | None) -> tuple[float, float] | None:
        """One full step (predict then correct). Feed a measurement or None."""
        if self.x is None:
            if mx is None or my is None:
                return None
            return self.init(mx, my)
        self.predict()
        if mx is None or my is None:
            return self.last_prediction
        return self.update(mx, my)
