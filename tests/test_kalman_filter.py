"""Unit tests for the constant-velocity Kalman filter used by the ball tracker."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman_filter import KalmanFilter


def test_uninitialized():
    kf = KalmanFilter()
    assert not kf.initialized
    assert kf.predict() is None


def test_init_sets_position():
    kf = KalmanFilter()
    pos = kf.update(10.0, 20.0)
    assert kf.initialized
    assert pos == (10.0, 20.0)
    assert kf.last_prediction == (10.0, 20.0)


def test_update_returns_near_measurement():
    kf = KalmanFilter(dt=1.0, process_noise=1e-2, measurement_noise=1e-1)
    kf.update(0.0, 0.0)
    pos = kf.update(10.0, 0.0)
    x, y = pos
    assert 8.0 <= x <= 12.0  # filtered point lands near the measurement
    assert abs(y) < 1.0


def test_tracks_straight_line_with_misses():
    kf = KalmanFilter(dt=1.0, process_noise=1e-3, measurement_noise=5e-2)
    xs = np.linspace(0, 100, 11)
    ys = np.linspace(0, 50, 11)
    for x, y in zip(xs, ys):
        kf.track(x, y)

    # 3 consecutive missed frames: filter should predict further along the line.
    px, py = kf.track(None, None)
    assert px > xs[-1]  # continues moving right
    assert py > ys[-1]  # continues moving down


def test_predict_monotonic_after_velocity_established():
    kf = KalmanFilter(dt=1.0, process_noise=1e-3, measurement_noise=1e-1)
    for i in range(5):
        kf.track(float(i), float(i))
    prev_x = kf.last_prediction[0]
    for _ in range(5):
        kf.track(None, None)
        assert kf.last_prediction[0] > prev_x
        prev_x = kf.last_prediction[0]


if __name__ == "__main__":
    for fn in [test_uninitialized, test_init_sets_position, test_update_returns_near_measurement,
               test_tracks_straight_line_with_misses, test_predict_monotonic_after_velocity_established]:
        fn()
        print(f"PASS {fn.__name__}")
    print("All kalman tests passed")
