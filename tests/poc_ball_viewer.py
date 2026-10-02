"""POC ball tracker viewer.

Loads the ball tracker ONNX model (YOLOv8 detect, class 'pickleball') and shows
the detected ball box + center over live or recorded footage.

Usage:
    python poc_ball_viewer.py                     # webcam 0
    python poc_ball_viewer.py --source video.mp4  # video file
    python poc_ball_viewer.py --source frame.jpg  # single image
    python poc_ball_viewer.py --conf 0.25         # tuning
    python poc_ball_viewer.py --no-kalman         # disable Kalman smoothing
    python poc_ball_viewer.py --headless --outdir out/

Keys (interactive):
    q / Esc   quit
    s         save current annotated frame
"""

import sys
from pathlib import Path

from viewer_cli import run

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "model" / "ball_tracker.onnx"


def main() -> int:
    return run(MODEL_PATH, title="Ball Tracker POC",
               description="Ball tracker POC viewer", use_kalman=True)


if __name__ == "__main__":
    sys.exit(main())
