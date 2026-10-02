"""POC court-corner detection viewer.

Loads the court detection ONNX model (YOLOv8 detect, classes C1-C4) and shows
the detected court corners over live or recorded footage.

Usage:
    python poc_viewer.py                          # webcam 0
    python poc_viewer.py --source video.mp4       # video file
    python poc_viewer.py --source frame.jpg       # single image
    python poc_viewer.py --conf 0.25 --nms 0.45   # tuning
    python poc_viewer.py --headless --outdir out/ # save annotated frames, no window

Keys (interactive):
    q / Esc   quit
    s         save current annotated frame
"""

import sys
from pathlib import Path

from viewer_cli import run

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "model" / "court_detection.onnx"


def main() -> int:
    return run(MODEL_PATH, title="Court Detector POC", description="Court-corner detection POC viewer")


if __name__ == "__main__":
    sys.exit(main())
