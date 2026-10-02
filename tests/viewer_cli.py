"""Shared CLI runner for the POC viewers.

Handles argument parsing, frame iteration, annotation, display, and headless
saving for any YOLOv8 ONNX detector. Thin viewer scripts provide the model path
and window title.

Optional Kalman tracking (enable with --kalman): the best detection center is
smoothed through a constant-velocity Kalman filter; when the detector misses a
frame the filter keeps predicting along the trajectory (drawn in blue). Green =
filtered position, red = raw detection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from kalman_filter import KalmanFilter
from yolo_detector import YoloDetector, draw_detections, iter_frames


def _draw_kalman(frame, kf: KalmanFilter, measured: tuple[int, int] | None):
    if not kf.initialized:
        return
    if kf.last_measurement is not None:
        mx, my = map(int, kf.last_measurement)
        cv2.circle(frame, (mx, my), 8, (0, 0, 255), 2)  # raw measurement
    px, py = map(int, kf.last_prediction)
    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)  # filtered position
    if measured is None:
        cv2.putText(frame, "PREDICTING", (px + 12, py), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 0, 0), 2)


def run(model_path: Path, title: str, description: str, use_kalman: bool = False) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", default="0", help="webcam index, video path, image path, or dir")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--headless", action="store_true", help="no display window")
    parser.add_argument("--outdir", default=None, help="save annotated frames here (headless)")
    parser.add_argument("--kalman", action="store_true", help="enable Kalman smoothing/prediction")
    parser.add_argument("--no-kalman", action="store_true", help="disable Kalman (overrides default)")
    args = parser.parse_args()

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    detector = YoloDetector(model_path, args.conf, args.nms)
    print(f"Loaded {model_path.name} | classes: {detector.class_names}")
    print(f"Providers: {detector.session.get_providers()}")

    kf = KalmanFilter() if (use_kalman or args.kalman) and not args.no_kalman else None
    if kf:
        print("Kalman filter: ON")

    outdir = Path(args.outdir) if args.outdir else None
    if outdir and not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    for frame, is_video in iter_frames(args.source):
        if isinstance(frame, str):
            img = cv2.imread(frame)
            if img is None:
                print(f"Skipping unreadable image: {frame}")
                continue
        else:
            img = frame

        dets = detector.detect(img)
        annotated = draw_detections(img, dets, detector.class_names)
        frame_idx += 1

        if kf:
            best = max(dets, key=lambda d: d.score) if dets else None
            measured = (best.cx, best.cy) if best else None
            kf.track(*measured) if measured else kf.track(None, None)
            _draw_kalman(annotated, kf, measured)

        if outdir:
            out_path = outdir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"frame {frame_idx}: saved -> {out_path}")

        if args.headless:
            continue

        h, w = annotated.shape[:2]
        if w > 1280:
            scale = 1280 / w
            annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

        cv2.imshow(title, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and not is_video:
            save_path = f"annotated_frame_{frame_idx}.jpg"
            cv2.imwrite(save_path, annotated)
            print(f"Saved {save_path}")

    cv2.destroyAllWindows()
    return 0
