"""Shared YOLOv8 ONNX detector used by the POC viewers.

Generic detection core: letterbox preprocessing, inference, YOLO post-processing
(decode + NMS), drawing, and frame iteration. Class names are read from the
model's Ultralytics metadata (`names` field) so the same code serves both the
court-corner model (C1-C4) and the ball-tracker model (pickleball).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_IMGSZ = 640


@dataclass
class Detection:
    class_id: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int


class YoloDetector:
    def __init__(self, model_path: Path, conf_threshold: float = 0.25, nms_threshold: float = 0.45):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = self._read_class_names()

    def _read_class_names(self) -> tuple[str, ...]:
        meta = self.session.get_modelmeta()
        names = meta.custom_metadata_map.get("names")
        if names:
            try:
                import ast
                parsed = ast.literal_eval(names)
                if isinstance(parsed, dict):
                    return tuple(parsed[i] for i in sorted(parsed))
            except (ValueError, SyntaxError, TypeError):
                pass
        n = self.session.get_outputs()[0].shape[1] - 4 if self.session.get_outputs()[0].shape else 0
        return tuple(str(i) for i in range(n))

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        scale = min(MODEL_IMGSZ / h, MODEL_IMGSZ / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((MODEL_IMGSZ, MODEL_IMGSZ, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas, scale

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        letterboxed, scale = self._letterbox(frame_bgr)
        blob = letterboxed[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0]

        pred = np.squeeze(out, axis=0)
        boxes = pred[:4].T
        scores = pred[4:].T

        class_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        keep = confs >= self.conf_threshold
        boxes, class_ids, confs = boxes[keep], class_ids[keep], confs[keep]

        detections: list[Detection] = []
        if boxes.size == 0:
            return detections

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        for i in range(len(boxes)):
            detections.append(
                Detection(
                    class_id=int(class_ids[i]),
                    score=float(confs[i]),
                    x1=int(x1[i]), y1=int(y1[i]), x2=int(x2[i]), y2=int(y2[i]),
                    cx=int(cx[i]), cy=int(cy[i]),
                )
            )

        idxs = cv2.dnn.NMSBoxes(
            [(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1) for d in detections],
            [d.score for d in detections],
            self.conf_threshold,
            self.nms_threshold,
        )
        idxs = np.array(idxs).flatten() if len(idxs) else np.array([], dtype=int)
        detections = [detections[i] for i in idxs]

        inv = 1.0 / scale
        for d in detections:
            d.x1 = int(d.x1 * inv); d.y1 = int(d.y1 * inv)
            d.x2 = int(d.x2 * inv); d.y2 = int(d.y2 * inv)
            d.cx = int(d.cx * inv); d.cy = int(d.cy * inv)
        return detections


def draw_detections(frame: np.ndarray, detections: list[Detection],
                    class_names: tuple[str, ...] = ()) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        color = (0, 255, 0)
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.circle(out, (d.cx, d.cy), 6, (0, 0, 255), -1)
        name = class_names[d.class_id] if d.class_id < len(class_names) else str(d.class_id)
        label = f"{name} {d.score:.2f}"
        cv2.putText(out, label, (d.x1, max(d.y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def iter_frames(source: str):
    if os.path.isdir(source):
        for path in sorted(Path(source).glob("*")):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                yield str(path), False
        return
    lower = source.lower()
    if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        yield source, False
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, True
    cap.release()
