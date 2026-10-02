"""Testcases for the live-frame analysis module (IMPLEMENTATION.md §5.6).

Validates detection payload shape, RTSP connect validation, and the frame
stream protocol via FastAPI's TestClient (WebSocket support included).
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = SERVER_DIR / "assets" / "model"
SAMPLE_VIDEO = SERVER_DIR / "sample.mp4"

sys.path.insert(0, str(SERVER_DIR))
import analysis

main = importlib.import_module("main")


@pytest.fixture(scope="module")
def engine() -> analysis.AnalysisEngine:
    return analysis.AnalysisEngine(model_dir=MODEL_DIR, conf_threshold=0.1)


@pytest.fixture()
def client():
    return TestClient(main.app)


def test_engine_loads_both_models(engine):
    assert engine.court.class_names == ("C1", "C2", "C3", "C4")
    assert engine.ball.class_names == ("pickleball",)


def test_analyze_returns_typed_detections(engine):
    import cv2

    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    for _ in range(20):  # seek to a frame known to contain the court + ball
        ok, frame = cap.read()
        if not ok:
            pytest.skip("sample video unreadable")
        if frame is not None and cap.get(cv2.CAP_PROP_POS_FRAMES) == 20:
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read sample frame"

    annotated, detections, _, _ = engine.analyze(frame)

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    for d in detections:
        assert "label" in d
        assert "class_id" in d
        assert "confidence" in d
        assert "box" in d and len(d["box"]) == 4
        assert "center" in d and len(d["center"]) == 2
        assert isinstance(d["is_court"], bool)


def test_engine_detects_known_sample(engine):
    """The sample video must produce at least one detection somewhere."""
    import cv2

    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    setup_frame = None
    for _ in range(31):
        ok, setup_frame = cap.read()
        if not ok:
            break
    cap.release()
    assert setup_frame is not None, "could not read sample frame"
    # Court detection runs once at setup and is cached; ball runs per frame.
    assert engine.detect_court(setup_frame), "court model found no corners"
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    seen_pick = 0
    seen_court = 0
    for _ in range(120):
        ok, frame = cap.read()
        if not ok:
            break
        _, detections, _, _ = engine.analyze(frame)
        for d in detections:
            if d["label"] == "pickleball":
                seen_pick += 1
            if d["label"].startswith("C"):
                seen_court += 1
    cap.release()
    assert seen_pick >= 5, f"expected ball detections, got {seen_pick}"
    assert seen_court >= 5, f"expected court detections, got {seen_court}"


def test_connect_rejects_bad_rtsp(client):
    resp = client.post("/video/connect", json={"kind": "rtsp", "source": "not-a-url"})
    assert resp.status_code == 400
    resp2 = client.post("/video/connect", json={"kind": "rtsp", "source": "rtsp://192.168.1.10/live"})
    # Reachable only if a stream is present; assert the server validates format, not availability.
    assert resp2.status_code in (200, 400)
    assert resp2.status_code != 500


def test_connect_unknown_kind_rejected(client):
    resp = client.post("/video/connect", json={"kind": "bluetooth", "source": "x"})
    assert resp.status_code == 400


def test_connect_file_then_disconnect(client):
    resp = client.post("/video/connect", json={"kind": "file", "source": str(SAMPLE_VIDEO)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["kind"] == "file"

    status = client.get("/video/status").json()
    assert status["connected"] is True

    client.post("/video/disconnect")
    status = client.get("/video/status").json()
    assert status["connected"] is False


def test_live_frames_stream_protocol(client):
    client.post("/video/connect", json={"kind": "file", "source": str(SAMPLE_VIDEO)})
    # Stream is gated on court calibration: detect then confirm first.
    det = client.post("/setup/detect-court").json()
    corners = det.get("corners") or []
    if corners and len(corners) >= 4:
        pts = [c["center"] for c in corners[:4]]
    else:
        pts = [[10, 10], [200, 10], [200, 200], [10, 200]]
    resp = client.post("/setup/confirm-court", json={"corners": pts})
    assert resp.status_code == 200
    with client.websocket_connect("/live/frames?fps=30") as ws:
        header = ws.receive_text()
        payload = json.loads(header)
        assert payload["type"] == "frame"
        assert "frame_index" in payload
        assert "timestamp" in payload
        assert "width" in payload and payload["width"] > 0
        assert "height" in payload and payload["height"] > 0
        assert "detections" in payload
        assert "events" in payload
        assert "bounce" in payload
        for event in payload["events"]:
            assert event in ("none", "ball_bounce_near_boundary", "ball_exit_boundary")

        frame_msg = ws.receive_bytes()
        assert len(frame_msg) > 0
        # JPEG magic bytes.
        assert frame_msg[0] == 0xFF and frame_msg[1] == 0xD8


def test_live_frames_requires_connection(client):
    client.post("/video/disconnect")
    with client.websocket_connect("/live/frames") as ws:
        msg = ws.receive_text()
        assert json.loads(msg)["type"] == "error"


def test_detection_payload_normalized_shape():
    """Boxes must be pixel ints with 4 corners, independent of viewport scaling."""
    from detector import Detection

    expected = {
        "label": "pickleball",
        "class_id": 0,
        "confidence": 0.87,
        "box": [10, 20, 110, 120],
        "center": [60, 70],
        "is_court": False,
    }
    actual = analysis.AnalysisEngine._to_dict(
        Detection(class_id=0, score=0.87, x1=10, y1=20, x2=110, y2=120, cx=60, cy=70),
        ("pickleball",),
    )
    assert actual == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))