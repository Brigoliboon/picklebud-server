"""Testcases for court calibration gate (IMPLEMENTATION.md §5.1, M1).

Expected behavior:
- POST /setup/detect-court proposes 4 corners, does NOT auto-confirm.
- GET /setup/snapshot returns a JPEG still for the calibration UI.
- POST /setup/confirm-court accepts exactly 4 referee-adjusted [x, y]
  points, caches them, marks court calibrated.
- GET /setup/court reflects proposed vs confirmed corners.
- GET /match/phase reports IDLE / SETUP / LIVE_TRACKING.
- WS /live/frames refuses to stream until court is confirmed.
- POST /match/recalibrate clears confirmation without dropping source.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = SERVER_DIR / "sample.mp4"

sys.path.insert(0, str(SERVER_DIR))

import analysis

main = importlib.import_module("main")


@pytest.fixture()
def client():
    return TestClient(main.app)


def _connect(client):
    resp = client.post("/video/connect", json={"kind": "file", "source": str(SAMPLE_VIDEO)})
    assert resp.status_code == 200, resp.text


def test_detect_proposes_without_confirming(client):
    _connect(client)
    client.post("/video/disconnect")
    _connect(client)
    resp = client.post("/setup/detect-court")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "court_ready" in body
    assert "corners" in body
    # Detection must not auto-confirm: phase stays SETUP, stream gated.
    phase = client.get("/match/phase").json()["phase"]
    assert phase == "SETUP"
    with client.websocket_connect("/live/frames") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "calibrat" in msg["detail"].lower()


def test_snapshot_requires_connection(client):
    client.post("/video/disconnect")
    resp = client.get("/setup/snapshot")
    assert resp.status_code == 400
    _connect(client)
    resp = client.get("/setup/snapshot")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content[:2] == b"\xff\xd8"


def test_confirm_validates_four_points(client):
    _connect(client)
    for bad in ([], [[0, 0]] * 3, [[0, 0]] * 5, "nope"):
        resp = client.post("/setup/confirm-court", json={"corners": bad})
        assert resp.status_code in (400, 422), bad
    resp = client.post(
        "/setup/confirm-court",
        json={"corners": [[10, 10], [100, 10], [100, 100], [10, 100]]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "calibrated"


def test_full_gate_flow_connect_detect_confirm_stream(client):
    _connect(client)
    snap = client.get("/setup/snapshot")
    assert snap.status_code == 200
    det = client.post("/setup/detect-court").json()
    corners = det.get("corners") or []
    if corners and len(corners) >= 4:
        pts = [c["center"] for c in corners[:4]]
    else:
        pts = [[10, 10], [200, 10], [200, 200], [10, 200]]
    resp = client.post("/setup/confirm-court", json={"corners": pts})
    assert resp.status_code == 200
    assert client.get("/match/phase").json()["phase"] == "LIVE_TRACKING"
    court = client.get("/setup/court").json()
    assert court["calibrated"] is True
    assert len(court["corners"]) == 4
    with client.websocket_connect("/live/frames?fps=30") as ws:
        header = json.loads(ws.receive_text())
        assert header["type"] == "frame"


def test_corner_numbering_is_top_pair_bottom_pair():
    """Numbering is 1=TR, 2=TL, 3=BR, 4=BL; edges never form an X."""
    order = analysis.AnalysisEngine.order_corners
    # Bowtie-ordered input is untangled.
    assert order([(10, 10), (100, 100), (100, 10), (10, 100)]) == [
        (100, 10),
        (10, 10),
        (100, 100),
        (10, 100),
    ]
    # Perspective trapezoid (far sideline shorter) keeps its shape.
    trapezoid = [(90, 90), (40, 10), (10, 90), (60, 10)]
    assert order(trapezoid) == [(60, 10), (40, 10), (90, 90), (10, 90)]
    # Already-numbered input is stable.
    quad = [(100, 10), (10, 10), (100, 100), (10, 100)]
    assert order(quad) == quad


def test_confirm_normalizes_crossed_corners(client):
    _connect(client)
    resp = client.post(
        "/setup/confirm-court",
        json={"corners": [[10, 10], [100, 100], [100, 10], [10, 100]]},
    )
    assert resp.status_code == 200
    court = client.get("/setup/court").json()
    assert court["corners"] == [[100, 10], [10, 10], [100, 100], [10, 100]]


def test_preview_streams_without_calibration(client):
    _connect(client)
    with client.websocket_connect("/live/preview?fps=30") as ws:
        header = json.loads(ws.receive_text())
        assert header["type"] == "preview"
        assert header["detections"] == []
        assert header["events"] == []
        assert header["width"] > 0
        frame = ws.receive_bytes()
        assert frame[:2] == b"\xff\xd8"


def test_preview_sends_every_frame_in_order(client):
    """Paced playback: no skipped frames, consecutive indices at source rate."""
    _connect(client)
    with client.websocket_connect("/live/preview?fps=30") as ws:
        indices = []
        start = __import__("time").monotonic()
        for _ in range(5):
            header = json.loads(ws.receive_text())
            assert header["type"] == "preview"
            indices.append(header["frame_index"])
            ws.receive_bytes()
        elapsed = __import__("time").monotonic() - start
    assert indices == [0, 1, 2, 3, 4]
    # 5 frames at ~30fps ≈ 0.17s; unpaced would finish in milliseconds.
    assert elapsed >= 0.1


def test_preview_requires_connection(client):
    client.post("/video/disconnect")
    with client.websocket_connect("/live/preview") as ws:
        assert json.loads(ws.receive_text())["type"] == "error"


def test_analysis_starts_from_current_frame(client):
    """Confirm must not rewind: stream continues where preview left off."""
    _connect(client)
    with client.websocket_connect("/live/preview?fps=30") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "preview"
        ws.receive_bytes()
    pos_after_preview = main._active_capture.get(
        __import__("cv2").CAP_PROP_POS_FRAMES
    )
    assert pos_after_preview > 0
    client.post(
        "/setup/confirm-court",
        json={"corners": [[10, 10], [100, 10], [100, 100], [10, 100]]},
    )
    pos_after_confirm = main._active_capture.get(
        __import__("cv2").CAP_PROP_POS_FRAMES
    )
    assert pos_after_confirm == pos_after_preview


def test_recalibrate_keeps_source_but_gates_stream(client):
    _connect(client)
    client.post(
        "/setup/confirm-court",
        json={"corners": [[10, 10], [100, 10], [100, 100], [10, 100]]},
    )
    assert client.get("/match/phase").json()["phase"] == "LIVE_TRACKING"
    resp = client.post("/match/recalibrate")
    assert resp.status_code == 200
    assert client.get("/video/status").json()["connected"] is True
    assert client.get("/match/phase").json()["phase"] == "SETUP"
    with client.websocket_connect("/live/frames") as ws:
        assert json.loads(ws.receive_text())["type"] == "error"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
