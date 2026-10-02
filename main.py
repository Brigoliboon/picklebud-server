"""Pickleball analysis sidecar — FastAPI app.

Exposes a localhost-only REST API that the Flutter desktop app talks to:
health checks, video source discovery, and video source open/close. The live
feed WebSocket pushes annotated frames plus per-frame detections back to the
frontend for display and timeline rendering.

Run (dev):
    .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8790
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Iterator, Optional

import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import analysis

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

app = FastAPI(title="Pickleball Analysis Sidecar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
MAX_CAMERA_INDEX = 9
_active_capture: cv2.VideoCapture | None = None
_analysis_engine: analysis.AnalysisEngine | None = None

# Max seconds to wait for a single WebSocket send. A client that went away
# without closing (killed app, reload, timeout) would otherwise block the
# send forever and wedge the whole event loop — including /health.
_SEND_TIMEOUT = 5.0


async def _send_frame(websocket: WebSocket, header: dict, jpg: bytes | None) -> bool:
    """Send one header-plus-JPEG frame. False if the client is gone."""
    try:
        await asyncio.wait_for(websocket.send_text(json.dumps(header)), _SEND_TIMEOUT)
        if jpg:
            await asyncio.wait_for(websocket.send_bytes(jpg), _SEND_TIMEOUT)
        return True
    except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):
        return False
    except Exception:
        return False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/video/samples")
def video_samples() -> dict:
    samples: list[str] = []
    for root_dir in (SERVER_DIR, os.path.join(SERVER_DIR, "assets")):
        if not os.path.isdir(root_dir):
            continue
        for entry in sorted(os.listdir(root_dir)):
            if entry.lower().endswith(VIDEO_EXTENSIONS):
                samples.append(os.path.join(root_dir, entry))
    return {"samples": samples}


@contextlib.contextmanager
def _silence_stderr() -> Iterator[None]:
    """Redirect fd 2 to /dev/null so OpenCV/v4l probe noise stays out of logs."""
    import sys

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        sys.stderr.flush()
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


@app.get("/video/sources")
def video_sources() -> dict:
    cameras: list[int] = []
    with _silence_stderr():
        for index in range(MAX_CAMERA_INDEX + 1):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cameras.append(index)
            cap.release()
    return {"cameras": cameras}


class ConnectRequest(BaseModel):
    kind: str  # "camera" | "file" | "rtsp"
    source: str = ""


def _get_analysis_engine() -> analysis.AnalysisEngine:
    global _analysis_engine
    if _analysis_engine is None:
        _analysis_engine = analysis.AnalysisEngine()
    return _analysis_engine


@app.post("/video/connect")
def video_connect(req: ConnectRequest) -> dict:
    global _active_capture

    if _active_capture is not None:
        _active_capture.release()
        _active_capture = None

    if req.kind == "camera":
        try:
            index = int(req.source)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"invalid camera index: {req.source}")
        cap = cv2.VideoCapture(index)
    elif req.kind == "file":
        path = req.source
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"video file not found: {path}")
        cap = cv2.VideoCapture(path)
    elif req.kind == "rtsp":
        url = req.source
        if not (url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")):
            raise HTTPException(status_code=400, detail="invalid RTSP/stream URL")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    else:
        raise HTTPException(status_code=400, detail=f"unknown source kind: {req.kind}")

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="could not open video source")

    # A new source always requires fresh court calibration.
    if _analysis_engine is not None:
        _analysis_engine.clear_confirmation()
        _analysis_engine.reset_ball_track()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = total_frames if total_frames > 0 else None

    _active_capture = cap
    return {
        "connected": True,
        "kind": req.kind,
        "source": req.source,
        "width": width,
        "height": height,
        "fps": round(fps, 2) if fps and fps > 0 else None,
        "frame_count": frame_count,
    }


@app.post("/video/disconnect")
def video_disconnect() -> dict:
    global _active_capture
    if _active_capture is not None:
        _active_capture.release()
        _active_capture = None
    if _analysis_engine is not None:
        _analysis_engine.clear_confirmation()
        _analysis_engine.reset_ball_track()
    return {"disconnected": True}


@app.get("/video/status")
def video_status() -> dict:
    if _active_capture is None:
        return {"connected": False}
    width = int(_active_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(_active_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {
        "connected": True,
        "width": width,
        "height": height,
    }


@app.post("/setup/detect-court")
def detect_court() -> dict:
    """Calibration: run the court-corner model once on the next frame and cache
    the corners. Once `court_ready`, the live stream draws them and only runs
    ball tracking per frame."""
    engine = _get_analysis_engine()
    if _active_capture is None:
        raise HTTPException(status_code=400, detail="no video source connected")

    ok, frame = _active_capture.read()
    if not ok:
        raise HTTPException(status_code=400, detail="could not read a frame from the source")

    ready = engine.detect_court(frame)
    detections = [
        analysis.AnalysisEngine._to_dict(d, engine.court.class_names)
        for d in engine._court_detections
    ]
    return {
        "court_ready": ready,
        "corners": detections,
        "points": [[x, y] for x, y in engine.ordered_corner_points()],
        "width": int(_active_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(_active_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


@app.get("/setup/snapshot")
def setup_snapshot() -> Response:
    """Still JPEG of the connected source for the calibration UI.

    Reads the current frame for corner review/drag-adjust. Analysis starts
    from the source's current position — no rewinding — so cameras and
    already-playing files continue where the preview left off.
    """
    if _active_capture is None:
        raise HTTPException(status_code=400, detail="no video source connected")
    ok, frame = _active_capture.read()
    if not ok or frame is None:
        raise HTTPException(status_code=400, detail="could not read a frame from the source")
    ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok_jpg:
        raise HTTPException(status_code=500, detail="could not encode snapshot frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/setup/court")
def setup_court() -> dict:
    engine = _get_analysis_engine()
    if engine.calibrated:
        points = [[x, y] for x, y in engine._confirmed_corners or []]
    else:
        points = [[x, y] for x, y in engine.ordered_corner_points()]
    cap_w = cap_h = None
    if _active_capture is not None:
        cap_w = int(_active_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(_active_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {
        "court_ready": engine.court_ready,
        "calibrated": engine.calibrated,
        "corners": points,
        "width": cap_w,
        "height": cap_h,
    }


class ConfirmCourtRequest(BaseModel):
    corners: list[list[float]]


@app.post("/setup/confirm-court")
def confirm_court(req: ConfirmCourtRequest) -> dict:
    """Referee confirms (possibly drag-adjusted) 4 court corners.

    Gates the live ball-tracking stream: /live/frames refuses connections
    until this has succeeded for the current source.
    """
    if _active_capture is None:
        raise HTTPException(status_code=400, detail="no video source connected")
    if len(req.corners) != 4 or any(len(p) != 2 for p in req.corners):
        raise HTTPException(status_code=400, detail="exactly 4 [x, y] corner points required")
    width = int(_active_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(_active_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for p in req.corners:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="corner coordinates must be numbers")
        if x < 0 or y < 0 or x > width or y > height:
            raise HTTPException(status_code=400, detail="corner point outside frame bounds")
    engine = _get_analysis_engine()
    try:
        engine.set_confirmed_corners(req.corners)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "calibrated"}


@app.get("/match/phase")
def match_phase() -> dict:
    if _active_capture is None:
        return {"phase": "IDLE"}
    engine = _get_analysis_engine()
    if engine.calibrated:
        return {"phase": "LIVE_TRACKING"}
    return {"phase": "SETUP"}


@app.post("/match/recalibrate")
def match_recalibrate() -> dict:
    if _active_capture is None:
        raise HTTPException(status_code=400, detail="no video source connected")
    _get_analysis_engine().clear_confirmation()
    return {"phase": "SETUP"}


@app.websocket("/live/preview")
async def live_preview(websocket: WebSocket) -> None:
    """Raw preview stream: plain JPEG frames, zero inference.

    Starts immediately after connect (never gated on calibration) so cameras
    can be aimed and files inspected before analysis begins. Same
    header-plus-JPEG protocol as /live/frames, but `type` is "preview" and
    detections/events are always empty.
    """
    await websocket.accept()
    try:
        fps = int(websocket.query_params.get("fps", "10"))
    except ValueError:
        fps = 10

    if _active_capture is None:
        await websocket.send_json({"type": "error", "detail": "no video source connected"})
        await websocket.close()
        return

    cap = _active_capture
    # Realtime pacing: files serve reads as fast as the CPU allows, so an
    # unpaced loop fast-forwards through the video and drops most frames.
    # Cameras self-pace; files are slept into their own frame rate (capped
    # by the requested fps). Every read frame is sent — none skipped.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = float(fps)
    pace = min(float(src_fps), float(fps))
    interval = 1.0 / max(1.0, min(pace, 60.0))
    next_due = time.monotonic()
    frame_index = 0
    try:
        while True:
            if _active_capture is not cap:
                break  # stale stream: source reconnected/disconnected
            try:
                ok, frame = cap.read()
            except Exception:
                break
            if not ok:
                await _send_frame(websocket, {"type": "end", "frame_index": frame_index}, None)
                break
            now = time.monotonic()
            if now < next_due:
                await asyncio.sleep(next_due - now)
            else:
                next_due = now  # behind schedule: drop backlog, stay live

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            header = {
                "type": "preview",
                "frame_index": frame_index,
                "timestamp": round(timestamp, 3),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "detections": [],
                "events": [],
            }
            ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg = buf.tobytes() if ok_jpg else None
            if not await _send_frame(websocket, header, jpg):
                break  # client gone: exit so the loop never wedges
            frame_index += 1
            next_due += interval
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/live/frames")
async def live_frames(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        fps = int(websocket.query_params.get("fps", "10"))
    except ValueError:
        fps = 10

    if _active_capture is None:
        await websocket.send_json({"type": "error", "detail": "no video source connected"})
        await websocket.close()
        return

    engine = _get_analysis_engine()
    if not engine.calibrated:
        await websocket.send_json({
            "type": "error",
            "detail": "court not calibrated — confirm court corners first",
        })
        await websocket.close()
        return
    interval = 1.0 / max(1, min(fps, 30))
    last_send = 0.0
    frame_index = 0
    cap = _active_capture
    try:
        while True:
            if _active_capture is not cap:
                break  # stale stream: source reconnected/disconnected
            try:
                ok, frame = cap.read()
            except Exception:
                break
            if not ok:
                await _send_frame(websocket, {"type": "end", "frame_index": frame_index}, None)
                break
            now = time.monotonic()
            if now - last_send < interval:
                continue
            last_send = now

            # Court detection is calibration-only — trigger it via
            # POST /setup/detect-court, never inline per frame. Ball tracking
            # only here so the live stream stays fast.
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            try:
                annotated, detections, bounce, exit_info = await asyncio.to_thread(
                    engine.analyze, frame, timestamp, frame_index
                )
            except Exception:
                break
            frame_events: list[str] = []
            if bounce:
                frame_events.append("ball_bounce_near_boundary")
            if exit_info:
                frame_events.append("ball_exit_boundary")
            header = {
                "type": "frame",
                "frame_index": frame_index,
                "timestamp": round(timestamp, 3),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "detections": detections,
                "events": frame_events if frame_events else ["none"],
                "bounce": bounce,
                "exit": exit_info,
            }
            ok_jpg, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg = buf.tobytes() if ok_jpg else None
            if not await _send_frame(websocket, header, jpg):
                break  # client gone: exit so the loop never wedges
            frame_index += 1
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


