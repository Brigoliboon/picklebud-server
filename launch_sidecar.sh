#!/usr/bin/env bash
# Launches the sidecar fully detached; returns immediately.
set -e
cd "$(dirname "$0")"
pkill -9 -f "uvicorn main:app" 2>/dev/null || true
sleep 1
nohup .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8790 \
  </dev/null >/tmp/uvicorn.log 2>&1 &
disown
sleep 2
curl -s --max-time 3 http://127.0.0.1:8790/health || echo "SERVER DOWN"