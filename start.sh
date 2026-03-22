#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python3"

BACKEND_PID=""
FRONTEND_PID=""

print_header() {
    echo "======================================"
    echo "  Gesture Recognition System"
    echo "======================================"
}

require_file() {
    local path="$1"
    local hint="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing required file: $path"
        echo "$hint"
        exit 1
    fi
}

cleanup() {
    echo ""
    echo "Shutting down..."
    if [[ -n "$BACKEND_PID" ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -n "$FRONTEND_PID" ]]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait || true
    echo "Done."
}

trap cleanup SIGINT SIGTERM EXIT

print_header

require_file "$VENV_PYTHON" "Create the virtual environment first: python3 -m venv venv"
require_file "$FRONTEND_DIR/package.json" "Frontend dependencies are missing from the repository layout."

echo ""
echo "[1/2] Starting FastAPI backend on port 8000..."
cd "$BACKEND_DIR"

# Keep backend imports scoped to backend/ so the vendored root ultralytics package does not shadow runtime dependencies.
PYTHONPATH="$BACKEND_DIR" "$VENV_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

echo "  Waiting for backend to become ready..."
for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        echo "  Backend is ready."
        break
    fi
    sleep 1
done

echo ""
echo "[2/2] Starting React frontend on port 5173..."
cd "$FRONTEND_DIR"
npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "======================================"
echo "  Both servers are running"
echo ""
echo "  Frontend:  http://localhost:5173"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both servers"
echo "======================================"

wait
