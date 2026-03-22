#!/bin/bash
# ==============================================
#  Gesture Recognition System - Startup Script
#  Usage: ./start.sh
# ==============================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv/bin"

echo "======================================"
echo "  Gesture Recognition System"
echo "======================================"

# --- 1. Start Backend ---
echo ""
echo "[1/2] Starting FastAPI backend on port 8000..."
cd "$PROJECT_DIR/backend"

# Use PYTHONPATH to avoid local ultralytics/ folder conflict
PYTHONPATH="$PROJECT_DIR/backend" "$VENV/python3" -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "  Waiting for backend to start..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
        echo "  Backend is ready!"
        break
    fi
    sleep 1
done

# --- 2. Start Frontend ---
echo ""
echo "[2/2] Starting React frontend on port 5173..."
cd "$PROJECT_DIR/frontend"
npx vite --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "======================================"
echo "  Both servers are running!"
echo ""
echo "  Frontend:  http://localhost:5173"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop all servers"
echo "======================================"

# Trap Ctrl+C to kill both processes
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
