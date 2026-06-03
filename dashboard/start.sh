#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="${TMPDIR:-/tmp}/simai_dashboard.pids"
BACKEND_LOG="${TMPDIR:-/tmp}/simai_backend.log"
FRONTEND_LOG="${TMPDIR:-/tmp}/simai_frontend.log"

# Defaults
BACKEND_PORT=5001
DEV_MODE=false
DATA_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --data-dir <path>   XML topology directory for monitor (optional)
  --port <port>       Backend port (default: 5001)
  --dev               Use Vite dev server instead of preview mode
  -h, --help          Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --data-dir topology --dev
  $(basename "$0") --data-dir /abs/path/to/xmls --port 5002
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        --port)      BACKEND_PORT="$2"; shift 2 ;;
        --dev)       DEV_MODE=true; shift ;;
        -h|--help)   usage ;;
        *)           echo "Error: unknown option '$1'"; usage ;;
    esac
done

if [[ -n "$DATA_DIR" && "$DATA_DIR" != /* ]]; then
    DATA_DIR="$PROJECT_DIR/$DATA_DIR"
fi

if [[ -n "$DATA_DIR" && ! -d "$DATA_DIR" ]]; then
    echo "Error: data directory not found: $DATA_DIR"
    exit 1
fi

# Stop any existing instances
if [[ -f "$PID_FILE" ]]; then
    echo "Stopping previous instance..."
    "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
fi

cleanup() {
    echo ""
    echo "Shutting down..."
    "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

echo "=== SimAI Dashboard ==="
echo "Data dir:  ${DATA_DIR:-'(none)'}"
echo "Backend:   http://localhost:$BACKEND_PORT"

# --- Start backend ---
echo "Starting backend..."
cd "$PROJECT_DIR"

BACKEND_ARGS=("--port" "$BACKEND_PORT")
if [[ -n "$DATA_DIR" ]]; then
    BACKEND_ARGS+=("--topology-dir" "$DATA_DIR")
fi

python3 -m server.app "${BACKEND_ARGS[@]}" \
    > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# --- Start frontend ---
cd "$SCRIPT_DIR"
npm install --silent 2>/dev/null || true

if [[ "$DEV_MODE" == true ]]; then
    echo "Starting frontend (dev mode)..."
    VITE_API_BASE_URL="http://localhost:$BACKEND_PORT" \
        npx vite --port 3000 > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    FRONTEND_PORT=3000
else
    echo "Building frontend..."
    npx vite build >> "$FRONTEND_LOG" 2>&1
    echo "Starting frontend (preview mode)..."
    VITE_API_BASE_URL="http://localhost:$BACKEND_PORT" \
        npx vite preview --port 4173 >> "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    FRONTEND_PORT=4173
fi

# Save PIDs
echo "$BACKEND_PID $FRONTEND_PID" > "$PID_FILE"

# Wait for services to be ready
echo "Waiting for services..."
for i in $(seq 1 30); do
    BACKEND_OK=false
    FRONTEND_OK=false

    if curl -sf "http://localhost:$BACKEND_PORT/healthz" > /dev/null 2>&1; then
        BACKEND_OK=true
    fi
    if curl -sf "http://localhost:$FRONTEND_PORT/" > /dev/null 2>&1; then
        FRONTEND_OK=true
    fi

    if [[ "$BACKEND_OK" == true && "$FRONTEND_OK" == true ]]; then
        break
    fi

    # Check if processes are still alive
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Error: backend failed to start. Check $BACKEND_LOG"
        cat "$BACKEND_LOG"
        "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "Error: frontend failed to start. Check $FRONTEND_LOG"
        cat "$FRONTEND_LOG"
        "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
        exit 1
    fi

    sleep 1
done

echo ""
echo "Ready!"
echo "  Frontend:  http://localhost:$FRONTEND_PORT"
echo "  Backend:   http://localhost:$BACKEND_PORT"
echo "  Logs:      $BACKEND_LOG / $FRONTEND_LOG"
echo ""
echo "Press Ctrl+C to stop, or run: ./dashboard/stop.sh"

# Keep script alive so Ctrl+C works
wait
