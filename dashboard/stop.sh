#!/usr/bin/env bash
PID_FILE="${TMPDIR:-/tmp}/simai_dashboard.pids"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No running instance found."
    exit 0
fi

read -r BACKEND_PID FRONTEND_PID < "$PID_FILE"

for PID in $BACKEND_PID $FRONTEND_PID; do
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo "Stopped PID $PID"
    fi
done

rm -f "$PID_FILE"
echo "All services stopped."
