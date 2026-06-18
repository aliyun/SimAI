#!/bin/bash
# SimAI Dashboard 一键启动脚本
# Usage: ./start_dashboard.sh [-t topology_dir] [-data server_port] [-gui dashboard_port]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 默认值
TOPOLOGY_DIR="$PROJECT_ROOT"
SERVER_PORT=5001
DASHBOARD_PORT=3000

usage() {
    echo "Usage: $0 [-t topology_dir] [-data server_port] [-gui dashboard_port]"
    echo ""
    echo "Options:"
    echo "  -t      Topology XML directory (default: project root)"
    echo "  -data   Server port            (default: 5001)"
    echo "  -gui    Dashboard port         (default: 3000)"
    echo "  -h      Show this help"
    echo ""
    echo "Example:"
    echo "  $0 -t /data/topology -data 5001 -gui 3000"
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -t)     TOPOLOGY_DIR="$(cd "$2" && pwd)"; shift 2 ;;
        -data)  SERVER_PORT="$2"; shift 2 ;;
        -gui)   DASHBOARD_PORT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# 检查拓扑目录
if [ ! -d "$TOPOLOGY_DIR/topology" ]; then
    echo "[WARN] $TOPOLOGY_DIR/topology not found, server may not find XML files"
fi

# 清理函数：退出时杀掉子进程
cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$DASHBOARD_PID" ] && kill "$DASHBOARD_PID" 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1. 启动 Server
EXISTING=$(lsof -ti :"$SERVER_PORT" 2>/dev/null)
if [ -n "$EXISTING" ]; then
    echo "Killing stale server on port $SERVER_PORT (PID: $EXISTING)..."
    kill $EXISTING 2>/dev/null
    sleep 1
fi
echo "Starting server on port $SERVER_PORT (topology: $TOPOLOGY_DIR) ..."
cd "$PROJECT_ROOT"
SIMAI_SERVER_PORT="$SERVER_PORT" python3 -m server.app -t "$TOPOLOGY_DIR" > ${TMPDIR:-/tmp}/simai_server.log 2>&1 &
SERVER_PID=$!

# 等待 server 就绪
for i in $(seq 1 10); do
    if curl -s "http://localhost:$SERVER_PORT/healthz" > /dev/null 2>&1; then
        echo "  Server ready (PID: $SERVER_PID)"
        break
    fi
    if [ "$i" -eq 10 ]; then
        echo "  [ERROR] Server failed to start. Check ${TMPDIR:-/tmp}/simai_server.log"
        cat ${TMPDIR:-/tmp}/simai_server.log
        exit 1
    fi
    sleep 1
done

# 2. 启动 Dashboard
echo "Starting dashboard on port $DASHBOARD_PORT ..."
cd "$PROJECT_ROOT/dashboard"
VITE_API_BASE_URL="http://localhost:$SERVER_PORT" npx vite --port "$DASHBOARD_PORT" > ${TMPDIR:-/tmp}/simai_dashboard.log 2>&1 &
DASHBOARD_PID=$!

# 等待 dashboard 就绪
for i in $(seq 1 10); do
    if curl -s "http://localhost:$DASHBOARD_PORT" > /dev/null 2>&1; then
        echo "  Dashboard ready (PID: $DASHBOARD_PID)"
        break
    fi
    # vite 可能换了端口
    ACTUAL_PORT=$(grep -oE 'localhost:[0-9]+' ${TMPDIR:-/tmp}/simai_dashboard.log 2>/dev/null | head -1 | cut -d: -f2)
    if [ -n "$ACTUAL_PORT" ] && [ "$ACTUAL_PORT" != "$DASHBOARD_PORT" ]; then
        DASHBOARD_PORT="$ACTUAL_PORT"
        echo "  [INFO] Port in use, vite switched to $DASHBOARD_PORT"
        break
    fi
    sleep 1
done

echo ""
echo "=========================================="
echo "  SimAI Dashboard"
echo "  Server:    http://localhost:$SERVER_PORT"
echo "  Dashboard: http://localhost:$DASHBOARD_PORT"
echo "  Topology:  $TOPOLOGY_DIR"
echo "=========================================="
echo "Press Ctrl+C to stop"
echo ""

# 保持前台运行
wait
