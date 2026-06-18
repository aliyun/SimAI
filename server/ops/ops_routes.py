"""运维工具 — 调用记录 + 日志查询."""

import json
import logging
import os
import re
from threading import Lock
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify

from server.auth.auth_service import require_auth
from server.config import WORKSPACE_ROOT

logger = logging.getLogger(__name__)
ops_bp = Blueprint("ops", __name__, url_prefix="/api/ops")

_interaction_lock = Lock()


# ---- Interaction Capture ----

def record_interaction(workspace_dir: str, entry: dict):
    """Append an interaction record to the workspace JSONL file."""
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        os.makedirs(os.path.join(workspace_dir, "ops"), exist_ok=True)
        path = os.path.join(workspace_dir, "ops", "interactions.jsonl")
        with _interaction_lock:
            with open(path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to record interaction: %s", e)


# ---- API: 调用记录 ----

@ops_bp.route("/interactions", methods=["GET"])
@require_auth
def api_interactions():
    """Return structured interaction records."""
    ws = request.workspace_dir
    entries = []
    # Scan all workspace directories for interaction logs
    for w in [ws] + [os.path.join(WORKSPACE_ROOT, d) for d in os.listdir(WORKSPACE_ROOT)
                     if os.path.isdir(os.path.join(WORKSPACE_ROOT, d, "ops"))]:
        path = os.path.join(w, "ops", "interactions.jsonl")
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            except Exception:
                pass

    algo = request.args.get("algo", "").strip()
    source = request.args.get("source", "").strip()
    if algo:
        entries = [e for e in entries if algo.lower() in (e.get("algorithm") or "").lower()]
    if source:
        entries = [e for e in entries if source.lower() == (e.get("source") or "").lower()]

    # Also scan process logs for OXC interactions
    process_entries = _parse_oxc_from_logs(ws)

    all_entries = entries + process_entries
    all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return jsonify({"interactions": all_entries[:200]})


def _parse_oxc_from_logs(ws: str) -> list:
    """Scan recent process logs for OXC integration interactions."""
    entries = []
    workspaces = [ws]
    if not os.path.isdir(os.path.join(ws, "logs")):
        workspaces.extend(
            os.path.join(WORKSPACE_ROOT, d) for d in os.listdir(WORKSPACE_ROOT)
            if os.path.isdir(os.path.join(WORKSPACE_ROOT, d, "logs"))
        )
    for w in workspaces:
        logs_dir = os.path.join(w, "logs")
        if not os.path.isdir(logs_dir):
            continue
        for logfile in sorted(os.listdir(logs_dir), reverse=True)[:5]:
            if logfile.endswith(".log"):
                entries.extend(_scan_log_for_oxc(os.path.join(logs_dir, logfile), w))
    return entries


def _scan_log_for_oxc(logpath: str, ws: str) -> list:
    """Parse OXC interactions from a simulation log tail."""
    entries = []
    try:
        with open(logpath, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200_000))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return entries

    algo_match = re.search(r"Algorithm:\s*(\S+)", tail)
    algo = algo_match.group(1) if algo_match else "unknown"
    ws_name = os.path.basename(ws)

    # OXC initialization
    if "[OXC Integration]" in tail:
        enabled = "Initialized" in tail and "disabled" not in tail.split("Initialized")[-1][:50]
        entries.append({
            "timestamp": "", "source": "OXC-HCCL", "type": "init",
            "endpoint": "internal", "method": "init",
            "algorithm": algo, "request": {}, "response": {"enabled": enabled},
            "workspace": ws_name,
        })

    # shouldUseOxc checks
    for m in re.finditer(r"shouldUseOxc:\s*ranks=\[([^\]]+)\],\s*cross_rack=(\w+)", tail):
        ranks = m.group(1)
        cross = m.group(2) == "true"
        entries.append({
            "timestamp": "", "source": "OXC-HCCL", "type": "decision",
            "endpoint": "internal", "method": "shouldUseOxc",
            "algorithm": algo, "request": {"ranks": ranks, "cross_rack": cross},
            "response": {"use_oxc": cross}, "workspace": ws_name,
        })

    # OXC flow generation
    for m in re.finditer(r"\[OXC\] Using OXC for AllReduce.*?size:\s*(\d+)", tail):
        entries.append({
            "timestamp": "", "source": "OXC-HCCL", "type": "oxc_call",
            "endpoint": "POST /api/oxc/allreduce", "method": "POST",
            "algorithm": algo, "request": {"comm_type": "ALLREDUCE", "group_size": int(m.group(1))},
            "response": {"status": "ok"}, "workspace": ws_name,
        })

    for m in re.finditer(r"\[OXC\] Generated (\d+) flows via OXC", tail):
        entries.append({
            "timestamp": "", "source": "OXC-HCCL", "type": "oxc_result",
            "endpoint": "POST /api/oxc/allreduce", "method": "POST",
            "algorithm": algo, "request": {},
            "response": {"flows_generated": int(m.group(1))}, "workspace": ws_name,
        })

    return entries


# ---- API: 日志查询 ----

@ops_bp.route("/logs", methods=["GET"])
@require_auth
def api_logs():
    """Search platform runtime logs."""
    query = request.args.get("q", "").strip()
    level = request.args.get("level", "").strip().upper()
    limit = min(int(request.args.get("limit", "100")), 500)

    results = []
    ws = request.workspace_dir
    logs_dir = os.path.join(ws, "logs")
    dirs = [logs_dir] if os.path.isdir(logs_dir) else []
    for name in os.listdir(WORKSPACE_ROOT):
        d = os.path.join(WORKSPACE_ROOT, name, "logs")
        if d not in dirs and os.path.isdir(d):
            dirs.append(d)
    for d in dirs[:3]:
        results.extend(_search_log_dir(d, query, level, limit - len(results)))
        if len(results) >= limit:
            break

    flask_log = "/tmp/flask.log"
    if os.path.isfile(flask_log):
        results.extend(_search_log_file(flask_log, query, level, limit - len(results)))

    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"logs": results[:limit], "total": len(results)})


def _search_log_dir(logs_dir: str, query: str, level: str, limit: int) -> list:
    results = []
    try:
        for fname in sorted(os.listdir(logs_dir), reverse=True)[:10]:
            if len(results) >= limit:
                break
            results.extend(_search_log_file(os.path.join(logs_dir, fname), query, level, limit - len(results)))
    except Exception:
        pass
    return results


def _search_log_file(fpath: str, query: str, level: str, limit: int) -> list:
    results = []
    try:
        with open(fpath, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 500_000))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return results

    fname = os.path.basename(fpath)
    for line in tail.split("\n"):
        if len(results) >= limit:
            break
        if not line.strip():
            continue
        if query and query.lower() not in line.lower():
            continue
        if level and level not in line.upper():
            continue
        results.append({"file": fname, "line": line.strip()[:500]})
    return results
