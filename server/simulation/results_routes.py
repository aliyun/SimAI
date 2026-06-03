"""Flask routes for simulation results parsing."""

import glob
import logging
import os
import re

from flask import Blueprint, request, jsonify
import math

from server.auth.auth_service import require_auth
from server.config import PROJECT_ROOT
from server.db.database import get_db
from server.simulation.visualizer import (
    parse_endtoend_csv,
    parse_console_output,
    find_ns3_output_files,
)

logger = logging.getLogger(__name__)


def _sanitize_nan(obj):
    """Replace NaN/Infinity float values with None for valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj

results_bp = Blueprint("results", __name__, url_prefix="/api/simulation/results")


# ---- Progress helpers ----

def _read_total_layers(workload_path):
    """Read total layer count from a workload file (second line)."""
    if not workload_path or not os.path.isfile(workload_path):
        return 0
    try:
        with open(workload_path) as f:
            f.readline()
            n = f.readline().strip()
            return int(n) if n.isdigit() else 0
    except Exception:
        return 0


def _read_current_layer(log_path):
    """Parse the last layer_num from a simulation log tail."""
    if not log_path or not os.path.isfile(log_path):
        return 0
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 60_000))
            tail = f.read().decode("utf-8", errors="replace")
        nums = re.findall(r"layer_num is:\s*(\d+)", tail)
        return int(nums[-1]) if nums else 0
    except Exception:
        return 0


@results_bp.route("/progress/<int:pid>", methods=["GET"])
@require_auth
def api_progress(pid):
    """Return simulation progress with estimated remaining time."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT pid, command, status, started_at, workspace_dir "
            "FROM processes WHERE pid = ?", (pid,),
        ).fetchone()

    if not row:
        return jsonify({"error": "Process not found"}), 404

    import time as _time
    elapsed = _time.time() - (row["started_at"] or 0)

    cmd = row["command"] or ""
    wl_match = re.search(r"(?:-w|--workload)\s+(\S+)", cmd)
    wl_path = wl_match.group(1) if wl_match else ""
    total = _read_total_layers(wl_path)

    ws = row["workspace_dir"]
    log_path = os.path.join(ws, "logs", f"{pid}.log") if ws else ""
    current = _read_current_layer(log_path)

    pct = round(current / total * 100, 1) if total and current else 0
    est = 0
    if current and total and elapsed > 1:
        rate = current / elapsed
        est = (total - current) / rate if rate > 0 else 0

    return jsonify({
        "pid": str(pid),
        "status": row["status"],
        "total_layers": total,
        "current_layer": current,
        "pct": pct,
        "elapsed_sec": round(elapsed),
        "estimated_remaining_sec": round(est),
    })


# ---- Route definitions ----


@results_bp.route("/parse-endtoend", methods=["POST"])
@require_auth
def api_parse_endtoend():
    """Parse EndToEnd CSV content and return structured data."""
    data = request.get_json(silent=True) or {}
    content = data.get("content", "")
    filepath = data.get("filepath", "")
    workload_path = data.get("workload_path", "")

    # Validate workload_path is within project root to prevent path traversal
    if workload_path:
        abs_wl = os.path.abspath(workload_path)
        if not abs_wl.startswith(PROJECT_ROOT):
            workload_path = ""

    if filepath and os.path.isfile(filepath):
        try:
            result = parse_endtoend_csv(filepath, workload_path=workload_path)
            return jsonify(_sanitize_nan(result))
        except Exception as e:
            logger.exception("EndToEnd CSV parsing failed")
            return jsonify({"error": str(e)}), 500

    if content:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = parse_endtoend_csv(tmp_path)
            return jsonify(_sanitize_nan(result))
        except Exception as e:
            logger.exception("EndToEnd CSV parsing failed")
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(tmp_path)

    return jsonify({"error": "Either 'content' or 'filepath' is required"}), 400


@results_bp.route("/parse-console", methods=["POST"])
@require_auth
def api_parse_console():
    """Parse console output lines and extract metrics."""
    data = request.get_json(silent=True) or {}
    log_lines = data.get("log_lines", [])

    if not log_lines:
        return jsonify({"error": "log_lines is required (array of strings)"}), 400

    try:
        result = parse_console_output(log_lines)
        return jsonify(result)
    except Exception as e:
        logger.exception("Console output parsing failed")
        return jsonify({"error": str(e)}), 500


@results_bp.route("/find-files", methods=["GET"])
@require_auth
def api_find_files():
    """Find simulation output files in result directories."""
    result_path = request.args.get("path", "")
    workspace = getattr(request, "workspace_dir", "")

    # Search workspace first, then provided path
    search_path = result_path or workspace or "."

    try:
        files = find_ns3_output_files(search_path)
        return jsonify({"files": files})
    except Exception as e:
        logger.exception("File search failed")
        return jsonify({"error": str(e)}), 500


def _extract_result_prefix(command: str) -> str:
    """Extract the -r / --result prefix from a command string.

    Uses the LAST occurrence because AstraParamParse processes args
    left-to-right, and the frontend may append its own -r after the
    placeholder -r that process_service provides.
    """
    matches = re.findall(r"(?:-r|--result)\s+(\S+)", command)
    return matches[-1] if matches else ""


def _is_ns3_command(command: str) -> bool:
    """NS3 binaries hardcode RESULT_PATH='./ncclFlowModel_' in their main and
    write to CWD. Their command line never carries -r/-o for the result path.
    """
    if not command:
        return False
    return "/SimAI_simulator" in command or "/SimAI_simulator_oxc" in command


def _find_ns3_result_files_in_workspace(workspace_dir: str, prefix: str = "") -> dict:
    """Locate NS3 EndToEnd CSV files for a specific task in the workspace.

    Matches files whose name contains the task's result prefix to avoid
    returning another task's output when the workspace is shared.
    Falls back to any *EndToEnd*.csv if no prefix match.
    """
    found: dict = {}
    if not workspace_dir or not os.path.isdir(workspace_dir):
        return found
    candidates: list = []
    for filename in os.listdir(workspace_dir):
        if not filename.endswith(".csv"):
            continue
        if "EndToEnd" not in filename and "detailed_" not in filename:
            continue
        path = os.path.join(workspace_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) == 0:
                continue
        except OSError:
            continue
        score = 2  # default
        if prefix and prefix in filename:
            score = 0  # exact prefix match
        elif filename.startswith("ncclFlowModel_") or filename.startswith("sim_result_"):
            score = 1  # generic name, fallback
        # Use mtime for tie-breaking: newest file first
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        candidates.append((score, -mtime, path))
    candidates.sort()
    # Strip the mtime tiebreaker
    candidates = [(s, p) for s, _, p in candidates]
    # Deduplicate by path
    seen = set()
    deduped = []
    for s, p in candidates:
        if p not in seen:
            seen.add(p)
            deduped.append((s, p))
    candidates = deduped
    # When an exact prefix was provided, only accept exact matches (score=0).
    # Avoid returning wrong results for tasks whose real output was lost/crashed.
    if prefix:
        candidates = [(s, p) for s, p in deduped if s == 0]
    else:
        candidates = deduped
    for _score, fpath in candidates:
        fname = os.path.basename(fpath)
        if "EndToEnd" in fname and "endtoend" not in found:
            found["endtoend"] = fpath
        elif "detailed_" in fname and "detailed" not in found:
            found["detailed"] = fpath
        if len(found) >= 2:
            break
    return found


def _extract_workload_path(command: str) -> str:
    """Extract the -w / --workload path from a command string."""
    match = re.search(r"(?:-w|--workload)\s+(\S+)", command)
    return match.group(1) if match else ""


def _find_result_files_for_prefix(prefix: str) -> dict:
    """Find EndToEnd CSV files matching a result prefix in PROJECT_ROOT/results/.

    Only returns files whose filename starts with the exact prefix basename.
    When no prefix is provided, returns nothing — avoids accidentally matching
    stale result files from previous runs.
    """
    results_dir = os.path.join(PROJECT_ROOT, "results")
    found: dict = {}
    if not os.path.isdir(results_dir):
        return found

    # prefix may be a path like "results/foo-" — extract just the filename part
    prefix_basename = os.path.basename(prefix) if prefix else ""
    if not prefix_basename:
        return found

    for filename in os.listdir(results_dir):
        filepath = os.path.join(results_dir, filename)
        if not os.path.isfile(filepath):
            continue
        # Exact prefix match: {prefix_basename}EndToEnd.csv
        if filename.startswith(prefix_basename) and "EndToEnd" in filename and filename.endswith(".csv"):
            found["endtoend"] = filepath
        elif filename.startswith(prefix_basename) and filename.startswith("detailed_") and filename.endswith(".csv"):
            found["detailed"] = filepath

    return found


@results_bp.route("/list-tasks", methods=["GET"])
@require_auth
def api_list_tasks():
    """List finished and error simulation tasks with their result files."""
    username = request.username

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, pid, command, status, started_at, finished_at, return_code, error_message, workspace_dir "
            "FROM processes WHERE username = ? AND status IN ('finished', 'error') "
            "ORDER BY finished_at DESC LIMIT 20",
            (username,),
        ).fetchall()

    tasks = []
    # Also scan all EndToEnd files in results/ for unmatched results
    results_dir = os.path.join(PROJECT_ROOT, "results")
    matched_files = set()

    for row in rows:
        cmd = row["command"] or ""
        workload_path = _extract_workload_path(cmd) if cmd else ""

        if _is_ns3_command(cmd):
            # Reconstruct exact output prefix from started_at (process_service uses int(time.time())).
            started = row["started_at"]
            prefix = f"sim_result_{int(started)}_" if started else "sim_result_"
            result_files = _find_ns3_result_files_in_workspace(row["workspace_dir"], prefix)
        else:
            prefix = _extract_result_prefix(cmd)
            result_files = _find_result_files_for_prefix(prefix)

        for path in result_files.values():
            matched_files.add(path)

        tasks.append({
            "tracking_id": row["id"],
            "pid": row["pid"],
            "command": row["command"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "return_code": row["return_code"],
            "error_message": row["error_message"],
            "result_prefix": prefix,
            "workload_path": workload_path,
            "result_files": result_files,
        })

    # Add standalone result files not matched to any process
    if os.path.isdir(results_dir):
        for filename in sorted(os.listdir(results_dir), reverse=True):
            filepath = os.path.join(results_dir, filename)
            if filepath in matched_files:
                continue
            if "EndToEnd" in filename and filename.endswith(".csv"):
                tasks.append({
                    "tracking_id": None,
                    "pid": None,
                    "command": None,
                    "status": "finished",
                    "started_at": os.path.getmtime(filepath),
                    "finished_at": os.path.getmtime(filepath),
                    "result_prefix": filename.replace("EndToEnd.csv", ""),
                    "workload_path": "",
                    "result_files": {"endtoend": filepath},
                })

    return jsonify({"tasks": tasks})
