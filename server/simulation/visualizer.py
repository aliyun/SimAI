"""
Visualizer — server-side parse-only functions.
No Plotly dependency; returns raw data for React/Recharts rendering.
"""

import os
import re
from typing import Dict, List, Optional

# Communication type → group mapping (mirrors scripts/visualize_workload.py)
COMM_TYPE_TO_GROUP = {
    "ALLREDUCE": "TP", "ALLGATHER": "TP", "REDUCESCATTER": "TP", "ALLTOALL": "TP",
    "ALLREDUCEALLTOALL": "TP",
    "ALLREDUCE_EP": "EP", "ALLGATHER_EP": "EP", "REDUCESCATTER_EP": "EP",
    "ALLTOALL_EP": "EP", "ALLREDUCEALLTOALL_EP": "EP",
    "ALLREDUCE_DP_EP": "DP_EP", "ALLGATHER_DP_EP": "DP_EP",
    "REDUCESCATTER_DP_EP": "DP_EP", "ALLTOALL_DP_EP": "DP_EP",
    "ALLREDUCEALLTOALL_DP_EP": "DP_EP",
    "NONE": "NONE",
}

# For wg phase, bare ALLREDUCE/ALLGATHER/etc default to DP domain
COMM_TYPE_TO_GROUP_WG = dict(COMM_TYPE_TO_GROUP)
COMM_TYPE_TO_GROUP_WG.update({
    "ALLREDUCE": "DP", "ALLGATHER": "DP", "REDUCESCATTER": "DP",
    "ALLTOALL": "DP", "ALLREDUCEALLTOALL": "DP",
})


def _parse_workload_header(workload_path: str) -> Dict:
    """Parse just the header from a workload file to extract parallelism config.

    Returns dict with tp_size, ep_size, pp_size, dp_size, dp_ep_size, all_gpus.
    """
    result = {"tp_size": 1, "ep_size": 1, "pp_size": 1, "dp_size": 1, "dp_ep_size": 1, "all_gpus": 0}
    try:
        with open(workload_path) as f:
            header_line = f.readline().strip()
        tokens = header_line.split()
        header = {}
        for i in range(1, len(tokens) - 1, 2):
            key = tokens[i].rstrip(":")
            header[key] = tokens[i + 1]
        tp_size = int(header.get("model_parallel_NPU_group", 1))
        ep_size = int(header.get("ep", 1))
        pp_size = int(header.get("pp", 1))
        all_gpus = int(header.get("all_gpus", 0))
        dp_full = all_gpus // (tp_size * pp_size) if tp_size * pp_size > 0 else 1
        dp_ep_size = dp_full // ep_size if ep_size > 0 else 1
        result = {"tp_size": tp_size, "ep_size": ep_size, "pp_size": pp_size,
                  "dp_size": dp_full, "dp_ep_size": dp_ep_size, "all_gpus": all_gpus}
    except Exception:
        pass
    return result


def _build_layer_comm_groups(workload_path: str) -> list:
    """Parse workload file and return per-layer comm group mapping.

    Returns list of dicts (one per layer, by index): [{fp_group, ig_group, wg_group}, ...]
    The groups are determined by the comm_type suffix:
      - ALLREDUCE_EP / ALLGATHER_EP etc → EP
      - ALLREDUCE_DP_EP / ALLGATHER_DP_EP etc → DP_EP
      - bare ALLREDUCE / ALLGATHER / etc → TP (fwd/ig) or DP (wg)
    """
    groups: list = []
    try:
        with open(workload_path) as f:
            header_line = f.readline().strip()
            count_line = f.readline().strip()
            layer_lines = [l.strip() for l in f if l.strip()]
        num_layers = int(count_line)
        for line in layer_lines[:num_layers]:
            parts = line.split()
            if len(parts) < 10:
                groups.append({"fp_group": "TP", "ig_group": "TP", "wg_group": "DP"})
                continue
            fp_comm_type = parts[3]
            ig_comm_type = parts[6]
            wg_comm_type = parts[9]
            groups.append({
                "fp_group": COMM_TYPE_TO_GROUP.get(fp_comm_type, "TP"),
                "ig_group": COMM_TYPE_TO_GROUP.get(ig_comm_type, "TP"),
                "wg_group": COMM_TYPE_TO_GROUP_WG.get(wg_comm_type, "DP"),
            })
    except Exception:
        pass
    return groups


def _count_workload_layers(wl_path: str) -> int:
    """Return the layer count declared in a workload file header."""
    try:
        with open(wl_path) as f:
            f.readline()  # skip header
            return int(f.readline().strip())
    except Exception:
        return -1


def _count_endtoend_layers(result_filepath: str) -> int:
    """Count data rows in an EndToEnd.csv (excluding header/totals rows)."""
    try:
        count = 0
        with open(result_filepath) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Skip dimension rows and header row
        start = 0
        if lines and lines[0].startswith("File name"):
            start = 3  # dim row, dim values, column header
        else:
            start = 1  # just column header
        for line in lines[start:]:
            parts = line.split(",")
            if parts and parts[0] not in ("total exposed comm", "SUM"):
                count += 1
        return count
    except Exception:
        return -1


def _find_workload_for_result(result_filepath: str) -> str:
    """Find the workload file that generated a result.

    Searches in:
    1. The same directory as the result file
    2. Parent directory of the result
    3. Workspace directories under server/workspaces/ — matched by layer count
    """
    if result_filepath:
        # Try same directory
        dir_path = os.path.dirname(result_filepath)
        for candidate in ["workload.txt", "workload_mini.txt"]:
            path = os.path.join(dir_path, candidate)
            if os.path.isfile(path):
                return path
        # Try parent
        parent = os.path.dirname(dir_path)
        for candidate in ["workload.txt", "workload_mini.txt"]:
            path = os.path.join(parent, candidate)
            if os.path.isfile(path):
                return path

    # Search workspaces — match by layer count
    target_layers = _count_endtoend_layers(result_filepath) if result_filepath else -1

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_dirs = []

    workspaces_root = os.path.join(server_root, "workspaces")
    if os.path.isdir(workspaces_root):
        for workspace in sorted(os.listdir(workspaces_root)):
            wpath = os.path.join(workspaces_root, workspace)
            if os.path.isdir(wpath):
                search_dirs.append(wpath)

    # Also search example/ directory for standalone workload files
    example_dir = os.path.join(project_root, "example")
    if os.path.isdir(example_dir):
        search_dirs.append(example_dir)

    if target_layers > 0:
        for search_dir in search_dirs:
            for fname in sorted(os.listdir(search_dir)):
                if not fname.endswith(".txt"):
                    continue
                wl_path = os.path.join(search_dir, fname)
                if os.path.isfile(wl_path) and _count_workload_layers(wl_path) == target_layers:
                    return wl_path
    return ""


def _parse_value(val_str: str) -> Optional[float]:
    """Parse a numeric value string, returning None for 'NONE', 'nan', etc."""
    val_str = val_str.strip()
    if not val_str or val_str.upper() in ("NONE", "NAN"):
        return None
    try:
        return float(val_str)
    except ValueError:
        return None


def parse_endtoend_csv(filepath: str, workload_path: str = "") -> Dict:
    """
    Parse the multi-section EndToEnd.csv produced by SimAI.

    Returns dict with keys: layers, summary, totals, dimensions, run_name.

    If workload_path is not provided, attempts to find the workload file that
    generated this result by searching alongside the result file and in workspace
    directories. The workload file is used to determine per-layer comm groups
    (TP/DP/EP/DP_EP) from the actual comm_type values.
    """
    result: Dict = {
        "layers": [],
        "summary": {},
        "totals": {},
        "dimensions": {},
        "run_name": "",
    }

    # Find workload file if not provided
    if not workload_path:
        workload_path = _find_workload_for_result(filepath)

    # Load comm group mapping from workload file (index-based, not name-based)
    layer_comm_groups: list = []
    if workload_path and os.path.isfile(workload_path):
        layer_comm_groups = _build_layer_comm_groups(workload_path)

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 4:
        return result

    dim_offset = 0
    if lines[0].startswith("File name"):
        dim_offset = 2
        dim_keys = [k.strip() for k in lines[0].split(",")]
        dim_vals = [v.strip() for v in lines[1].split(",")]

        for key, val in zip(dim_keys[1:], dim_vals[1:]):
            # Extract totals from dimension row — values may be plain numbers or "X (YY%)"
            key_lower = key.lower().strip()
            if key_lower in ("total comp", "total exposed comm", "total time", "bubble time"):
                # Try plain number first
                parsed = _parse_value(val)
                if parsed is None:
                    # Try percentage format: "1112 (87.98%)" → extract 1112
                    m = re.match(r"([\d.eE+\-]+)", val)
                    if m:
                        parsed = _parse_value(m.group(1))
                if parsed is not None:
                    if key_lower == "total comp":
                        result["totals"]["total_compute"] = parsed
                    elif key_lower == "total exposed comm":
                        result["totals"]["total_exposed"] = parsed
                    elif key_lower == "total time":
                        result["totals"]["total_time"] = parsed
                    elif key_lower == "bubble time":
                        result["totals"]["bubble_time"] = parsed
                continue

            match = re.match(r"([\d.eE+\-]+)\s*\(([\d.]+)%\)", val)
            if match:
                result["dimensions"][key] = {
                    "value": float(match.group(1)),
                    "percentage": float(match.group(2)),
                }
            else:
                parsed = _parse_value(val)
                if parsed is not None:
                    result["dimensions"][key] = {"value": parsed, "percentage": 0.0}

        if dim_vals:
            result["run_name"] = dim_vals[0]

    header_idx = dim_offset
    if header_idx >= len(lines):
        return result

    # Parse per-layer rows
    layer_idx = 0
    for i in range(header_idx + 1, len(lines)):
        parts = [p.strip() for p in lines[i].split(",")]
        if not parts:
            continue

        # Detect totals row
        if parts[0] == "total exposed comm":
            for j in range(0, len(parts) - 1, 2):
                key = parts[j].strip()
                val = _parse_value(parts[j + 1]) if j + 1 < len(parts) else None
                if key == "total exposed comm":
                    result["totals"]["total_exposed"] = val
                elif key == "total comp":
                    result["totals"]["total_compute"] = val
                elif key == "total time":
                    result["totals"]["total_time"] = val
                elif key == "bubble time":
                    result["totals"]["bubble_time"] = val
            continue

        if len(parts) < 8:
            continue

        layer_name = parts[0]
        run_name = parts[1] if len(parts) > 1 else ""

        if not result["run_name"] and run_name:
            result["run_name"] = run_name

        # Get comm groups from workload file by index (not name, since names repeat)
        groups = layer_comm_groups[layer_idx] if layer_idx < len(layer_comm_groups) else {}

        row = {
            "layer_name": layer_name,
            "run_name": run_name,
            "fwd_compute": _parse_value(parts[2]) if len(parts) > 2 else None,
            "wg_compute": _parse_value(parts[3]) if len(parts) > 3 else None,
            "ig_compute": _parse_value(parts[4]) if len(parts) > 4 else None,
            "fwd_exposed_comm": _parse_value(parts[5]) if len(parts) > 5 else None,
            "wg_exposed_comm": _parse_value(parts[6]) if len(parts) > 6 else None,
            "ig_exposed_comm": _parse_value(parts[7]) if len(parts) > 7 else None,
            "fwd_total_comm": _parse_value(parts[8]) if len(parts) > 8 else None,
            "fwd_algbw": _parse_value(parts[9]) if len(parts) > 9 else None,
            "fwd_busbw": _parse_value(parts[10]) if len(parts) > 10 else None,
            "wg_total_comm": _parse_value(parts[11]) if len(parts) > 11 else None,
            "wg_algbw": _parse_value(parts[12]) if len(parts) > 12 else None,
            "wg_busbw": _parse_value(parts[13]) if len(parts) > 13 else None,
            "ig_total_comm": _parse_value(parts[14]) if len(parts) > 14 else None,
            "ig_algbw": _parse_value(parts[15]) if len(parts) > 15 else None,
            "ig_busbw": _parse_value(parts[16]) if len(parts) > 16 else None,
            "workload_finished_at": _parse_value(parts[17]) if len(parts) > 17 else None,
            # Comm groups from workload file
            "fwd_group": groups.get("fp_group", "TP"),
            "wg_group": groups.get("wg_group", "DP"),
            "ig_group": groups.get("ig_group", "TP"),
        }

        if layer_name == "SUM":
            result["summary"] = row
        else:
            result["layers"].append(row)
            layer_idx += 1

    return result


def parse_console_output(log_lines: List[str]) -> Dict:
    """
    Extract key metrics from SimAI simulator console stdout.

    Returns dict with: finish_time, streams_injected, streams_finished, nodes.
    """
    result: Dict = {
        "finish_time": None,
        "streams_injected": None,
        "streams_finished": None,
        "nodes": [],
    }

    node_sent: Dict[int, float] = {}
    node_recv: Dict[int, float] = {}

    for line in log_lines:
        m = re.search(r"all passes finished at time:\s*([\d.eE+\-]+)", line, re.IGNORECASE)
        if m:
            result["finish_time"] = float(m.group(1))

        m = re.search(r"pass:\s*\d+\s+finished at time:\s*([\d.eE+\-]+)", line, re.IGNORECASE)
        if m:
            result["finish_time"] = float(m.group(1))

        m = re.search(r"Total streams injected:\s*(\d+)", line)
        if m:
            result["streams_injected"] = int(m.group(1))

        m = re.search(r"Total streams finished:\s*(\d+)", line)
        if m:
            result["streams_finished"] = int(m.group(1))

        m = re.search(r"All data sent from node\s+(\d+)\s+is\s+([\d.eE+\-]+)", line)
        if m:
            node_sent[int(m.group(1))] = float(m.group(2))

        m = re.search(r"All data received by node\s+(\d+)\s+is\s+([\d.eE+\-]+)", line)
        if m:
            node_recv[int(m.group(1))] = float(m.group(2))

    for nid in sorted(set(node_sent) | set(node_recv)):
        result["nodes"].append({
            "id": nid,
            "sent": node_sent.get(nid, 0),
            "received": node_recv.get(nid, 0),
        })

    return result


def find_ns3_output_files(result_path: str) -> Dict[str, str]:
    """Locate EndToEnd.csv and detailed CSV files in result directories."""
    files: Dict[str, str] = {}

    search_dirs = []
    if result_path:
        search_dirs.append(result_path)
        parent = os.path.dirname(result_path)
        if parent and parent != result_path:
            search_dirs.append(parent)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    search_dirs.extend([
        os.path.join(project_root, "results"),
        project_root,
        ".",
        "./results",
        "results",
    ])

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        try:
            for filename in os.listdir(search_dir):
                filepath = os.path.join(search_dir, filename)
                if not os.path.isfile(filepath):
                    continue
                if "EndToEnd" in filename and filename.endswith(".csv"):
                    if "endtoend" not in files:
                        files["endtoend"] = filepath
                elif filename.startswith("detailed_") and filename.endswith(".csv"):
                    if "detailed" not in files:
                        files["detailed"] = filepath
        except OSError:
            continue

    return files
