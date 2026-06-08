"""Flask routes for EDG (OXC network coordinator) integration."""

import json
import logging
import os
import subprocess
import sys

from flask import Blueprint, request, jsonify

from server.auth.auth_service import require_auth
from server.edg import edg_client
from server.edg.crosses import apply_batches, parse_add_list
from server.edg.merger import resolve_paths, split_graph_by_pod
from server.edg.ns3_emitter import write_ns3_topology
from server.simulation.ranktable_generator import generate_ranktable

logger = logging.getLogger(__name__)

edg_bp = Blueprint("edg", __name__, url_prefix="/api/edg")

# PLACEHOLDER_ROUTES_APPEND


def _edg_dir(workspace_dir: str) -> str:
    p = os.path.join(workspace_dir, "edg")
    os.makedirs(p, exist_ok=True)
    return p


def _edg_global_dir(topology_dir: str) -> str:
    """Global EDG data dir keyed by topology_dir (survives session restarts)."""
    from server.config import EDG_DATA_ROOT
    p = os.path.join(EDG_DATA_ROOT, topology_dir, "edg")
    os.makedirs(p, exist_ok=True)
    return p


def _edg_load(topology_dir: str) -> tuple | None:
    """Load baseline EDG data (lld, crosses) from global or workspace store."""
    lld = None
    saved = None

    # Prefer global store when topology_dir is provided
    if topology_dir:
        global_dir = _edg_global_dir(topology_dir)
        lld_path = os.path.join(global_dir, "lld.json")
        crosses_path = os.path.join(global_dir, "init_crosses.json")
        if os.path.exists(lld_path) and os.path.exists(crosses_path):
            with open(lld_path, "r") as f:
                lld = json.load(f)
            with open(crosses_path, "r") as f:
                saved = json.load(f)

    # Fallback: try workspace store
    if lld is None:
        from server.config import WORKSPACE_ROOT
        ws = WORKSPACE_ROOT
        for user_dir in os.listdir(ws) if os.path.exists(ws) else []:
            edg_dir = os.path.join(ws, user_dir, "edg")
            lld_path = os.path.join(edg_dir, "lld.json")
            crosses_path = os.path.join(edg_dir, "init_crosses.json")
            if os.path.exists(lld_path) and os.path.exists(crosses_path):
                with open(lld_path, "r") as f:
                    lld = json.load(f)
                with open(crosses_path, "r") as f:
                    saved = json.load(f)
                break

    if lld is None or saved is None:
        return None
    base_crosses = {tuple(c) for c in saved.get("crosses", [])}
    return lld, base_crosses


def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@edg_bp.route("/init", methods=["POST"])
@require_auth
def api_edg_init():
    """Initialize network: forward lld.json to EDG, store baseline crosses."""
    data = request.get_json(silent=True) or {}
    lld = data.get("lld")
    topology_dir = data.get("topology_dir", "")
    if not lld or not isinstance(lld, dict):
        return jsonify({"error": "lld (object) is required"}), 400

    ws = getattr(request, "workspace_dir", ".")
    edg_dir = _edg_dir(ws)

    # Also write to global store (keyed by topology_dir) for session persistence
    global_edg_dir = _edg_global_dir(topology_dir) if topology_dir else None

    # Save lld.json (both workspace and global)
    lld_path = os.path.join(edg_dir, "lld.json")
    _save_json(lld_path, lld)
    if global_edg_dir:
        _save_json(os.path.join(global_edg_dir, "lld.json"), lld)

    # Call EDG
    try:
        resp = edg_client.import_full_topo(lld)
    except edg_client.EdgError as e:
        logger.exception("EDG init failed")
        return jsonify({"error": str(e)}), 502

    warning = resp.get("_warning")

    # Parse and save baseline crosses (both workspace and global)
    orders = resp.get("oxc_oper_orders", {})
    base_crosses = apply_batches(set(), orders)
    crosses_data = {
        "crosses": [list(c) for c in base_crosses],
        "raw_response": resp,
    }
    crosses_path = os.path.join(edg_dir, "init_crosses.json")
    _save_json(crosses_path, crosses_data)
    if global_edg_dir:
        _save_json(os.path.join(global_edg_dir, "init_crosses.json"), crosses_data)

    # Generate monitor dashboard XML via lld_to_topology.py
    _regenerate_dashboard_xml(lld, ws, topology_dir)

    return jsonify({
        "lld_path": lld_path,
        "crosses_count": len(base_crosses),
        "oxc_count": len({c[0] for c in base_crosses}),
        "warning": warning,
        "persisted": bool(global_edg_dir),
    })


@edg_bp.route("/baseline-graph", methods=["POST"])
@require_auth
def api_edg_baseline_graph():
    """Return the baseline connectivity graph (before any task adjustment)."""
    data = request.get_json(silent=True) or {}
    server_ips = data.get("server_ips", [])
    npu_per_server = data.get("npu_per_server", 8)
    topology_dir = data.get("topology_dir", "")

    _load = _edg_load(topology_dir)
    if _load is None:
        return jsonify({"error": "No baseline data. Call /api/edg/init first."}), 400
    lld, base_crosses = _load

    participating = server_ips if server_ips else None
    graph = resolve_paths(lld, base_crosses, participating_server_ips=participating)

    return jsonify({
        "graph": _serialize_graph(graph),
        "crosses_count": len(base_crosses),
    })


def _serialize_graph(graph):
    """Convert graph dict to JSON-safe format (tuples → lists)."""
    return {
        "servers": graph["servers"],
        "leaves": graph["leaves"],
        "server_leaf_edges": [list(e) for e in graph["server_leaf_edges"]],
        "leaf_leaf_edges": [list(e) for e in graph["leaf_leaf_edges"]],
        "oxc_port_map": graph.get("oxc_port_map", {}),
    }


def _compute_diff(base_graph, new_graph):
    """Compute leaf-leaf edge diff between baseline and adjusted graphs."""
    def _edge_key(e):
        return tuple(sorted([e[0], e[1]]))

    base_map = {_edge_key(e): e for e in base_graph["leaf_leaf_edges"]}
    new_map = {_edge_key(e): e for e in new_graph["leaf_leaf_edges"]}

    base_keys = set(base_map.keys())
    new_keys = set(new_map.keys())

    return {
        "added": [list(new_map[k]) for k in new_keys - base_keys],
        "removed": [list(base_map[k]) for k in base_keys - new_keys],
        "unchanged": [list(new_map[k]) for k in new_keys & base_keys],
    }


@edg_bp.route("/register-task", methods=["POST"])
@require_auth
def api_edg_register_task():
    """Register AI training task: forward npu_match to EDG, merge crosses, emit NS3 topo."""
    data = request.get_json(silent=True) or {}
    npu_match = data.get("npu_match")
    task_id = data.get("task_id", "T001")
    topo_params = data.get("topo_params", {})
    topology_dir = data.get("topology_dir", "")

    # Auto-build npu_match from server_ips if provided
    if npu_match and "server_ips" in npu_match and "npu_matrix" not in npu_match:
        server_ips = npu_match["server_ips"]
        npu_per_server = npu_match.get("npu_per_server", 8)
        npu_match = _build_npu_match(server_ips, task_id, npu_per_server)

    if not npu_match or not isinstance(npu_match, dict):
        return jsonify({"error": "npu_match (object) is required"}), 400

    # Load baseline from global store (fallback to workspace)
    _load = _edg_load(topology_dir)
    if _load is None:
        return jsonify({"error": "No baseline data. Call /api/edg/init first."}), 400
    lld, base_crosses = _load

    ws = getattr(request, "workspace_dir", ".")
    edg_dir = _edg_dir(ws)

    # Save npu_match
    task_dir = os.path.join(edg_dir, "tasks", task_id)
    os.makedirs(task_dir, exist_ok=True)
    _save_json(os.path.join(task_dir, "npu_match.json"), npu_match)

    # Call EDG (inject mock context so fallback produces visible diff)
    edg_client.set_mock_context(lld, [list(c) for c in base_crosses])
    try:
        resp = edg_client.notify_node_matrix(npu_match)
    except edg_client.EdgError as e:
        logger.exception("EDG register-task failed")
        return jsonify({"error": str(e)}), 502
    finally:
        edg_client.set_mock_context(None, None)

    task_warning = resp.get("_warning")

    # Merge crosses
    orders = resp.get("oxc_oper_orders", [])
    merged = apply_batches(base_crosses, orders)
    _save_json(os.path.join(task_dir, "merged_crosses.json"), {
        "crosses": [list(c) for c in merged],
        "raw_response": resp,
    })

    # Extract participating server IPs from npu_match
    server_ips = _extract_server_ips(npu_match)

    # Resolve paths (sub-topology)
    graph = resolve_paths(lld, merged, participating_server_ips=server_ips)

    # Emit NS3 topology — split into per-ODC pods for multi-ODC scenarios
    pods = split_graph_by_pod(graph, lld)
    emit_kwargs = {"lld": lld}
    if topo_params:
        if topo_params.get("npu_per_server"):
            emit_kwargs["npu_per_server"] = int(topo_params["npu_per_server"])
        if topo_params.get("npu_type"):
            emit_kwargs["npu_type"] = topo_params["npu_type"]
        if topo_params.get("intra_bw"):
            emit_kwargs["intra_bw"] = topo_params["intra_bw"]
        if topo_params.get("bandwidth"):
            emit_kwargs["bandwidth"] = topo_params["bandwidth"]
            emit_kwargs["ap_bandwidth"] = topo_params["bandwidth"]

    topology_files = []
    if len(pods) == 1:
        # Single pod — backward-compatible naming
        topo_key, pod_graph = next(iter(pods.items()))
        out_name = f"edg_topo_{task_id}"
        out_path = os.path.join(ws, out_name)
        write_ns3_topology(pod_graph, out_path, **emit_kwargs)
        topology_files.append(out_name)
        logger.info("Single-pod topology: %s (oxc=%s)", out_name, topo_key)
    else:
        for idx, (oxc_ip, pod_graph) in enumerate(sorted(pods.items())):
            out_name = f"edg_topo_{task_id}_pod{idx}"
            out_path = os.path.join(ws, out_name)
            write_ns3_topology(pod_graph, out_path, **emit_kwargs)
            topology_files.append(out_name)
            logger.info("Multi-pod topology [%d/%d]: %s (oxc=%s, servers=%d)",
                        idx + 1, len(pods), out_name, oxc_ip,
                        len(pod_graph.get("servers", [])))

    # Compute diff against baseline on FULL network (not sub-topology)
    full_base_graph = resolve_paths(lld, base_crosses, participating_server_ips=None)
    full_adjusted_graph = resolve_paths(lld, merged, participating_server_ips=None)
    diff = _compute_diff(full_base_graph, full_adjusted_graph)

    # Auto-generate ranktable
    npu_per_srv = int(topo_params.get("npu_per_server", 8)) if topo_params else 8
    rank_count = len(graph["servers"]) * npu_per_srv
    rt_data, rt_map = generate_ranktable(rank_count, npu_per_srv)
    rt_path = os.path.join(ws, "ranktable.json")
    _save_json(rt_path, rt_data)

    # Regenerate dashboard XML with crosses + npu_match highlights
    _regenerate_dashboard_xml_with_crosses(lld, merged, npu_match, task_id, ws)

    return jsonify({
        "task_id": task_id,
        "topology_file": topology_files[0],  # backward compat
        "topology_files": topology_files,     # full list for multi-pod
        "graph_stats": {
            "servers": len(graph["servers"]),
            "leaves": len(graph["leaves"]),
            "leaf_leaf_edges": len(graph["leaf_leaf_edges"]),
            "server_leaf_edges": len(graph["server_leaf_edges"]),
            "pod_count": len(pods),
        },
        "graph": _serialize_graph(graph),
        "diff": diff,
        "ranktable_path": "ranktable.json",
        "warning": task_warning,
    })


def _extract_server_ips(npu_match: dict) -> list:
    seen = set()
    result = []
    for entry in npu_match.get("npu_matrix", []):
        for npu in entry.get("npu_set", []):
            ip = npu.get("server_ip", "")
            if ip and ip not in seen:
                seen.add(ip)
                result.append(ip)
    return result


def _build_npu_match(server_ips: list, task_id: str, npu_per_server: int = 8) -> dict:
    """Auto-build npu_match.json from a list of server IPs."""
    npu_matrix = []
    for ip in server_ips:
        npu_matrix.append({
            "inst_key": {
                "task_id": task_id,
                "k8s_task_id": task_id,
                "inst_id": "NULL",
                "inst_type": "TRAINING",
            },
            "npu_set": [{"server_ip": ip, "npu_id": -1}],
        })
    return {
        "version": "v2.0.260430",
        "request_id": abs(hash(task_id)) % 100000,
        "message_type": "NOTIFY_NODE_MATRIX",
        "npu_matrix": npu_matrix,
    }


def _regenerate_dashboard_xml(lld: dict, workspace_dir: str, topology_dir: str = ""):
    """Run lld_to_topology.py to generate baseline monitor XML."""
    try:
        from server.config import PROJECT_ROOT
    except ImportError:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    script = os.path.join(PROJECT_ROOT, "scripts", "lld_to_topology.py")
    lld_path = os.path.join(_edg_dir(workspace_dir), "lld.json")

    if topology_dir:
        topo_dir = os.path.join(PROJECT_ROOT, topology_dir) if not os.path.isabs(topology_dir) else topology_dir
    else:
        topo_dir = os.path.join(PROJECT_ROOT, "topology")
    os.makedirs(topo_dir, exist_ok=True)

    try:
        subprocess.run(
            [sys.executable, script, lld_path, "--output-dir", topo_dir],
            check=True, capture_output=True, text=True, timeout=30,
        )
        logger.info("Dashboard XML regenerated at %s", topo_dir)
    except Exception as e:
        logger.warning("Failed to regenerate dashboard XML: %s", e)


def _regenerate_dashboard_xml_with_crosses(
    lld: dict, crosses: set, npu_match: dict, task_id: str, workspace_dir: str,
):
    """Regenerate pod XML with OXC cross highlights and task server markers."""
    try:
        from server.config import PROJECT_ROOT
    except ImportError:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    try:
        from lld_to_topology import generate_pod_xml_with_crosses
    except ImportError:
        logger.warning("generate_pod_xml_with_crosses not available yet, skipping dashboard update")
        return

    server_ips = set(_extract_server_ips(npu_match))
    crosses_list = [{"node_ip": c[0], "a_port_id": c[1], "b_port_id": c[2]} for c in crosses]

    try:
        xml_str = generate_pod_xml_with_crosses(lld, crosses_list, server_ips, task_id)
        pods_dir = os.path.join(PROJECT_ROOT, "topology", "pods")
        os.makedirs(pods_dir, exist_ok=True)
        pod_file = os.path.join(pods_dir, "POD#1.xml")
        with open(pod_file, "w", encoding="utf-8") as f:
            f.write(xml_str)
        logger.info("Dashboard pod XML updated with task %s highlights", task_id)
    except Exception as e:
        logger.warning("Failed to update dashboard XML with crosses: %s", e)
