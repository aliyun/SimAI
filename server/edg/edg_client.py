"""HTTP client for the EDG (OXC network coordinator) service."""

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

EDG_URL = os.environ.get("AS_EDG_URL", "http://127.0.0.1:9000/api/port_allocation")
EDG_TIMEOUT = int(os.environ.get("AS_EDG_TIMEOUT", "30"))
EDG_MOCK = os.environ.get("AS_EDG_MOCK", "0") in ("1", "true", "True")

# Topology context used by the smart mock when the real EDG is unreachable.
# Routes sets this before calling notify_node_matrix so the mock can produce
# a visible, topology-aware adjustment instead of a hardcoded no-op.
_mock_ctx: Optional[Dict[str, Any]] = None


def _record_edg_call(msg_type: str, req: dict, resp: dict, duration: float,
                    fallback: bool, warning: str):
    """Record an EDG interaction for the ops tool."""
    try:
        from server.ops.ops_routes import record_interaction
        ws = os.environ.get("SIMAI_ACTIVE_WORKSPACE", "")
        if not ws:
            return
        algo = req.get("alg_name") or req.get("algorithm", "")
        entry = {
            "source": "EDG",
            "type": msg_type,
            "endpoint": EDG_URL,
            "method": "POST",
            "algorithm": algo,
            "request": {"message_type": msg_type, "request_id": req.get("request_id", "")},
            "response": {
                "result": resp.get("result", "OK"),
                "duration_ms": round(duration * 1000, 1),
                "fallback": fallback,
                "del_count": len(resp.get("oxc_oper_orders", {}).get("del_oper_info_list", [])),
                "add_count": len(resp.get("oxc_oper_orders", {}).get("add_oper_info_list", [])),
            },
        }
        if warning:
            entry["response"]["warning"] = warning
        record_interaction(ws, entry)
    except Exception:
        pass


def set_mock_context(lld: Optional[Dict[str, Any]], baseline_crosses: Optional[list]) -> None:
    """Set lld+baseline context for the smart mock. Call with None to clear."""
    global _mock_ctx
    if lld is None or baseline_crosses is None:
        _mock_ctx = None
    else:
        _mock_ctx = {"lld": lld, "baseline_crosses": baseline_crosses}


class EdgError(Exception):
    pass


def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    if EDG_MOCK:
        result = _mock_response(payload)
        result["_fallback"] = True
        result["_warning"] = "EDG mock mode enabled"
        return result

    ts = None
    try:
        resp = requests.post(EDG_URL, json=payload, timeout=EDG_TIMEOUT)
        resp.raise_for_status()
        ts = resp.elapsed.total_seconds()
    except requests.RequestException as e:
        logger.warning("EDG unreachable (%s), falling back to mock", e)
        result = _mock_response(payload)
        result["_fallback"] = True
        result["_warning"] = f"EDG 服务不可达，已降级为 mock 模式 ({type(e).__name__})"
        _record_edg_call(payload.get("message_type", "unknown"), payload, result,
                         0, True, result["_warning"])
        return result

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as e:
        raise EdgError(f"EDG returned invalid JSON: {e}") from e

    if data.get("result") == "OUTPUT_ERROR":
        raise EdgError(f"EDG returned error for request_id={data.get('request_id')}")

    # Record interaction for ops tool
    _record_edg_call(payload.get("message_type", "unknown"), payload, data,
                     ts or 0, False, "")
    return data


def import_full_topo(lld: Dict[str, Any]) -> Dict[str, Any]:
    """POST IMPORT_FULL_TOPO to EDG, return the response dict."""
    logger.info("EDG import_full_topo: request_id=%s", lld.get("request_id"))
    return _post(lld)


def notify_node_matrix(npu_match: Dict[str, Any]) -> Dict[str, Any]:
    """POST NOTIFY_NODE_MATRIX to EDG, return the response dict."""
    logger.info("EDG notify_node_matrix: request_id=%s", npu_match.get("request_id"))
    return _post(npu_match)


# --- Mock fallback (deterministic, no HTTP) ---

def _mock_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    msg_type = payload.get("message_type", "")
    req_id = payload.get("request_id", 0)

    if msg_type == "IMPORT_FULL_TOPO":
        add_list = _mock_baseline_crosses(payload)
        return {
            "request_id": req_id,
            "message_type": msg_type,
            "result": "OUTPUT_OXCCROSS",
            "oxc_oper_orders": {
                "del_oper_info_list": [],
                "add_oper_info_list": add_list,
            },
        }

    if msg_type == "NOTIFY_NODE_MATRIX":
        return {
            "request_id": req_id,
            "message_type": msg_type,
            "result": "OUTPUT_OXCCROSS",
            "oxc_oper_orders": [_mock_task_adjustment(_mock_ctx)],
        }

    raise EdgError(f"Mock: unknown message_type '{msg_type}'")


def _mock_baseline_crosses(lld: Dict[str, Any]) -> list:
    """Generate baseline OXC crosses from lld topology.

    Pairs OXC ports connecting different leaves into cross-connects.
    """
    topo = lld.get("topology", {})
    oxc_nodes = topo.get("oxc_nodes", [])
    edges = topo.get("edges", [])
    if not oxc_nodes:
        return []

    oxc_ips = {n["node_ip"] for n in oxc_nodes}

    # Group OXC ports by which leaf they connect to
    # oxc_port_to_leaf: {(oxc_ip, port) -> leaf_ip}
    oxc_port_leaf: Dict[str, Dict[str, str]] = {}  # oxc_ip -> {port -> leaf_ip}
    for e in edges:
        a_ip, b_ip = e["a_node_ip"], e["b_node_ip"]
        if a_ip in oxc_ips:
            oxc_port_leaf.setdefault(a_ip, {})[str(e["a_node_port_id"])] = b_ip
        elif b_ip in oxc_ips:
            oxc_port_leaf.setdefault(b_ip, {})[str(e["b_node_port_id"])] = a_ip

    crosses = []
    for oxc_ip, port_map in oxc_port_leaf.items():
        # Group ports by leaf
        leaf_ports: Dict[str, list] = {}
        for port, leaf_ip in port_map.items():
            leaf_ports.setdefault(leaf_ip, []).append(port)

        # Create cross-connects between ports of different leaves
        leaf_list = sorted(leaf_ports.keys())
        for i in range(len(leaf_list)):
            for j in range(i + 1, len(leaf_list)):
                ports_a = sorted(leaf_ports[leaf_list[i]])
                ports_b = sorted(leaf_ports[leaf_list[j]])
                # Connect all available ports per leaf pair
                for k in range(min(len(ports_a), len(ports_b))):
                    crosses.append({
                        "node_ip": oxc_ip,
                        "a_port_id": ports_a[k],
                        "b_port_id": ports_b[k],
                    })

    return crosses


# Cached baseline crosses for mock task adjustment
_mock_baseline_cache: list = []


def _mock_task_adjustment(ctx: Optional[Dict[str, Any]] = None) -> dict:
    """Generate mock OXC adjustments.

    When `ctx` carries the current lld + baseline crosses, produce a
    topology-aware adjustment guaranteed to flip at least one leaf-leaf
    edge in the diff. Falls back to a hardcoded no-op only when context
    is unavailable.
    """
    if ctx:
        smart = _smart_adjustment(ctx)
        if smart is not None:
            return smart

    # Legacy hardcoded fallback (kept for request_id symmetry with the real API
    # when lld context is missing). Likely produces empty diff.
    return {
        "del_oper_info_list": [
            {"node_ip": "10.118.241.50", "a_port_id": "1", "b_port_id": "5",
             "a_switch_info": {"switch_ip": "10.118.241.25", "port_info": "400GE/0/17/1"},
             "b_switch_info": {"switch_ip": "10.118.241.26", "port_info": "400GE/0/17/1"}},
            {"node_ip": "10.118.241.50", "a_port_id": "10", "b_port_id": "13",
             "a_switch_info": {"switch_ip": "10.118.241.27", "port_info": "400GE/0/18/1"},
             "b_switch_info": {"switch_ip": "10.118.241.28", "port_info": "400GE/0/17/1"}},
        ],
        "add_oper_info_list": [
            {"node_ip": "10.118.241.50", "a_port_id": "3", "b_port_id": "15"},
            {"node_ip": "10.118.241.50", "a_port_id": "7", "b_port_id": "9"},
        ],
    }


def _smart_adjustment(ctx: Dict[str, Any]) -> Optional[dict]:
    """Compute a topology-aware adjustment that yields a visible diff.

    Strategy:
      1. Pick one leaf pair fully removable (delete ALL its crosses) → diff.removed+1
      2. Try to connect a currently unconnected leaf pair via unused ports → diff.added+1
      3. If (1) is the only viable move on a fully-connected mesh, ship just the deletes.
    Returns None when no viable adjustment can be computed.
    """
    lld = ctx.get("lld") or {}
    baseline = ctx.get("baseline_crosses") or []
    topo = lld.get("topology", {})
    oxc_ips = {n["node_ip"] for n in topo.get("oxc_nodes", [])}
    if not oxc_ips or not baseline:
        return None

    # Build (oxc_ip, port) -> leaf_ip
    port_to_leaf: Dict[tuple, str] = {}
    leaf_to_ports: Dict[str, Dict[str, list]] = {}  # oxc_ip -> leaf_ip -> [port]
    for e in topo.get("edges", []):
        a_ip, b_ip = e["a_node_ip"], e["b_node_ip"]
        a_port, b_port = str(e["a_node_port_id"]), str(e["b_node_port_id"])
        if a_ip in oxc_ips:
            port_to_leaf[(a_ip, a_port)] = b_ip
            leaf_to_ports.setdefault(a_ip, {}).setdefault(b_ip, []).append(a_port)
        elif b_ip in oxc_ips:
            port_to_leaf[(b_ip, b_port)] = a_ip
            leaf_to_ports.setdefault(b_ip, {}).setdefault(a_ip, []).append(b_port)

    def _norm(c):
        a, b = sorted([str(c[1]), str(c[2])])
        return (c[0], a, b)

    baseline_set = {_norm(c) for c in baseline}

    # Group baseline crosses by (oxc_ip, leaf_pair)
    pair_to_crosses: Dict[tuple, list] = {}
    used_ports: Dict[str, set] = {}  # oxc_ip -> set(port)
    for (oxc, pa, pb) in baseline_set:
        la = port_to_leaf.get((oxc, pa))
        lb = port_to_leaf.get((oxc, pb))
        used_ports.setdefault(oxc, set()).update([pa, pb])
        if la and lb and la != lb:
            key = (oxc, tuple(sorted([la, lb])))
            pair_to_crosses.setdefault(key, []).append((pa, pb))

    if not pair_to_crosses:
        return None

    # Pick the first pair to fully remove
    (oxc_del, leaf_pair_del), crosses_to_del = next(iter(pair_to_crosses.items()))
    del_list = [
        {"node_ip": oxc_del, "a_port_id": pa, "b_port_id": pb}
        for (pa, pb) in crosses_to_del
    ]

    # Try to add a cross on a leaf pair NOT currently in baseline
    add_list: list = []
    leaves_in_oxc = list(leaf_to_ports.get(oxc_del, {}).keys())
    existing_pairs = {lp for (_oxc, lp) in pair_to_crosses.keys() if _oxc == oxc_del}
    for i in range(len(leaves_in_oxc)):
        for j in range(i + 1, len(leaves_in_oxc)):
            la, lb = sorted([leaves_in_oxc[i], leaves_in_oxc[j]])
            if (la, lb) in existing_pairs:
                continue
            # Find unused ports for each leaf
            free_a = [p for p in leaf_to_ports[oxc_del][la]
                      if p not in used_ports.get(oxc_del, set())]
            free_b = [p for p in leaf_to_ports[oxc_del][lb]
                      if p not in used_ports.get(oxc_del, set())]
            if free_a and free_b:
                add_list.append({
                    "node_ip": oxc_del,
                    "a_port_id": free_a[0],
                    "b_port_id": free_b[0],
                })
                break
        if add_list:
            break

    # Only return deletions if we successfully provisioned a replacement.
    # Blindly deleting crosses without a replacement destroys leaf connectivity.
    if not add_list:
        return None
    return {
        "del_oper_info_list": del_list,
        "add_oper_info_list": add_list,
    }
