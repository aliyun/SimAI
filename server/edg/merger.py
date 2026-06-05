"""Resolve OXC crosses + lld.json edges into a connectivity graph.

The graph contains only the servers listed in npu_match (sub-topology).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from server.edg.crosses import Cross

logger = logging.getLogger(__name__)


def _chassis_to_npu_type(chassis_topo: str) -> str:
    """Extract NPU type from chassis_topo prefix, e.g. 'A5_1DPOD' -> 'A5'."""
    return chassis_topo.split("_")[0] if "_" in chassis_topo else chassis_topo


def _build_edge_maps(edges: List[Dict[str, Any]], oxc_ips: Set[str], server_ips: Set[str],
                    spine_ips: Optional[Set[str]] = None):
    """Index lld.topology.edges into lookup tables.

    Supports both flat (OXC→leaf) and spine-based (OXC→spine→leaf) topologies.

    Returns:
        oxc_port_to_leaves: {(oxc_ip, port) -> [(leaf_ip, leaf_port), ...]}  (N:N via spine)
        leaf_port_to_server: {(leaf_ip, port) -> (server_ip, server_port)}
        server_to_leaves: {server_ip -> set of leaf_ips}
    """
    if spine_ips is None:
        spine_ips = set()

    # Intermediate mappings for spine-based topologies
    oxc_spine_edge: Dict[Tuple[str, str], Tuple[str, str]] = {}  # (oxc_ip, port) -> (spine_ip, spine_port)
    spine_port_to_leaf: Dict[Tuple[str, str], Tuple[str, str]] = {}  # (spine_ip, spine_port) -> (leaf_ip, leaf_port)

    oxc_port_to_leaves: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    leaf_port_to_server: Dict[Tuple[str, str], Tuple[str, str]] = {}
    server_to_leaves: Dict[str, Set[str]] = {}

    for e in edges:
        a_id, b_id = e["a_node_id"], e["b_node_id"]
        a_clean = re.sub(r"\(\d+\)$", "", a_id).strip()
        b_clean = re.sub(r"\(\d+\)$", "", b_id).strip()
        a_port, b_port = str(e["a_node_port_id"]), str(e["b_node_port_id"])

        # OXC ↔ spine edges — record exactly which spine port
        if spine_ips:
            if a_clean in oxc_ips and b_id in spine_ips:
                oxc_spine_edge[(a_clean, a_port)] = (b_id, b_port)
                continue
            elif b_clean in oxc_ips and a_id in spine_ips:
                oxc_spine_edge[(b_clean, b_port)] = (a_id, a_port)
                continue

        # Spine ↔ leaf edges — keyed by (spine_ip, spine_port), use the
        # other side as leaf target regardless of which field it's in
        if spine_ips:
            if a_id in spine_ips and b_clean not in oxc_ips:
                spine_port_to_leaf[(a_id, a_port)] = (b_clean, b_port)
                continue
            elif b_id in spine_ips and a_clean not in oxc_ips:
                spine_port_to_leaf[(b_id, b_port)] = (a_clean, a_port)
                continue

        # Direct OXC ↔ leaf edges (flat topology, no spine)
        if not spine_ips:
            if a_clean in oxc_ips:
                oxc_port_to_leaves.setdefault((a_clean, a_port), []).append((b_clean, b_port))
            elif b_clean in oxc_ips:
                oxc_port_to_leaves.setdefault((b_clean, b_port), []).append((a_clean, a_port))

        # Leaf ↔ server edges (1 server can connect to N leaves)
        if a_clean in server_ips:
            leaf_port_to_server[(b_clean, b_port)] = (a_clean, a_port)
            server_to_leaves.setdefault(a_clean, set()).add(b_clean)
        elif b_clean in server_ips:
            leaf_port_to_server[(a_clean, a_port)] = (b_clean, b_port)
            server_to_leaves.setdefault(b_clean, set()).add(a_clean)

    # Chain OXC→spine→leaf: each OXC port reaches its connected spine, and
    # through the spine (which acts as a switch) can reach ALL leaves that
    # the spine connects to on any of its ports.
    if spine_ips and oxc_spine_edge and spine_port_to_leaf:
        # Group leaves by spine IP for fan-out
        spine_ip_to_leaves: Dict[str, List[Tuple[str, str]]] = {}
        for (spine_ip, _sp), (leaf_ip, leaf_port) in spine_port_to_leaf.items():
            spine_ip_to_leaves.setdefault(spine_ip, []).append((leaf_ip, leaf_port))

        for (oxc_ip, oxc_port), (spine_ip, _spine_port) in oxc_spine_edge.items():
            leaf_targets = spine_ip_to_leaves.get(spine_ip, [])
            if leaf_targets:
                oxc_port_to_leaves[(oxc_ip, oxc_port)] = list(leaf_targets)

    return oxc_port_to_leaves, leaf_port_to_server, server_to_leaves


def resolve_paths(
    lld: Dict[str, Any],
    crosses: Set[Cross],
    participating_server_ips: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Resolve OXC crosses into a connectivity graph.

    Args:
        lld: parsed lld.json
        crosses: set of (oxc_ip, port_a, port_b) tuples
        participating_server_ips: if given, only include these servers (sub-topology)

    Returns:
        {
            "servers": [{ip, node_id, server_type, leaf_ip}],
            "leaves": [{ip, node_id}],
            "server_leaf_edges": [(server_ip, leaf_ip, link_count)],
            "leaf_leaf_edges": [(leaf_a_ip, leaf_b_ip, oxc_ip, port_a, port_b)],
        }
    """
    topo = lld.get("topology", {})

    _strip = lambda s: re.sub(r"\(\d+\)$", "", s).strip()
    oxc_ips = {_strip(n["node_id"]) for n in topo.get("oxc_nodes", [])}
    all_server_ips = {_strip(n["node_id"]) for n in topo.get("server_nodes", [])}
    all_leaf_ips = {_strip(n["node_id"]) for n in topo.get("leaf_nodes", [])}
    spine_ips = {_strip(n["node_id"]) for n in topo.get("spine_nodes", [])}

    if participating_server_ips is not None:
        target_servers = set(participating_server_ips)
    else:
        target_servers = all_server_ips

    edges = topo.get("edges", [])
    oxc_port_to_leaves, leaf_port_to_server, server_to_leaves = _build_edge_maps(
        edges, oxc_ips, all_server_ips, spine_ips,
    )

    # Determine participating leaves (all leaves connected to participating servers)
    participating_leaves: Set[str] = set()
    for srv_ip in target_servers:
        leaves = server_to_leaves.get(srv_ip, set())
        participating_leaves.update(leaves)

    # Resolve OXC crosses to leaf-leaf edges (N:N: iterate all leaf targets
    # reachable from each OXC port via spine fan-out)
    leaf_leaf_edges: List[Tuple[str, str, str, str, str]] = []
    seen_ll: Set[Tuple[str, str]] = set()
    for (oxc_ip, pa, pb) in crosses:
        targets_a = oxc_port_to_leaves.get((oxc_ip, pa), [])
        targets_b = oxc_port_to_leaves.get((oxc_ip, pb), [])
        if not targets_a or not targets_b:
            logger.debug("Skipping cross (%s, %s, %s): dangling port", oxc_ip, pa, pb)
            continue
        for (leaf_a_ip, _) in targets_a:
            for (leaf_b_ip, _) in targets_b:
                if leaf_a_ip == leaf_b_ip:
                    continue
                if leaf_a_ip in participating_leaves and leaf_b_ip in participating_leaves:
                    pair = tuple(sorted([leaf_a_ip, leaf_b_ip]))
                    if pair not in seen_ll:
                        seen_ll.add(pair)
                        leaf_leaf_edges.append((leaf_a_ip, leaf_b_ip, oxc_ip, pa, pb))

    # Count server-leaf links per pair
    srv_leaf_count: Dict[Tuple[str, str], int] = {}
    for e in edges:
        a_id, b_id = e["a_node_id"], e["b_node_id"]
        if a_id in target_servers and b_id in all_leaf_ips:
            key = (a_id, b_id)
            srv_leaf_count[key] = srv_leaf_count.get(key, 0) + 1
        elif b_id in target_servers and a_id in all_leaf_ips:
            key = (b_id, a_id)
            srv_leaf_count[key] = srv_leaf_count.get(key, 0) + 1

    # Build participating-only oxc_port_map: oxc_ip -> {port -> leaf_ip}
    # N:N: each OXC port can reach multiple leaves; use first participating leaf
    oxc_port_map: Dict[str, Dict[str, str]] = {}
    for (oxc_ip, port), leaf_targets in oxc_port_to_leaves.items():
        for (leaf_ip, _leaf_port) in leaf_targets:
            if leaf_ip in participating_leaves:
                oxc_port_map.setdefault(oxc_ip, {})[port] = leaf_ip
                break

    # Build server info list (ordered by participating_server_ips if given)
    server_node_map = {n["node_id"]: n for n in topo.get("server_nodes", [])}
    leaf_node_map = {n["node_id"]: n for n in topo.get("leaf_nodes", [])}

    ordered_servers = participating_server_ips if participating_server_ips else sorted(target_servers)
    servers = []
    for ip in ordered_servers:
        if ip not in target_servers:
            continue
        node = server_node_map.get(ip, {})
        srv_leaves = sorted(server_to_leaves.get(ip, set()))
        primary_leaf = srv_leaves[0] if srv_leaves else ""
        servers.append({
            "ip": ip,
            "node_id": node.get("node_id", ""),
            "server_type": _chassis_to_npu_type(node.get("chassis_topo", "SERVER")),
            "leaf_ip": primary_leaf,
            "leaf_ips": srv_leaves,
        })

    leaves = []
    for lip in sorted(participating_leaves):
        node = leaf_node_map.get(lip, {})
        leaves.append({"ip": lip, "node_id": node.get("node_id", "")})

    server_leaf_edges = [
        (srv, leaf, cnt) for (srv, leaf), cnt in srv_leaf_count.items()
    ]

    return {
        "servers": servers,
        "leaves": leaves,
        "server_leaf_edges": server_leaf_edges,
        "leaf_leaf_edges": leaf_leaf_edges,
        "oxc_port_map": oxc_port_map,
    }


def split_graph_by_pod(
    graph: Dict[str, Any],
    lld: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Split a flat connectivity graph into per-ODC-domain sub-graphs.

    Each OXC node + its directly-attached leaves + their servers form one pod.
    When the graph cannot be split (single-ODC or no OXC info) the result is
    a single-entry dict, keeping backward compatibility.

    Returns:
        {"<oxc_ip>": sub_graph, ...}  — one entry per OXC domain.
        When there is only one OXC the dict has a single entry.
        When the graph cannot be split (no OXC info) the key is "default".
    """
    servers = graph.get("servers", [])
    leaves = graph.get("leaves", [])
    server_leaf_edges = graph.get("server_leaf_edges", [])
    leaf_leaf_edges = graph.get("leaf_leaf_edges", [])

    # ── Determine pod boundaries ──────────────────────────────────
    # leaf_ip → set of oxc_ips (which OXC nodes this leaf connects to)
    leaf_to_oxc: Dict[str, Set[str]] = {}
    # oxc_ip → set of leaf_ips
    oxc_to_leaves: Dict[str, Set[str]] = {}
    all_oxc_ips: Set[str] = set()

    if lld:
        topo = lld.get("topology", {})
        oxc_ips_from_lld = {n["node_id"] for n in topo.get("oxc_nodes", [])}
        all_leaf_ips = {n["node_id"] for n in topo.get("leaf_nodes", [])}
        for e in topo.get("edges", []):
            a_id, b_id = e["a_node_id"], e["b_node_id"]
            if a_id in oxc_ips_from_lld and b_id in all_leaf_ips:
                leaf, oxc = b_id, a_id
            elif b_id in oxc_ips_from_lld and a_id in all_leaf_ips:
                leaf, oxc = a_id, b_id
            else:
                continue
            leaf_to_oxc.setdefault(leaf, set()).add(oxc)
            oxc_to_leaves.setdefault(oxc, set()).add(leaf)
            all_oxc_ips.add(oxc)

    # Fallback: group by oxc_ip carried inside leaf_leaf_edges
    if not all_oxc_ips:
        for (_la, _lb, oxc_ip, _pa, _pb) in leaf_leaf_edges:
            all_oxc_ips.add(oxc_ip)
        for leaf_info in leaves:
            leaf_to_oxc.setdefault(leaf_info["ip"], set()).update(all_oxc_ips)
        for oxc in all_oxc_ips:
            for leaf_info in leaves:
                oxc_to_leaves.setdefault(oxc, set()).add(leaf_info["ip"])

    # Single-pod → return as-is
    if len(all_oxc_ips) <= 1:
        key = next(iter(all_oxc_ips)) if all_oxc_ips else "default"
        return {key: graph}

    # ── Build sub-graph per OXC ───────────────────────────────────
    # server_ip -> set of leaf_ips (derived from server_leaf_edges)
    srv_leaves_map: Dict[str, Set[str]] = {}
    for (srv_ip, leaf_ip, _cnt) in server_leaf_edges:
        srv_leaves_map.setdefault(srv_ip, set()).add(leaf_ip)

    pods: Dict[str, Dict[str, Any]] = {}
    for oxc_ip in sorted(all_oxc_ips):
        pod_leaf_ips = oxc_to_leaves.get(oxc_ip, set())

        # Servers that connect to any of this pod's leaves
        pod_servers = [
            s for s in servers
            if srv_leaves_map.get(s["ip"], set()) & pod_leaf_ips
        ]
        pod_srv_ips = {s["ip"] for s in pod_servers}
        pod_leaves = [l for l in leaves if l["ip"] in pod_leaf_ips]

        # Leaf-leaf edges: keep only intra-pod pairs
        pod_ll_edges = [
            e for e in leaf_leaf_edges
            if e[0] in pod_leaf_ips and e[1] in pod_leaf_ips
        ]

        # Server-leaf edges: keep only intra-pod pairs
        pod_sl_edges = [
            e for e in server_leaf_edges
            if e[0] in pod_srv_ips and e[1] in pod_leaf_ips
        ]

        # oxc_port_map subset for this OXC
        pod_oxc_port: Dict[str, Dict[str, str]] = {}
        full_oxc_port = graph.get("oxc_port_map", {})
        if oxc_ip in full_oxc_port:
            port_map = full_oxc_port[oxc_ip]
            pod_oxc_port[oxc_ip] = {
                port: leaf for port, leaf in port_map.items()
                if leaf in pod_leaf_ips
            }

        pods[oxc_ip] = {
            "servers": pod_servers,
            "leaves": pod_leaves,
            "server_leaf_edges": pod_sl_edges,
            "leaf_leaf_edges": pod_ll_edges,
            "oxc_port_map": pod_oxc_port,
        }

    return pods
