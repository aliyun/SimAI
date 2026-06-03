"""Resolve OXC crosses + lld.json edges into a connectivity graph.

The graph contains only the servers listed in npu_match (sub-topology).
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from server.edg.crosses import Cross

logger = logging.getLogger(__name__)


def _build_edge_maps(edges: List[Dict[str, Any]], oxc_ips: Set[str], server_ips: Set[str]):
    """Index lld.topology.edges into lookup tables.

    Returns:
        oxc_port_to_leaf: {(oxc_ip, port) -> (leaf_ip, leaf_port)}
        leaf_port_to_server: {(leaf_ip, port) -> (server_ip, server_port)}
        server_to_leaf: {server_ip -> leaf_ip}
    """
    oxc_port_to_leaf: Dict[Tuple[str, str], Tuple[str, str]] = {}
    leaf_port_to_server: Dict[Tuple[str, str], Tuple[str, str]] = {}
    server_to_leaf: Dict[str, str] = {}

    for e in edges:
        a_ip, b_ip = e["a_node_ip"], e["b_node_ip"]
        a_port, b_port = str(e["a_node_port_id"]), str(e["b_node_port_id"])

        if a_ip in oxc_ips:
            oxc_port_to_leaf[(a_ip, a_port)] = (b_ip, b_port)
        elif b_ip in oxc_ips:
            oxc_port_to_leaf[(b_ip, b_port)] = (a_ip, a_port)

        if a_ip in server_ips:
            leaf_port_to_server[(b_ip, b_port)] = (a_ip, a_port)
            server_to_leaf[a_ip] = b_ip
        elif b_ip in server_ips:
            leaf_port_to_server[(a_ip, a_port)] = (b_ip, b_port)
            server_to_leaf[b_ip] = a_ip

    return oxc_port_to_leaf, leaf_port_to_server, server_to_leaf


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

    oxc_ips = {n["node_ip"] for n in topo.get("oxc_nodes", [])}
    all_server_ips = {n["node_ip"] for n in topo.get("server_nodes", [])}
    all_leaf_ips = {n["node_ip"] for n in topo.get("leaf_nodes", [])}

    if participating_server_ips is not None:
        target_servers = set(participating_server_ips)
    else:
        target_servers = all_server_ips

    edges = topo.get("edges", [])
    oxc_port_to_leaf, leaf_port_to_server, server_to_leaf = _build_edge_maps(
        edges, oxc_ips, all_server_ips,
    )

    # Determine participating leaves (those connected to participating servers)
    participating_leaves: Set[str] = set()
    for srv_ip in target_servers:
        leaf_ip = server_to_leaf.get(srv_ip)
        if leaf_ip:
            participating_leaves.add(leaf_ip)

    # Resolve OXC crosses to leaf-leaf edges
    leaf_leaf_edges: List[Tuple[str, str, str, str, str]] = []
    for (oxc_ip, pa, pb) in crosses:
        leaf_a = oxc_port_to_leaf.get((oxc_ip, pa))
        leaf_b = oxc_port_to_leaf.get((oxc_ip, pb))
        if not leaf_a or not leaf_b:
            logger.debug("Skipping cross (%s, %s, %s): dangling port", oxc_ip, pa, pb)
            continue
        leaf_a_ip = leaf_a[0]
        leaf_b_ip = leaf_b[0]
        if leaf_a_ip == leaf_b_ip:
            continue
        if leaf_a_ip in participating_leaves and leaf_b_ip in participating_leaves:
            leaf_leaf_edges.append((leaf_a_ip, leaf_b_ip, oxc_ip, pa, pb))

    # Count server-leaf links per pair
    srv_leaf_count: Dict[Tuple[str, str], int] = {}
    for e in edges:
        a_ip, b_ip = e["a_node_ip"], e["b_node_ip"]
        if a_ip in target_servers and b_ip in all_leaf_ips:
            key = (a_ip, b_ip)
            srv_leaf_count[key] = srv_leaf_count.get(key, 0) + 1
        elif b_ip in target_servers and a_ip in all_leaf_ips:
            key = (b_ip, a_ip)
            srv_leaf_count[key] = srv_leaf_count.get(key, 0) + 1

    # Build participating-only oxc_port_map: oxc_ip -> {port -> leaf_ip}
    # Only include ports whose attached leaf is part of this (sub-)topology.
    oxc_port_map: Dict[str, Dict[str, str]] = {}
    for (oxc_ip, port), (leaf_ip, _leaf_port) in oxc_port_to_leaf.items():
        if leaf_ip in participating_leaves:
            oxc_port_map.setdefault(oxc_ip, {})[port] = leaf_ip

    # Build server info list (ordered by participating_server_ips if given)
    server_node_map = {n["node_ip"]: n for n in topo.get("server_nodes", [])}
    leaf_node_map = {n["node_ip"]: n for n in topo.get("leaf_nodes", [])}

    ordered_servers = participating_server_ips if participating_server_ips else sorted(target_servers)
    servers = []
    for ip in ordered_servers:
        if ip not in target_servers:
            continue
        node = server_node_map.get(ip, {})
        loc = node.get("node_location", {})
        servers.append({
            "ip": ip,
            "node_id": loc.get("node_id", ""),
            "server_type": node.get("server_type", "SERVER"),
            "leaf_ip": server_to_leaf.get(ip, ""),
        })

    leaves = []
    for lip in sorted(participating_leaves):
        node = leaf_node_map.get(lip, {})
        loc = node.get("node_location", {})
        leaves.append({"ip": lip, "node_id": loc.get("node_id", "")})

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
        oxc_ips_from_lld = {n["node_ip"] for n in topo.get("oxc_nodes", [])}
        all_leaf_ips = {n["node_ip"] for n in topo.get("leaf_nodes", [])}
        for e in topo.get("edges", []):
            a_ip, b_ip = e["a_node_ip"], e["b_node_ip"]
            if a_ip in oxc_ips_from_lld and b_ip in all_leaf_ips:
                leaf, oxc = b_ip, a_ip
            elif b_ip in oxc_ips_from_lld and a_ip in all_leaf_ips:
                leaf, oxc = a_ip, b_ip
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
    # server_ip -> leaf_ip mapping (derived from server_leaf_edges)
    srv_leaf_map: Dict[str, str] = {}
    for (srv_ip, leaf_ip, _cnt) in server_leaf_edges:
        srv_leaf_map[srv_ip] = leaf_ip

    pods: Dict[str, Dict[str, Any]] = {}
    for oxc_ip in sorted(all_oxc_ips):
        pod_leaf_ips = oxc_to_leaves.get(oxc_ip, set())

        # Servers that connect to any of this pod's leaves
        pod_servers = [
            s for s in servers
            if srv_leaf_map.get(s["ip"]) in pod_leaf_ips
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
