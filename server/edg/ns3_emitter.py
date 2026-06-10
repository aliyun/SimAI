"""Emit NS3-format topology file from a resolved connectivity graph.

Output format matches gen_Topo_Template.py (Rail_Opti_SingleToR):
  Line 1: <total_nodes> <npu_per_server> <nv_switch_num> <switch_nodes> <links> <npu_type>
  Line 2: space-separated switch node IDs
  Lines 3+: <src> <dst> <bandwidth> <latency> <error_rate>
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_NPU_PER_SERVER = int(os.environ.get("AS_EDG_NPU_PER_SERVER", "8"))
DEFAULT_NPU_TYPE = os.environ.get("AS_EDG_NPU_TYPE", "A3")
DEFAULT_BW = os.environ.get("AS_EDG_BW", "100Gbps")
DEFAULT_INTRA_BW = os.environ.get("AS_EDG_INTRA_BW", "2400Gbps")
DEFAULT_LATENCY = os.environ.get("AS_EDG_LATENCY", "0.0005ms")
DEFAULT_NV_LATENCY = os.environ.get("AS_EDG_NV_LATENCY", "0.000025ms")
DEFAULT_ERROR_RATE = os.environ.get("AS_EDG_ERROR_RATE", "0")
DEFAULT_AP_BW = os.environ.get("AS_EDG_AP_BW", "100Gbps")

# NPU model -> server-internal bandwidth (Gb/s per NPU link)
NPU_INTRA_BW_MAP: Dict[str, str] = {
    "A2": "200Gbps",
    "A3": "400Gbps",
    "A5": "400Gbps",
    "A6": "400Gbps",
    "A100": "2400Gbps",
    "H100": "2880Gbps",
    "H800": "2880Gbps",
}


def _parse_port_bandwidth(port_name: str) -> Optional[str]:
    """Extract bandwidth from port_name like '400GE/0/1/1' -> '400Gbps'."""
    m = re.match(r"(\d+)GE", port_name)
    if m:
        return f"{m.group(1)}Gbps"
    return None


def _detect_bandwidth_from_lld(lld: Optional[Dict[str, Any]]) -> Optional[str]:
    """Scan leaf port_id_list in lld to detect link bandwidth."""
    if not lld:
        return None
    for leaf in lld.get("topology", {}).get("leaf_nodes", []):
        for port in leaf.get("port_id_list", []):
            bw = _parse_port_bandwidth(port.get("port_id", ""))
            if bw:
                return bw
    return None


def write_ns3_topology(
    graph: Dict[str, Any],
    out_path: str,
    npu_per_server: int = DEFAULT_NPU_PER_SERVER,
    npu_type: str = DEFAULT_NPU_TYPE,
    bandwidth: str = DEFAULT_BW,
    intra_bw: str = DEFAULT_INTRA_BW,
    latency: str = DEFAULT_LATENCY,
    nv_latency: str = DEFAULT_NV_LATENCY,
    error_rate: str = DEFAULT_ERROR_RATE,
    ap_bandwidth: str = DEFAULT_AP_BW,
    lld: Optional[Dict[str, Any]] = None,
) -> str:
    """Write NS3 topology file. Returns the output path.

    Topology model: NO NVSwitch. Every server uses full-mesh NPU<->NPU
    intra-connect (1-hop) plus one NPU->Leaf uplink per NPU for
    cross-server traffic. Cross-server paths are NPU->Leaf->Leaf->NPU.
    """
    detected_bw = _detect_bandwidth_from_lld(lld)
    if detected_bw and bandwidth == DEFAULT_BW:
        bandwidth = detected_bw
        ap_bandwidth = detected_bw
        logger.info("Auto-detected bandwidth from lld port_id: %s", bandwidth)

    servers: List[Dict] = graph["servers"]
    leaves: List[Dict] = graph["leaves"]
    leaf_leaf_edges = graph["leaf_leaf_edges"]

    num_servers = len(servers)
    num_npus = num_servers * npu_per_server
    num_nv_switches = 0
    num_leaf_switches = len(leaves)
    num_switch_nodes = num_leaf_switches
    total_nodes = num_npus + num_switch_nodes

    # Resolve per-server intra-bandwidth from each server's NPU type
    server_intra_bw: Dict[int, str] = {}
    detected_types: set = set()
    for srv_idx, srv in enumerate(servers):
        srv_type = srv.get("server_type", npu_type)
        detected_types.add(srv_type)
        bw = NPU_INTRA_BW_MAP.get(srv_type)
        if bw is None:
            bw = intra_bw
        server_intra_bw[srv_idx] = bw

    header_npu_type = npu_type
    if len(detected_types) == 1:
        header_npu_type = next(iter(detected_types))
    elif detected_types:
        header_npu_type = "MIXED"

    # Build unfolded topology when lld is available
    spine_ips: List[str] = []
    oxc_ips: List[str] = []
    oxc_crosses: Dict[str, List[Tuple[str, str]]] = {}  # oxc_ip -> [(port_a, port_b)]
    spine_to_leaf: Dict[str, List[str]] = {}  # spine_ip -> [leaf_ip]
    leaf_to_spine: Dict[str, str] = {}  # leaf_ip -> spine_ip
    spine_to_oxc: Dict[str, List[str]] = {}  # spine_ip -> [oxc_ip]

    if lld:
        topo = lld.get("topology", {})
        _strip = lambda s: __import__("re").sub(r"\(\d+\)$", "", s).strip()
        spine_ips = sorted({_strip(n["node_id"]) for n in topo.get("spine_nodes", [])})
        oxc_ips_raw = sorted({_strip(n["node_id"]) for n in topo.get("oxc_nodes", [])})
        all_leaf_ips = {_strip(n["node_id"]) for n in topo.get("leaf_nodes", [])}

        # Build leaf↔spine and spine↔OXC mappings from edges
        for e in topo.get("edges", []):
            a_id = _strip(e["a_node_id"])
            b_id = _strip(e["b_node_id"])
            a_port = str(e.get("a_node_port_id", ""))
            b_port = str(e.get("b_node_port_id", ""))

            # Leaf ↔ Spine
            if a_id in all_leaf_ips and b_id in spine_ips:
                leaf_to_spine[a_id] = b_id
                spine_to_leaf.setdefault(b_id, []).append(a_id)
            elif b_id in all_leaf_ips and a_id in spine_ips:
                leaf_to_spine[b_id] = a_id
                spine_to_leaf.setdefault(a_id, []).append(b_id)

            # Spine ↔ OXC
            if a_id in spine_ips and b_id in oxc_ips_raw:
                spine_to_oxc.setdefault(a_id, []).append(b_id)
            elif b_id in spine_ips and a_id in oxc_ips_raw:
                spine_to_oxc.setdefault(b_id, []).append(a_id)

        # Use only 1 OXC to avoid multi-path loops; pick the first one
        oxc_ips = oxc_ips_raw[:1] if oxc_ips_raw else []

    # Map leaf IP to node ID (needed before unfolded link pre-compute)
    leaf_start = num_npus
    leaf_ip_to_id: Dict[str, int] = {}
    for i, leaf in enumerate(leaves):
        leaf_ip_to_id[leaf["ip"]] = leaf_start + i

    num_spines = len(spine_ips)
    num_oxc = len(oxc_ips)
    spine_start = num_npus + num_leaf_switches
    oxc_start = spine_start + num_spines
    num_switch_nodes = num_leaf_switches + num_spines + num_oxc
    total_nodes = num_npus + num_switch_nodes

    # Map spine/OXC IPs to node IDs
    spine_ip_to_id: Dict[str, int] = {}
    for i, sip in enumerate(spine_ips):
        spine_ip_to_id[sip] = spine_start + i
    oxc_ip_to_id: Dict[str, int] = {}
    for i, oip in enumerate(oxc_ips):
        oxc_ip_to_id[oip] = oxc_start + i

    # Pre-compute unfolded links
    leaf_spine_links: List[Tuple[int, int]] = []
    spine_oxc_links: List[Tuple[int, int]] = []
    if lld and oxc_ips:
        for leaf_ip, spine_ip in leaf_to_spine.items():
            leaf_id = leaf_ip_to_id.get(leaf_ip)
            spine_id = spine_ip_to_id.get(spine_ip)
            if leaf_id is not None and spine_id is not None:
                leaf_spine_links.append((leaf_id, spine_id))
        for spine_ip, oxc_list in spine_to_oxc.items():
            spine_id = spine_ip_to_id.get(spine_ip)
            if spine_id is not None:
                for oip in oxc_list:
                    oid = oxc_ip_to_id.get(oip)
                    if oid is not None:
                        spine_oxc_links.append((spine_id, oid))

    lines: List[str] = []

    # Links:
    #   per-server full-mesh NPU<->NPU  = num_servers * C(npu_per_server, 2)
    #   NPU<->Leaf (one per NPU)        = num_npus
    #   Leaf<->Leaf (cross-server)      = len(leaf_leaf_edges)
    full_mesh_per_server = npu_per_server * (npu_per_server - 1) // 2
    total_full_mesh_links = num_servers * full_mesh_per_server
    leaf_leaf_links = len(leaf_leaf_edges)
    # Each NPU connects to all its server's leaves (one port per leaf)
    npu_leaf_links = 0
    for srv in servers:
        leaf_ips = srv.get("leaf_ips", [])
        if not leaf_ips:
            leaf_ip = srv.get("leaf_ip", "")
            leaf_ips = [leaf_ip] if leaf_ip else []
        leaf_count = len([lip for lip in leaf_ips if leaf_ip_to_id.get(lip) is not None])
        npu_leaf_links += npu_per_server * leaf_count if leaf_count else npu_per_server
    unfolded_links = len(leaf_spine_links) + len(spine_oxc_links)
    total_links = total_full_mesh_links + npu_leaf_links + (unfolded_links if unfolded_links else leaf_leaf_links)

    # Line 1: header
    lines.append(
        f"{total_nodes} {npu_per_server} {num_nv_switches} "
        f"{num_switch_nodes} {total_links} {header_npu_type}"
    )

    # Line 2: switch node IDs (leaf + spine + OXC)
    switch_ids = list(range(leaf_start, total_nodes))
    lines.append(" ".join(str(x) for x in switch_ids))

    # Full-mesh NPU<->NPU links per server (intra-server, per-server bandwidth)
    for srv_idx in range(num_servers):
        npu_base = srv_idx * npu_per_server
        link_bw = server_intra_bw.get(srv_idx, intra_bw)
        for i in range(npu_per_server):
            for j in range(i + 1, npu_per_server):
                lines.append(
                    f"{npu_base + i} {npu_base + j} {link_bw} {nv_latency} {error_rate}"
                )

    # NPU <-> Leaf links (each NPU connects to ALL assigned leaves via per-port links)
    for srv_idx, srv in enumerate(servers):
        leaf_ips = srv.get("leaf_ips", [])
        if not leaf_ips:
            leaf_ip = srv.get("leaf_ip", "")
            leaf_ips = [leaf_ip] if leaf_ip else []

        leaf_ids = [leaf_ip_to_id.get(lip) for lip in leaf_ips]
        leaf_ids = [lid for lid in leaf_ids if lid is not None]
        if not leaf_ids:
            logger.warning(
                "Server %s has no leaf mapping, skipping NPU-leaf links", srv["ip"]
            )
            continue

        npu_base = srv_idx * npu_per_server
        for npu_offset in range(npu_per_server):
            npu_id = npu_base + npu_offset
            # Each NPU connects to ALL leaves (one port per leaf)
            for leaf_id in leaf_ids:
                lines.append(f"{npu_id} {leaf_id} {bandwidth} {latency} {error_rate}")
                npu_leaf_links += 1

    if unfolded_links:
        # Unfolded: Leaf→Spine + Spine→OXC (OXC acts as circuit switch)
        for (leaf_id, spine_id) in leaf_spine_links:
            lines.append(f"{leaf_id} {spine_id} {bandwidth} {latency} {error_rate}")
        for (spine_id, oxc_id) in spine_oxc_links:
            lines.append(f"{spine_id} {oxc_id} {bandwidth} {latency} {error_rate}")
    else:
        # Folded: Leaf↔Leaf (OXC cross abstracted as direct link)
        for (leaf_a_ip, leaf_b_ip, _oxc_ip, _pa, _pb) in leaf_leaf_edges:
            a_id = leaf_ip_to_id.get(leaf_a_ip)
            b_id = leaf_ip_to_id.get(leaf_b_ip)
            if a_id is not None and b_id is not None:
                lines.append(f"{a_id} {b_id} {ap_bandwidth} {latency} {error_rate}")

    os.makedirs(
        os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True
    )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    logger.info(
        "NS3 topology written: %s (npus=%d, leaves=%d, links=%d, npu=%s, per_server_bw=%s)",
        out_path,
        num_npus,
        num_leaf_switches,
        total_links,
        header_npu_type,
        {
            srv["ip"]: server_intra_bw.get(i, "?")
            for i, srv in enumerate(servers)
        },
    )
    return out_path
