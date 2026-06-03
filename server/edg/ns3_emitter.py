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
    """Scan leaf port_infos in lld to detect link bandwidth."""
    if not lld:
        return None
    for leaf in lld.get("topology", {}).get("leaf_nodes", []):
        for port in leaf.get("port_infos", []):
            bw = _parse_port_bandwidth(port.get("port_name", ""))
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
        logger.info("Auto-detected bandwidth from lld port_name: %s", bandwidth)

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

    # Node ID assignment:
    # 0 .. num_npus-1               = NPUs
    # num_npus .. total_nodes-1     = Leaf switches
    leaf_start = num_npus

    # Map leaf IP to node ID
    leaf_ip_to_id: Dict[str, int] = {}
    for i, leaf in enumerate(leaves):
        leaf_ip_to_id[leaf["ip"]] = leaf_start + i

    lines: List[str] = []

    # Links:
    #   per-server full-mesh NPU<->NPU  = num_servers * C(npu_per_server, 2)
    #   NPU<->Leaf (one per NPU)        = num_npus
    #   Leaf<->Leaf (cross-server)      = len(leaf_leaf_edges)
    full_mesh_per_server = npu_per_server * (npu_per_server - 1) // 2
    total_full_mesh_links = num_servers * full_mesh_per_server
    npu_leaf_links = num_npus
    leaf_leaf_links = len(leaf_leaf_edges)
    total_links = total_full_mesh_links + npu_leaf_links + leaf_leaf_links

    # Line 1: header
    lines.append(
        f"{total_nodes} {npu_per_server} {num_nv_switches} "
        f"{num_leaf_switches} {total_links} {header_npu_type}"
    )

    # Line 2: switch node IDs (only leaf switches)
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

    # NPU <-> Leaf links (one direct uplink per NPU for cross-server traffic)
    for srv_idx, srv in enumerate(servers):
        leaf_ip = srv.get("leaf_ip", "")
        leaf_id = leaf_ip_to_id.get(leaf_ip)
        if leaf_id is None:
            logger.warning(
                "Server %s has no leaf mapping, skipping NPU-leaf links", srv["ip"]
            )
            continue
        npu_base = srv_idx * npu_per_server
        for npu_offset in range(npu_per_server):
            npu_id = npu_base + npu_offset
            lines.append(f"{npu_id} {leaf_id} {bandwidth} {latency} {error_rate}")

    # Leaf <-> Leaf links (from OXC crosses)
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
