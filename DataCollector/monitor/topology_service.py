"""Topology XML parser: all 4 API responses derived from two XML files.

The hardware side only needs to periodically update all_pod.xml and a_pod.xml.
This service re-parses them when file mtime changes (no restart needed).

Edge label format (in draw.io edge labels):
    "0.61% ~x6 0.31% 400Gbps 3.2us"
    → utilization=0.0061, linkCount=6, errorRate=0.0031, bandwidthGbps=400, latencyUs=3.2

Node metadata format (in draw.io node HTML value, key:value per line):
    node_id: A3#1
    node_type: SERVER
    node_ip: 101.101.101.101
    server_type: A3_8P
    cpu_utilization: 0.45
    gpu_utilization: 0.82
    memory_used_gb: 62.4
    memory_total_gb: 80
    temperature: 72
    power_watts: 450
    status: healthy
"""

import glob
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


# --- mtime-based cache ---

_cache: Dict[str, Tuple[float, dict]] = {}  # path → (mtime, parsed_data)


def _parse_cached(xml_path: str) -> dict:
    """Parse XML with mtime-based cache. Re-parses when file changes."""
    try:
        mtime = os.path.getmtime(xml_path)
    except OSError:
        return {"nodes": [], "edges": [], "edge_labels": {}}

    cached = _cache.get(xml_path)
    if cached and cached[0] == mtime:
        return cached[1]

    result = _parse_drawio_xml(xml_path)
    _cache[xml_path] = (mtime, result)
    return result


def clear_cache() -> None:
    """Force clear all cached data."""
    _cache.clear()


# --- XML parsing ---

def _parse_drawio_xml(xml_path: str) -> dict:
    """Parse a draw.io XML file into nodes, edges, and labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes = []
    edges = []
    edge_labels = {}

    object_attrs = {}
    for obj in root.iter("object"):
        obj_id = obj.get("id", "")
        attrs = {}
        for key in ("A", "B"):
            val = obj.get(key)
            if val:
                attrs[key] = [int(x.strip()) for x in val.split(",") if x.strip()]
        if attrs:
            object_attrs[obj_id] = attrs
        for cell in obj.findall("mxCell"):
            cell_id = cell.get("id", obj_id)
            source = cell.get("source")
            target = cell.get("target")
            if source or target:
                edges.append({
                    "id": cell_id,
                    "source": source,
                    "target": target,
                    "label": "",
                    "port_mapping": attrs if attrs else None,
                })

    for cell in root.iter("mxCell"):
        cell_id = cell.get("id", "")
        value = cell.get("value", "")
        style = cell.get("style", "")
        parent = cell.get("parent", "")

        if cell_id in ("0", "1"):
            continue

        source = cell.get("source")
        target = cell.get("target")

        if source or target or "edge" in cell.attrib:
            existing_ids = {e["id"] for e in edges}
            if cell_id not in existing_ids:
                edges.append({
                    "id": cell_id,
                    "source": source,
                    "target": target,
                    "label": _clean_html(value),
                    "port_mapping": object_attrs.get(cell_id),
                })
        elif "edgeLabel" in style:
            geom = cell.find("mxGeometry")
            edge_labels[cell_id] = {
                "text": _clean_html(value),
                "parent": parent,
                "x": float(geom.get("x", 0)) if geom is not None else 0,
                "y": float(geom.get("y", 0)) if geom is not None else 0,
            }
        elif cell.get("vertex") == "1":
            geom = cell.find("mxGeometry")
            if geom is not None:
                nodes.append({
                    "id": cell_id,
                    "label": _clean_html(value),
                    "raw_value": value,
                    "x": float(geom.get("x", 0)),
                    "y": float(geom.get("y", 0)),
                    "width": float(geom.get("width", 40)),
                    "height": float(geom.get("height", 40)),
                    "style": style,
                    "metadata": _extract_metadata(value),
                })

    for label_data in edge_labels.values():
        parent_id = label_data.get("parent", "")
        for edge in edges:
            if edge["id"] == parent_id and not edge.get("label"):
                edge["label"] = label_data["text"]

    return {"nodes": nodes, "edges": edges, "edge_labels": edge_labels}


def _clean_html(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "\n", text)
    clean = re.sub(r"&[a-z]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _extract_metadata(value: str) -> dict:
    """Extract key:value pairs from node HTML content."""
    metadata = {}
    if not value:
        return metadata
    clean = re.sub(r"<[^>]+>", "\n", value)
    for line in clean.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().replace(" ", "_").lower()
            val = val.strip()
            if key and val:
                metadata[key] = val
    return metadata


def _safe_float(val: Optional[str], default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# --- Edge label parsing ---

def parse_edge_label(label: str) -> dict:
    """Parse edge label into structured metrics.

    Supported formats:
        "0.61% ~x6 0.31%"                         → basic
        "0.61% ~x6 0.31% 400Gbps"                 → with bandwidth
        "0.61% ~x6 0.31% 400Gbps 3.2us"           → full
        "utilization:0.61% linkCount:6 errorRate:0.31% bandwidth:400Gbps latency:3.2us"  → key:value
    """
    metrics: dict = {"utilization": 0.0, "linkCount": 1, "errorRate": 0.0}
    if not label:
        return metrics

    # Percentages → utilization, errorRate
    percents = re.findall(r"(\d+\.?\d*)%", label)
    if len(percents) >= 1:
        metrics["utilization"] = float(percents[0]) / 100
    if len(percents) >= 2:
        metrics["errorRate"] = float(percents[1]) / 100

    # ~x6 → linkCount
    link_match = re.search(r"~?x(\d+)", label)
    if link_match:
        metrics["linkCount"] = int(link_match.group(1))

    # 400Gbps → bandwidthGbps
    bw_match = re.search(r"(\d+\.?\d*)\s*[Gg]bps", label)
    if bw_match:
        metrics["bandwidthGbps"] = float(bw_match.group(1))

    # 3.2us → latencyUs
    lat_match = re.search(r"(\d+\.?\d*)\s*us", label)
    if lat_match:
        metrics["latencyUs"] = float(lat_match.group(1))

    return metrics


# --- Public API: Topology ---

def get_overview_topology(xml_path: str) -> dict:
    """Return multi-POD overview topology."""
    if not os.path.exists(xml_path):
        return {"nodes": [], "edges": []}

    raw = _parse_cached(xml_path)
    id_to_label = {}
    nodes = []

    for n in raw["nodes"]:
        if "ellipse" in n.get("style", ""):
            label = n["label"]
            id_to_label[n["id"]] = label
            nodes.append({
                "id": label,
                "type": "POD",
                "label": label,
                "position": {"x": n["x"], "y": n["y"]},
                "metadata": n.get("metadata", {}),
            })

    edges = []
    seen_pairs = set()
    for e in raw["edges"]:
        src = e.get("source")
        tgt = e.get("target")
        if src and tgt:
            src_label = id_to_label.get(src, src)
            tgt_label = id_to_label.get(tgt, tgt)
            pair = tuple(sorted([src_label, tgt_label]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                label_text = e.get("label", "")
                if not label_text:
                    for el in raw["edge_labels"].values():
                        if el.get("parent") == e["id"]:
                            label_text = el["text"]
                            break
                edges.append({
                    "id": f"{src_label}-{tgt_label}",
                    "source": src_label,
                    "target": tgt_label,
                    "metrics": parse_edge_label(label_text),
                })

    return {"nodes": nodes, "edges": edges}


def get_pod_detail(xml_path: str, pod_id: str) -> dict:
    """Return single-POD detail topology."""
    if not os.path.exists(xml_path):
        return {"podId": pod_id, "paraPlaneId": pod_id, "nodes": [], "edges": [], "superNodes": []}

    raw = _parse_cached(xml_path)
    id_to_label = {}
    nodes = []
    super_nodes = []

    for n in raw["nodes"]:
        meta = n.get("metadata", {})
        node_type = meta.get("node_type", "").upper()
        label = meta.get("node_id", n["label"])

        if "super_node_id" in meta:
            super_nodes.append({
                "id": meta["super_node_id"],
                "serverIds": [],
                "boundingBox": {
                    "x": n["x"], "y": n["y"],
                    "width": n["width"], "height": n["height"],
                },
            })
            continue

        if "para_plane_id" in meta:
            continue

        if node_type not in ("OXC", "SPINE", "LEAF", "SERVER"):
            if meta.get("server_type"):
                node_type = "SERVER"
            elif "ellipse" in n.get("style", ""):
                node_type = "POD"
            else:
                continue

        id_to_label[n["id"]] = label
        nodes.append({
            "id": label,
            "type": node_type,
            "label": label,
            "position": {"x": n["x"], "y": n["y"]},
            "metadata": {
                "nodeIp": meta.get("node_ip"),
                "serverType": meta.get("server_type"),
                "superNodeId": meta.get("super_node_id"),
            },
        })

    for sn in super_nodes:
        bb = sn["boundingBox"]
        for node in nodes:
            if node["type"] == "SERVER":
                nx, ny = node["position"]["x"], node["position"]["y"]
                if (bb["x"] <= nx <= bb["x"] + bb["width"] and
                        bb["y"] <= ny <= bb["y"] + bb["height"]):
                    sn["serverIds"].append(node["id"])
                    node["metadata"]["superNodeId"] = sn["id"]

    edges = []
    for e in raw["edges"]:
        src = e.get("source")
        tgt = e.get("target")
        if src and tgt:
            src_label = id_to_label.get(src, src)
            tgt_label = id_to_label.get(tgt, tgt)
            label_text = e.get("label", "")
            edge_data = {
                "id": f"{src_label}-{tgt_label}-{e['id']}",
                "source": src_label,
                "target": tgt_label,
                "metrics": parse_edge_label(label_text),
            }
            if e.get("port_mapping"):
                pm = e["port_mapping"]
                edge_data["portMapping"] = {
                    "sourcePorts": pm.get("A", []),
                    "targetPorts": pm.get("B", []),
                }
            edges.append(edge_data)

    return {
        "podId": pod_id,
        "paraPlaneId": pod_id,
        "nodes": nodes,
        "edges": edges,
        "superNodes": super_nodes,
    }


# --- Public API: Node Metrics (from XML metadata) ---

def _list_pod_xmls(pods_dir: str) -> List[str]:
    """List all per-POD XML files in the pods directory."""
    if not os.path.isdir(pods_dir):
        return []
    return sorted(glob.glob(os.path.join(pods_dir, "*.xml")))


def get_node_metrics_from_xml(xml_dir: str, node_id: str) -> Optional[dict]:
    """Extract live metrics for a single node by searching all pod XMLs.

    The hardware updater should write these fields into the node's value:
        cpu_utilization: 0.45
        gpu_utilization: 0.82
        memory_used_gb: 62.4
        memory_total_gb: 80
        temperature: 72
        power_watts: 450
        status: healthy
    """
    for xml_path in _list_pod_xmls(xml_dir):
        raw = _parse_cached(xml_path)
        for n in raw["nodes"]:
            meta = n.get("metadata", {})
            label = meta.get("node_id", n["label"])
            if label == node_id:
                return {
                    "nodeId": node_id,
                    "cpuUtilization": _safe_float(meta.get("cpu_utilization")),
                    "gpuUtilization": _safe_float(meta.get("gpu_utilization")),
                    "memoryUsedGb": _safe_float(meta.get("memory_used_gb")),
                    "memoryTotalGb": _safe_float(meta.get("memory_total_gb"), 80),
                    "temperature": _safe_float(meta.get("temperature")),
                    "powerWatts": _safe_float(meta.get("power_watts")),
                    "status": meta.get("status", "unknown"),
                }
    return None


# --- Public API: Cluster Summary (aggregated from all pod XMLs) ---

def get_cluster_summary(overview_xml: str, xml_dir: str) -> dict:
    """Aggregate cluster summary from overview XML and all pod detail XMLs."""
    overview = get_overview_topology(overview_xml)
    total_pods = len(overview.get("nodes", []))

    total_servers = 0
    total_gpus = 0
    util_sum = 0.0
    util_count = 0

    for xml_path in _list_pod_xmls(xml_dir):
        raw = _parse_cached(xml_path)
        for n in raw["nodes"]:
            meta = n.get("metadata", {})
            if meta.get("server_type") or meta.get("node_type", "").upper() == "SERVER":
                total_servers += 1
                total_gpus += 8  # default GPUs per server
                gpu_util = meta.get("gpu_utilization")
                if gpu_util is not None:
                    util_sum += _safe_float(gpu_util)
                    util_count += 1

    overall_util = round(util_sum / util_count, 4) if util_count > 0 else 0.0

    return {
        "totalPods": total_pods,
        "totalServers": total_servers,
        "totalGpus": total_gpus,
        "overallUtilization": overall_util,
        "activeAlerts": 0,
    }
