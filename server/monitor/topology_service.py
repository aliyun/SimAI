"""Topology XML parser: converts draw.io XML to API-ready data structures."""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

_cache: Dict[str, dict] = {}
_cache_mtime: Dict[str, float] = {}


def parse_drawio_xml(xml_path: str) -> dict:
    """Parse a draw.io XML file into nodes, edges, and labels."""
    mtime = os.path.getmtime(xml_path)
    if xml_path in _cache and _cache_mtime.get(xml_path) == mtime:
        return _cache[xml_path]

    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes = []
    edges = []
    edge_labels = {}

    # Also scan <object> elements for edge port mappings
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
        # Also check child mxCell for edge data
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
            # Skip mxCells with empty id (children of <object> already handled above)
            # and skip duplicates
            existing_ids = {e["id"] for e in edges}
            if cell_id and cell_id not in existing_ids:
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

    # Attach edge labels to their parent edges
    for label_data in edge_labels.values():
        parent_id = label_data.get("parent", "")
        for edge in edges:
            if edge["id"] == parent_id and not edge.get("label"):
                edge["label"] = label_data["text"]

    result = {"nodes": nodes, "edges": edges, "edge_labels": edge_labels}
    _cache[xml_path] = result
    _cache_mtime[xml_path] = mtime
    return result


def _clean_html(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "\n", text)
    clean = re.sub(r"&[a-z]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _extract_metadata(value: str) -> dict:
    metadata = {}
    if not value:
        return metadata
    clean = re.sub(r"<[^>]+>", "\n", value)
    lines = [l.strip() for l in clean.split("\n") if l.strip()]

    # New format: "E0319_SSW_182\nSPINE\n10.44.231.182"
    # or "worker-172\nSERVER\n10.44.231.172"
    # Detect by checking if lines do NOT contain ":"
    has_kv = any(":" in l for l in lines)

    if not has_kv and len(lines) >= 2:
        # New format: line0=name, line1=type, line2=ip (optional)
        node_name = lines[0].strip()
        node_type = lines[1].strip().upper() if len(lines) > 1 else ""
        node_ip = lines[2].strip() if len(lines) > 2 else ""
        metadata["node_id"] = node_name
        if node_type:
            metadata["node_type"] = node_type
        if node_ip:
            metadata["node_ip"] = node_ip
        return metadata

    # Legacy format: "node_id: LEAF#1\nnode_type: LEAF\nnode_ip: 102.101.101.101"
    for line in lines:
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().replace(" ", "_").lower()
            val = val.strip()
            if key and val:
                metadata[key] = val
    return metadata


def parse_edge_label(label: str) -> dict:
    """Parse edge label into structured metrics.

    Supports two formats:
    - Legacy: '0.61% ~x6 0.31%' -> utilization, linkCount, errorRate
    - New range: '0.01%~0.08%' -> uplinkErrorRange / downlinkErrorRange
    """
    metrics = {"utilization": 0.0, "linkCount": 1, "errorRate": 0.0}
    if not label:
        return metrics

    # Check for range format: "0.01%~0.08%"
    range_match = re.match(r"(\d+\.?\d*%)\s*~\s*(\d+\.?\d*%)\s*$", label.strip())
    if range_match:
        # Range format — store as-is, parse first value as utilization
        val = float(range_match.group(1).rstrip("%")) / 100
        metrics["utilization"] = val
        metrics["errorRate"] = val
        return metrics

    # Legacy format: "1.47% ~x4 0.73%"
    percents = re.findall(r"(\d+\.?\d*)%", label)
    if len(percents) >= 1:
        metrics["utilization"] = float(percents[0]) / 100
    if len(percents) >= 2:
        metrics["errorRate"] = float(percents[1]) / 100
    link_match = re.search(r"~?x(\d+)", label)
    if link_match:
        metrics["linkCount"] = int(link_match.group(1))
    return metrics


def _collect_edge_labels_for_edge(edge_id: str, edge_labels: dict) -> tuple:
    """Collect uplink/downlink labels for an edge from its child edgeLabels.

    Returns (uplink_label, downlink_label) strings, or (None, None).
    """
    labels = []
    for el in edge_labels.values():
        if el.get("parent") == edge_id and el.get("text"):
            labels.append(el)

    if len(labels) >= 2:
        # Sort by geometry x: negative x = uplink (toward source), positive = downlink
        labels.sort(key=lambda l: l.get("x", 0))
        return labels[0]["text"].strip(), labels[1]["text"].strip()
    elif len(labels) == 1:
        return labels[0]["text"].strip(), None
    return None, None


def _resolve_overview_xml_path(project_root: str) -> str:
    """Resolve the overview topology XML path.

    Search order:
    1. topology/overview/all_pod.xml
    2. all_pod.xml (backward-compatible fallback)
    """
    primary = os.path.join(project_root, "topology", "overview", "all_pod.xml")
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(project_root, "all_pod.xml")
    if os.path.exists(fallback):
        return fallback
    return ""


def get_overview_topology(project_root: str) -> dict:
    """Return multi-POD overview topology from all_pod.xml."""
    xml_path = _resolve_overview_xml_path(project_root)
    if not xml_path:
        return {"nodes": [], "edges": []}

    raw = parse_drawio_xml(xml_path)
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

                # Collect uplink/downlink labels
                uplink, downlink = _collect_edge_labels_for_edge(e["id"], raw["edge_labels"])

                label_text = e.get("label", "")
                if not label_text and not uplink:
                    for el in raw["edge_labels"].values():
                        if el.get("parent") == e["id"]:
                            label_text = el["text"]
                            break

                metrics = parse_edge_label(uplink or label_text)
                # Only set range fields when labels use range format (e.g. "0.01%~0.08%")
                if uplink and "~" in uplink and not re.search(r"x\d+", uplink):
                    metrics["uplinkErrorRange"] = uplink
                if downlink and "~" in downlink and not re.search(r"x\d+", downlink):
                    metrics["downlinkErrorRange"] = downlink

                edge_data = {
                    "id": f"{src_label}-{tgt_label}",
                    "source": src_label,
                    "target": tgt_label,
                    "metrics": metrics,
                }

                # Pass per-side metrics for overview labels (proximity principle).
                # uplink = near source (negative x), downlink = near target (positive x).
                if uplink:
                    edge_data["sourceMetrics"] = parse_edge_label(uplink)
                if downlink:
                    edge_data["targetMetrics"] = parse_edge_label(downlink)

                edges.append(edge_data)

    # Add standalone edge labels as metrics for nearby edges
    return {"nodes": nodes, "edges": edges}


def _resolve_pod_xml_path(project_root: str, pod_id: str) -> str:
    """Resolve the XML file path for a given pod_id.

    Search order:
    1. topology/pods/<pod_id>.xml  (e.g., topology/pods/POD#1.xml)
    2. a_pod.xml                   (backward-compatible fallback)

    Returns the first path that exists, or empty string if none found.
    """
    pods_dir = os.path.join(project_root, "topology", "pods")
    pod_file = os.path.join(pods_dir, f"{pod_id}.xml")
    if os.path.exists(pod_file):
        return pod_file

    fallback = os.path.join(project_root, "a_pod.xml")
    if os.path.exists(fallback):
        return fallback

    return ""


def get_pod_detail(project_root: str, pod_id: str) -> dict:
    """Return single-POD detail topology from per-POD XML file.

    Looks for pods/<pod_id>.xml first, falls back to a_pod.xml.
    """
    xml_path = _resolve_pod_xml_path(project_root, pod_id)
    if not xml_path:
        return {"podId": pod_id, "paraPlaneId": pod_id, "nodes": [], "edges": [], "superNodes": []}

    raw = parse_drawio_xml(xml_path)
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
            "size": {"width": n["width"], "height": n["height"]},
            "metadata": {
                "nodeIp": meta.get("node_ip"),
                "serverType": meta.get("server_type"),
                "superNodeId": meta.get("super_node_id"),
            },
        })

    # Assign servers to super_nodes by geometric containment
    for sn in super_nodes:
        bb = sn["boundingBox"]
        for node in nodes:
            if node["type"] == "SERVER":
                nx, ny = node["position"]["x"], node["position"]["y"]
                if (bb["x"] <= nx <= bb["x"] + bb["width"] and
                        bb["y"] <= ny <= bb["y"] + bb["height"]):
                    sn["serverIds"].append(node["id"])
                    node["metadata"]["superNodeId"] = sn["id"]

    # Pre-load link rate data once (outside edge loop for performance)
    link_rates = get_pod_link_rates(project_root, pod_id)

    edges = []
    for e in raw["edges"]:
        src = e.get("source")
        tgt = e.get("target")
        if src and tgt:
            src_label = id_to_label.get(src, src)
            tgt_label = id_to_label.get(tgt, tgt)

            # Collect uplink/downlink labels from child edgeLabels
            uplink, downlink = _collect_edge_labels_for_edge(e["id"], raw["edge_labels"])

            # Fallback to single label on the edge itself
            label_text = e.get("label", "")
            if not label_text and not uplink:
                for el in raw["edge_labels"].values():
                    if el.get("parent") == e["id"]:
                        label_text = el["text"]
                        break

            metrics = parse_edge_label(uplink or label_text)
            if uplink:
                metrics["uplinkErrorRange"] = uplink
            if downlink:
                metrics["downlinkErrorRange"] = downlink

            edge_data = {
                "id": f"{src_label}-{tgt_label}-{e['id']}",
                "source": src_label,
                "target": tgt_label,
                "metrics": metrics,
            }

            # Attach port details from link rate CSV if available
            key1 = f"{src_label}-{tgt_label}"
            key2 = f"{tgt_label}-{src_label}"
            matched_key = None
            for k in [key1, key2]:
                if k in link_rates:
                    matched_key = k
                    break
            if matched_key:
                edge_data["portDetails"] = link_rates[matched_key]
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


def _parse_link_rate_csv(csv_path: str) -> Dict[str, List[dict]]:
    """Parse the pod link rate CSV file.

    CSV format: 边ID, 下层节点名称, 上层节点名称, 端口名称, in速率, out速率

    Returns:
        Dict mapping "lower_node-upper_node" -> list of port entries
    """
    if not os.path.exists(csv_path):
        return {}

    result: Dict[str, List[dict]] = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return result

        # Skip header
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue

            edge_id = parts[0]
            lower_node = parts[1]
            upper_node = parts[2]
            port = parts[3]
            in_rate = parts[4]
            out_rate = parts[5]

            # Create key based on node names for matching
            key = f"{lower_node}-{upper_node}"

            entry = {
                'edge_id': edge_id,
                'lower_node': lower_node,
                'upper_node': upper_node,
                'port': port,
                'in_rate': in_rate,
                'out_rate': out_rate,
            }

            if key not in result:
                result[key] = []
            result[key].append(entry)

    except Exception:
        pass

    return result


def _resolve_link_rate_csv_path(project_root: str, pod_id: str) -> str:
    """Resolve the link rate CSV path for a given pod_id."""
    link_rate_dir = os.path.join(project_root, "topology", "pod_link_rate")
    csv_path = os.path.join(link_rate_dir, f"{pod_id}_link_rate.csv")
    if os.path.exists(csv_path):
        return csv_path
    return ""


def get_pod_link_rates(project_root: str, pod_id: str) -> Dict[str, List[dict]]:
    """Get link rate data for a given pod from its CSV file.

    Returns:
        Dict mapping "lower_node-upper_node" -> list of port details
    """
    csv_path = _resolve_link_rate_csv_path(project_root, pod_id)
    if not csv_path:
        return {}
    return _parse_link_rate_csv(csv_path)
