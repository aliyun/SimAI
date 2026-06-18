"""Monitor API routes: all data derived from XML files.

Directory layout:
    DATA_DIR/
        overview/
            all_pod.xml          ← overview topology (all PODs)
        pods/
            POD#1.xml            ← per-POD detail topology
            POD#2.xml
            ...

The hardware side updates these XML files periodically.
This service re-parses them when file mtime changes (no restart needed).
"""

import logging
import time
from flask import Blueprint, request, jsonify

import DataCollector.config as cfg
from DataCollector.monitor.topology_service import (
    get_overview_topology,
    get_pod_detail,
    get_node_metrics_from_xml,
    get_cluster_summary,
)

logger = logging.getLogger(__name__)

monitor_bp = Blueprint("monitor", __name__, url_prefix="/api/monitor")


# === Topology Endpoints ===

@monitor_bp.route("/overview", methods=["GET"])
def api_overview():
    """Get multi-POD overview topology. Edge metrics parsed from XML labels."""
    xml_path = cfg.get_overview_xml_path()
    logger.info("GET /overview → xml_path=%s, exists=%s", xml_path, __import__('os').path.exists(xml_path))
    topo = get_overview_topology(xml_path)
    logger.info("GET /overview → %d nodes, %d edges", len(topo.get("nodes", [])), len(topo.get("edges", [])))
    return jsonify({"data": topo, "error": None, "timestamp": time.time()})


@monitor_bp.route("/pods", methods=["GET"])
def api_list_pods():
    """List all PODs with summary info."""
    topo = get_overview_topology(cfg.get_overview_xml_path())
    pods = [
        {"pod_id": n["id"], "label": n["label"], "status": "normal", "node_count": 9}
        for n in topo.get("nodes", [])
    ]
    return jsonify({"data": pods, "error": None, "timestamp": time.time()})


@monitor_bp.route("/pods/<path:pod_id>", methods=["GET"])
def api_pod_detail(pod_id):
    """Get single POD internal topology. Edge metrics parsed from XML labels."""
    xml_path = cfg.get_pod_detail_xml_path(pod_id)
    pods_dir = cfg.get_pods_dir()
    logger.info("GET /pods/%s → pods_dir=%s, xml_path=%s", pod_id, pods_dir, xml_path)
    if xml_path is None:
        import os
        available = os.listdir(pods_dir) if os.path.isdir(pods_dir) else []
        logger.warning("POD XML not found: pod_id=%s, pods_dir=%s, available=%s", pod_id, pods_dir, available)
        return jsonify({"data": None, "error": f"POD XML not found: {pod_id}"}), 404
    detail = get_pod_detail(xml_path, pod_id)
    logger.info("GET /pods/%s → %d nodes, %d edges", pod_id, len(detail.get("nodes", [])), len(detail.get("edges", [])))
    if not detail.get("nodes"):
        logger.warning("POD parsed but no nodes: pod_id=%s, xml_path=%s", pod_id, xml_path)
        return jsonify({"data": None, "error": f"POD not found: {pod_id}"}), 404
    return jsonify({"data": detail, "error": None, "timestamp": time.time()})


# === Metrics Endpoints ===

@monitor_bp.route("/metrics/cluster", methods=["GET"])
def api_cluster_metrics():
    """Get cluster-wide summary metrics (aggregated from all pod XMLs)."""
    summary = get_cluster_summary(
        cfg.get_overview_xml_path(),
        cfg.get_pods_dir(),
    )
    summary["timestamp"] = time.time() * 1000
    return jsonify({"data": summary, "error": None, "timestamp": time.time()})


@monitor_bp.route("/metrics/node/<path:node_id>", methods=["GET"])
def api_node_metrics(node_id):
    """Get per-node metrics (searches all pod XMLs for the node)."""
    metrics = get_node_metrics_from_xml(cfg.get_pods_dir(), node_id)
    if metrics is None:
        return jsonify({
            "data": None,
            "error": f"Node not found: {node_id}",
        }), 404
    metrics["timestamp"] = time.time() * 1000
    return jsonify({"data": metrics, "error": None, "timestamp": time.time()})


# === Alerts Endpoints (mock) ===

_MOCK_ALERTS = [
    {
        "alert_id": "ALT-001",
        "alert_type": "link_utilization_high",
        "severity": "warning",
        "pod_id": "POD#1",
        "source_link_id": "POD#1-POD#2",
        "title": "Inter-POD link utilization above threshold",
        "message": "Link POD#1 <-> POD#2 utilization at 85.3%, threshold is 80%",
        "created_at": time.time() - 300,
        "acknowledged": False,
    },
]


@monitor_bp.route("/alerts", methods=["GET"])
def api_list_alerts():
    severity = request.args.get("severity")
    pod_id = request.args.get("pod_id")
    alerts = list(_MOCK_ALERTS)
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    if pod_id:
        alerts = [a for a in alerts if a.get("pod_id") == pod_id]
    return jsonify({"data": {"alerts": alerts, "total": len(alerts)}, "error": None, "timestamp": time.time()})


@monitor_bp.route("/alerts/<alert_id>/acknowledge", methods=["POST"])
def api_acknowledge_alert(alert_id):
    for alert in _MOCK_ALERTS:
        if alert["alert_id"] == alert_id:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = time.time()
            return jsonify({"data": {"ok": True}, "error": None, "timestamp": time.time()})
    return jsonify({"data": None, "error": f"Alert not found: {alert_id}"}), 404


@monitor_bp.route("/healthz", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "subsystem": "monitor"})


# === Frontend-compatible aliases ===

topology_bp = Blueprint("topology", __name__, url_prefix="/api/topology")
metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")


@topology_bp.route("/overview", methods=["GET"])
def api_topology_overview():
    return api_overview()


@topology_bp.route("/pod/<path:pod_id>", methods=["GET"])
def api_topology_pod_detail(pod_id):
    return api_pod_detail(pod_id)


@metrics_bp.route("/cluster", methods=["GET"])
def api_metrics_cluster():
    return api_cluster_metrics()


@metrics_bp.route("/node/<path:node_id>", methods=["GET"])
def api_metrics_node(node_id):
    return api_node_metrics(node_id)
