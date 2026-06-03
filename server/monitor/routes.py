"""Monitor API routes: topology, metrics, and alerts."""

import time
import random
from flask import Blueprint, request, jsonify

from server.config import TOPOLOGY_DIR
from server.monitor.topology_service import get_overview_topology, get_pod_detail, get_pod_link_rates

monitor_bp = Blueprint("monitor", __name__, url_prefix="/api/monitor")


# === Topology Endpoints ===

@monitor_bp.route("/overview", methods=["GET"])
def api_overview():
    """Get multi-POD overview topology."""
    topo = get_overview_topology(TOPOLOGY_DIR)
    return jsonify({"data": topo, "error": None, "timestamp": time.time()})


@monitor_bp.route("/pods", methods=["GET"])
def api_list_pods():
    """List all PODs with summary info."""
    topo = get_overview_topology(TOPOLOGY_DIR)
    pods = []
    for n in topo.get("nodes", []):
        detail = get_pod_detail(TOPOLOGY_DIR, n["id"])
        pods.append({
            "pod_id": n["id"],
            "label": n["label"],
            "status": "normal",
            "node_count": len(detail.get("nodes", [])),
        })
    return jsonify({"data": pods, "error": None, "timestamp": time.time()})


@monitor_bp.route("/pods/<path:pod_id>", methods=["GET"])
def api_pod_detail(pod_id):
    """Get single POD internal topology."""
    detail = get_pod_detail(TOPOLOGY_DIR, pod_id)
    if not detail.get("nodes"):
        return jsonify({"data": None, "error": f"POD not found: {pod_id}"}), 404
    return jsonify({"data": detail, "error": None, "timestamp": time.time()})


@monitor_bp.route("/pods/<path:pod_id>/link-rates", methods=["GET"])
def api_pod_link_rates(pod_id):
    """Get link rate data for a POD.

    Optional query params:
    - source: filter by source node name
    - target: filter by target node name
    """
    link_rates = get_pod_link_rates(TOPOLOGY_DIR, pod_id)
    if not link_rates:
        return jsonify({"data": {}, "error": None, "timestamp": time.time()})

    source = request.args.get("source")
    target = request.args.get("target")

    if source and target:
        key1 = f"{source}-{target}"
        key2 = f"{target}-{source}"
        matched = link_rates.get(key1) or link_rates.get(key2)
        if matched:
            return jsonify({"data": {"ports": matched}, "error": None, "timestamp": time.time()})
        return jsonify({"data": {"ports": []}, "error": None, "timestamp": time.time()})

    return jsonify({"data": link_rates, "error": None, "timestamp": time.time()})


# === Metrics Endpoints (mock data, to be replaced with real source) ===

@monitor_bp.route("/metrics/cluster", methods=["GET"])
def api_cluster_metrics():
    """Get cluster-wide summary metrics."""
    topo = get_overview_topology(TOPOLOGY_DIR)
    pod_nodes = topo.get("nodes", [])
    total_pods = len(pod_nodes)

    total_servers = 0
    for pod_node in pod_nodes:
        detail = get_pod_detail(TOPOLOGY_DIR, pod_node["id"])
        total_servers += sum(
            1 for n in detail.get("nodes", []) if n.get("type") == "SERVER"
        )

    summary = {
        "totalPods": total_pods,
        "totalServers": total_servers,
        "totalNpus": total_servers * 8,
        "overallUtilization": round(0.68 + random.random() * 0.1, 4),
        "activeAlerts": random.randint(0, 3),
        "timestamp": time.time() * 1000,
    }
    return jsonify({"data": summary, "error": None, "timestamp": time.time()})


@monitor_bp.route("/metrics/node/<path:node_id>", methods=["GET"])
def api_node_metrics(node_id):
    """Get per-node live metrics (mock)."""
    status = "warning" if random.random() > 0.95 else "healthy"
    metrics = {
        "nodeId": node_id,
        "cpuUtilization": round(0.3 + random.random() * 0.5, 4),
        "gpuUtilization": round(0.5 + random.random() * 0.4, 4),
        "memoryUsedGb": round(60 + random.random() * 20, 1),
        "memoryTotalGb": 80,
        "temperature": round(55 + random.random() * 25, 1),
        "powerWatts": round(200 + random.random() * 300, 0),
        "status": status,
        "timestamp": time.time() * 1000,
    }
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
    {
        "alert_id": "ALT-002",
        "alert_type": "node_offline",
        "severity": "critical",
        "pod_id": "POD#2",
        "source_node_id": "A3#5",
        "title": "Server node offline",
        "message": "Server A3#5 in POD#2 is unreachable for 120 seconds",
        "created_at": time.time() - 120,
        "acknowledged": False,
    },
]


@monitor_bp.route("/alerts", methods=["GET"])
def api_list_alerts():
    """List alerts, optionally filtered by severity or pod_id."""
    severity = request.args.get("severity")
    pod_id = request.args.get("pod_id")

    alerts = list(_MOCK_ALERTS)
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    if pod_id:
        alerts = [a for a in alerts if a.get("pod_id") == pod_id]

    return jsonify({
        "data": {"alerts": alerts, "total": len(alerts)},
        "error": None,
        "timestamp": time.time(),
    })


@monitor_bp.route("/alerts/<alert_id>/acknowledge", methods=["POST"])
def api_acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    for alert in _MOCK_ALERTS:
        if alert["alert_id"] == alert_id:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = time.time()
            return jsonify({"data": {"ok": True}, "error": None, "timestamp": time.time()})
    return jsonify({"data": None, "error": f"Alert not found: {alert_id}"}), 404


# === Health ===

@monitor_bp.route("/healthz", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "subsystem": "monitor"})


# === Frontend-compatible aliases ===
# The React frontend calls /api/topology/* and /api/metrics/*
# These aliases route to the same handlers.

topology_bp = Blueprint("topology", __name__, url_prefix="/api/topology")
metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")


@topology_bp.route("/overview", methods=["GET"])
def api_topology_overview():
    return api_overview()


@topology_bp.route("/pod/<path:pod_id>", methods=["GET"])
def api_topology_pod_detail(pod_id):
    return api_pod_detail(pod_id)


@topology_bp.route("/pod/<path:pod_id>/link-rates", methods=["GET"])
def api_topology_pod_link_rates(pod_id):
    return api_pod_link_rates(pod_id)


@metrics_bp.route("/cluster", methods=["GET"])
def api_metrics_cluster():
    return api_cluster_metrics()


@metrics_bp.route("/node/<path:node_id>", methods=["GET"])
def api_metrics_node(node_id):
    return api_node_metrics(node_id)
