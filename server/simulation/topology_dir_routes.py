"""Flask routes for topology directory management."""

import logging
import os

from flask import Blueprint, request, jsonify

from server.auth.auth_service import require_auth
import server.config as config

logger = logging.getLogger(__name__)

topology_dir_bp = Blueprint("topology_dir", __name__, url_prefix="/api/simulation/topology-dir")


@topology_dir_bp.route("", methods=["POST"])
@require_auth
def api_set_topology_dir():
    """Update the topology directory path."""
    data = request.get_json(silent=True) or {}
    path = data.get("path", "")

    if not path:
        return jsonify({"error": "path is required"}), 400

    # Resolve relative paths from PROJECT_ROOT, not CWD
    if not os.path.isabs(path):
        abs_path = os.path.join(config.PROJECT_ROOT, path)
    else:
        abs_path = path
    if not os.path.isdir(abs_path):
        return jsonify({"error": f"Directory does not exist: {abs_path}"}), 400

    config.TOPOLOGY_DIR = abs_path
    logger.info("Topology directory updated to: %s", abs_path)

    return jsonify({"path": abs_path})


@topology_dir_bp.route("", methods=["GET"])
@require_auth
def api_get_topology_dir():
    """Get current topology directory path."""
    return jsonify({"path": config.TOPOLOGY_DIR})


@topology_dir_bp.route("/scan", methods=["GET"])
@require_auth
def api_scan_topology_dir():
    """Scan topology directory for XML/CSV files."""
    topo_dir = config.TOPOLOGY_DIR

    if not os.path.isdir(topo_dir):
        return jsonify({"error": f"Topology directory not found: {topo_dir}", "files": []}), 404

    files = []
    try:
        for entry in sorted(os.listdir(topo_dir)):
            entry_path = os.path.join(topo_dir, entry)

            if os.path.isdir(entry_path):
                # Scan subdirectory for topology files
                sub_files = []
                for sub in sorted(os.listdir(entry_path)):
                    if sub.endswith((".xml", ".csv", ".txt")):
                        sub_files.append(sub)
                if sub_files:
                    files.append({
                        "name": entry,
                        "type": "directory",
                        "path": entry_path,
                        "children": sub_files,
                    })
            elif entry.endswith((".xml", ".csv", ".txt")):
                stat = os.stat(entry_path)
                files.append({
                    "name": entry,
                    "type": "file",
                    "path": entry_path,
                    "size": stat.st_size,
                })
    except OSError as e:
        logger.exception("Failed to scan topology directory")
        return jsonify({"error": str(e), "files": []}), 500

    return jsonify({"path": topo_dir, "files": files})
