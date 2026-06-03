"""Flask application entry point with API blueprints."""

import sys; sys.dont_write_bytecode = True  # source is truth, never use stale .pyc

import logging
import os
import subprocess

def _ensure_packages():
    """Auto-install Flask and flask-cors if missing."""
    for pkg, import_name in [("flask", "flask"), ("flask-cors", "flask_cors")]:
        try:
            __import__(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure_packages()

from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import SERVER_HOST, SERVER_PORT, SECRET_KEY
from server.db.database import init_db
from server.auth.auth_service import login, logout, verify_token, require_auth
from server.workspace.workspace_service import (
    get_workspace_info,
    save_file,
    load_file,
    list_files,
    cleanup_stale_workspaces,
)
from server.process.process_service import (
    launch,
    kill_process,
    list_processes,
    get_logs,
    cleanup_dead_processes,
)
from server.monitor.routes import monitor_bp, topology_bp, metrics_bp
from server.simulation.workload_routes import workload_bp as sim_workload_bp
from server.simulation.ranktable_routes import ranktable_bp as sim_ranktable_bp
from server.simulation.results_routes import results_bp as sim_results_bp
from server.simulation.topology_dir_routes import topology_dir_bp as sim_topology_dir_bp
from server.ops.ops_routes import ops_bp
from server.edg import edg_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Blueprints ---

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
workspace_bp = Blueprint("workspace", __name__, url_prefix="/api/workspace")
files_bp = Blueprint("files", __name__, url_prefix="/api/files")
process_bp = Blueprint("process", __name__, url_prefix="/api/process")


# === Auth API ===

@auth_bp.route("/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")

    token, username, error = login(username, password)
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"token": token, "username": username})


@auth_bp.route("/logout", methods=["POST"])
@require_auth
def api_logout():
    logout(request.session_token)
    return jsonify({"ok": True})


@auth_bp.route("/me", methods=["GET"])
@require_auth
def api_me():
    return jsonify({
        "username": request.username,
        "workspace": request.workspace_dir,
    })


# === Workspace API ===

@workspace_bp.route("/info", methods=["GET"])
@require_auth
def api_workspace_info():
    info = get_workspace_info(request.workspace_dir)
    return jsonify(info)


# === Files API ===

@files_bp.route("/save", methods=["POST"])
@require_auth
def api_save_file():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    content = data.get("content", "")

    if not filename:
        return jsonify({"error": "filename is required"}), 400

    try:
        path = save_file(request.workspace_dir, filename, content)
        return jsonify({"path": path, "filename": filename})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@files_bp.route("/load", methods=["GET"])
@require_auth
def api_load_file():
    filename = request.args.get("filename", "")
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    try:
        content, path = load_file(request.workspace_dir, filename)
        return jsonify({"content": content, "path": path, "filename": filename})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@files_bp.route("/list", methods=["GET"])
@require_auth
def api_list_files():
    files = list_files(request.workspace_dir)
    return jsonify({"files": files})


# === Process API ===

@process_bp.route("/launch", methods=["POST"])
@require_auth
def api_launch():
    params = request.get_json(silent=True) or {}

    tracking_id, pid, error = launch(
        session_token=request.session_token,
        username=request.username,
        workspace_dir=request.workspace_dir,
        params=params,
    )
    if error:
        status_code = 429 if "limit" in error.lower() else 400
        return jsonify({"error": error}), status_code
    return jsonify({"tracking_id": tracking_id, "pid": pid, "status": "running"})


@process_bp.route("/list", methods=["GET"])
@require_auth
def api_list_processes():
    processes = list_processes(username=request.username)
    return jsonify({"processes": processes})


@process_bp.route("/logs/<int:pid>", methods=["GET"])
@require_auth
def api_get_logs(pid):
    lines, status, error = get_logs(request.username, pid)
    if error:
        return jsonify({"error": error}), 403 if "Permission" in error else 404
    return jsonify({"logs": lines, "status": status, "pid": pid})


@process_bp.route("/<int:pid>", methods=["GET"])
@require_auth
def api_get_process(pid):
    """Get a single process entry by PID."""
    processes = list_processes(username=request.username)
    for p in processes:
        if p["pid"] == pid:
            return jsonify(p)
    return jsonify({"error": "Process not found"}), 404


@process_bp.route("/<int:pid>", methods=["DELETE"])
@require_auth
def api_kill_process(pid):
    success, error = kill_process(request.username, pid)
    if not success:
        return jsonify({"error": error}), 403 if "Permission" in error else 404
    return jsonify({"ok": True, "status": "killed"})


# --- App Factory ---

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY
    CORS(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(workspace_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(process_bp)
    app.register_blueprint(monitor_bp)
    app.register_blueprint(topology_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(sim_workload_bp)
    app.register_blueprint(sim_ranktable_bp)
    app.register_blueprint(sim_results_bp)
    app.register_blueprint(sim_topology_dir_bp)
    app.register_blueprint(edg_bp)
    app.register_blueprint(ops_bp)

    @app.route("/healthz", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    # Startup tasks
    with app.app_context():
        init_db()
        cleanup_dead_processes()
        cleanup_stale_workspaces()

    logger.info("SimAI server initialized")
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SimAI Dashboard Server")
    parser.add_argument("-t", "--topology-dir", help="Path to topology XML directory")
    parser.add_argument("-p", "--port", type=int, help="Server port (default: 5000)")
    parser.add_argument("--host", help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()

    from server import config
    if args.topology_dir:
        config.TOPOLOGY_DIR = os.path.abspath(args.topology_dir)
    if args.port:
        config.SERVER_PORT = args.port
    if args.host:
        config.SERVER_HOST = args.host

    app = create_app()
    logger.info("Topology dir: %s", config.TOPOLOGY_DIR)
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)
