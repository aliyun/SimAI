"""DataCollector — lightweight topology & metrics API for hardware deployment.

Usage:
    python -m DataCollector.app [OPTIONS]

Options:
    --host HOST         Listen address (default: 0.0.0.0)
    --port PORT         Listen port (default: 5000)
    --data-dir DIR      Base data directory (contains overview/ and pods/)

Environment variables (lower priority than CLI):
    DC_HOST, DC_PORT, DC_DATA_DIR
"""

import argparse
import logging
import os
import sys
import subprocess


def _ensure_packages():
    """Auto-install Flask and flask-cors if missing."""
    for pkg, import_name in [("flask", "flask"), ("flask-cors", "flask_cors")]:
        try:
            __import__(import_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


_ensure_packages()

from flask import Flask, jsonify
from flask_cors import CORS

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import DataCollector.config as cfg
from DataCollector.monitor.routes import monitor_bp, topology_bp, metrics_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("DataCollector")


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(monitor_bp)
    app.register_blueprint(topology_bp)
    app.register_blueprint(metrics_bp)

    @app.route("/healthz", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    logger.info("DataCollector initialized")
    logger.info("  data_dir     = %s (exists=%s)", cfg.get_data_dir(), os.path.isdir(cfg.get_data_dir()))
    logger.info("  overview_dir = %s (exists=%s)", cfg.get_overview_dir(), os.path.isdir(cfg.get_overview_dir()))
    logger.info("  pods_dir     = %s (exists=%s)", cfg.get_pods_dir(), os.path.isdir(cfg.get_pods_dir()))
    overview_xml = cfg.get_overview_xml_path()
    logger.info("  overview_xml = %s (exists=%s)", overview_xml, os.path.isfile(overview_xml))
    pods_dir = cfg.get_pods_dir()
    if os.path.isdir(pods_dir):
        pod_files = [f for f in os.listdir(pods_dir) if f.endswith(".xml")]
        logger.info("  pod XMLs     = %s", pod_files if pod_files else "(none)")
    else:
        logger.warning("  pods_dir does not exist: %s", pods_dir)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="DataCollector",
        description="SimAI topology & metrics data collection service",
    )
    parser.add_argument("--host", default=None, help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Listen port (default: 5000)")
    parser.add_argument("--data-dir", default=None, help="Base data directory (contains overview/ and pods/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.host is not None:
        cfg.SERVER_HOST = args.host
    if args.port is not None:
        cfg.SERVER_PORT = args.port
    if args.data_dir is not None:
        cfg.DATA_DIR = os.path.abspath(args.data_dir)

    app = create_app()
    app.run(host=cfg.SERVER_HOST, port=cfg.SERVER_PORT, debug=False)


if __name__ == "__main__":
    main()
