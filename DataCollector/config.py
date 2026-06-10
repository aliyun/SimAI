"""DataCollector configuration.

All settings can be overridden via environment variables or CLI arguments.
CLI arguments take precedence over environment variables.

Directory layout:
    DATA_DIR/
        overview/
            all_pod.xml          ← overview topology (all PODs)
        pods/
            POD#1.xml            ← per-POD detail topology
            POD#2.xml
            ...
"""

import os
import re
from typing import Optional


# --- Server ---
SERVER_HOST = os.environ.get("DC_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("DC_PORT", "5000"))

# --- Data paths ---
DATA_DIR = os.environ.get("DC_DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
))


def get_data_dir() -> str:
    """Return the base data directory."""
    return DATA_DIR


def get_overview_dir() -> str:
    """Return the directory containing overview XML."""
    return os.path.join(DATA_DIR, "overview")


def get_pods_dir() -> str:
    """Return the directory containing per-POD detail XMLs."""
    return os.path.join(DATA_DIR, "pods")


def get_overview_xml_path() -> str:
    """Return the path to the overview topology XML."""
    return os.path.join(DATA_DIR, "overview", "all_pod.xml")


def _sanitize_pod_id(pod_id: str) -> str:
    """Sanitize pod_id for use as filename: 'POD#1' → 'pod_1'."""
    return re.sub(r"[^a-zA-Z0-9]", "_", pod_id).strip("_").lower()


def get_pod_detail_xml_path(pod_id: str) -> Optional[str]:
    """Find the XML file for a specific POD.

    Search order:
        1. pods/<pod_id>.xml         (exact, e.g. POD#1.xml)
        2. pods/<sanitized>.xml      (sanitized, e.g. pod_1.xml)
    """
    pods_dir = get_pods_dir()
    candidates = [
        os.path.join(pods_dir, f"{pod_id}.xml"),
        os.path.join(pods_dir, f"{_sanitize_pod_id(pod_id)}.xml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None
