#!/usr/bin/env python3
"""Test live XML update: modifies edge metrics every second and polls API to verify."""

import os
import re
import sys
import time
import random
import urllib.request
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERVIEW_XML = os.path.join(PROJECT_ROOT, "topology", "overview", "all_pod.xml")
PODS_DIR = os.path.join(PROJECT_ROOT, "topology", "pods")
API_BASE = os.environ.get("API_BASE", "http://localhost:5050")

ITERATIONS = int(os.environ.get("ITERATIONS", "10"))
INTERVAL = 1.0


def update_edge_metrics(xml_path: str, util_percent: float, link_count: int):
    """Replace edge label metrics in a draw.io XML file."""
    with open(xml_path, "r") as f:
        content = f.read()

    # Replace patterns like "0.61%" or "0.31%" with new values
    err_percent = round(util_percent * 0.5, 2)
    new_label = f"{util_percent:.2f}%\\n~x{link_count}\\n{err_percent:.2f}%"

    # Replace edgeLabel values containing percent patterns
    content = re.sub(
        r'(style="edgeLabel[^"]*"[^>]*>)\s*<mxGeometry',
        r'\1<mxGeometry',
        content,
    )

    # Simpler: just replace the metric text in value attributes
    # Pattern: number% ~xN number%
    content = re.sub(
        r'(\d+\.?\d*)%(\s|&#xa;|\\n)*~x(\d+)(\s|&#xa;|\\n)*(\d+\.?\d*)%',
        f'{util_percent:.2f}%\\n~x{link_count}\\n{err_percent:.2f}%',
        content,
    )

    with open(xml_path, "w") as f:
        f.write(content)


def api_get(path: str) -> dict:
    """Simple GET request."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def main():
    pod_files = sorted(
        [os.path.join(PODS_DIR, f) for f in os.listdir(PODS_DIR) if f.endswith(".xml")]
    ) if os.path.isdir(PODS_DIR) else []
    print(f"=== Live Update Test ({ITERATIONS} iterations, {INTERVAL}s interval) ===")
    print(f"API: {API_BASE}")
    print(f"Overview XML: {OVERVIEW_XML}")
    print(f"POD XMLs: {len(pod_files)} files")
    print()

    for i in range(ITERATIONS):
        # Generate random metrics
        util = round(random.uniform(0.1, 5.0), 2)
        links = random.choice([4, 6, 8, 12, 16])

        # Update overview and all POD XML files
        update_edge_metrics(OVERVIEW_XML, util, links)
        for pf in pod_files:
            update_edge_metrics(pf, util, links)

        # Wait a beat for file write to flush
        time.sleep(0.1)

        # Query API
        try:
            overview = api_get("/api/topology/overview")
            edges = overview["data"]["edges"]

            pod1 = api_get("/api/topology/pod/POD%231")
            pod1_edges = pod1["data"]["edges"]

            # Check overview metrics
            overview_utils = [
                e["metrics"]["utilization"] for e in edges if e["metrics"]["utilization"] > 0
            ]
            overview_links = [
                e["metrics"]["linkCount"] for e in edges if e["metrics"]["linkCount"] > 1
            ]

            # Check POD#1 metrics
            pod1_utils = [
                e["metrics"]["utilization"] for e in pod1_edges if e["metrics"]["utilization"] > 0
            ]

            # Find expected values
            expected_util = util / 100  # percent to ratio
            ov_ok = any(abs(u - expected_util) < 0.001 for u in overview_utils) or links in overview_links
            pod_ok = any(abs(u - expected_util) < 0.001 for u in pod1_utils)

            status = "OK" if (ov_ok and pod_ok) else ("POD_STALE" if ov_ok else "STALE")
            print(
                f"  [{i+1:2d}/{ITERATIONS}] util={util}% links={links} "
                f"| overview={[f'{u:.4f}' for u in overview_utils[:3]]} "
                f"| pod1={[f'{u:.4f}' for u in pod1_utils[:3]]} "
                f"| {status}"
            )

        except Exception as e:
            print(f"  [{i+1:2d}/{ITERATIONS}] ERROR: {e}")

        time.sleep(INTERVAL - 0.1)

    print()
    print("=== Test Complete ===")


if __name__ == "__main__":
    main()
