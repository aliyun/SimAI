#!/usr/bin/env python3
"""Robustness test: swap topology XMLs and verify rendering via Playwright."""

import json
import os
import shutil
import subprocess
import signal
import time
import sys

PROJECT_ROOT = "/Users/anthony/PycharmProjects/SimAI"
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")
TOPO_DIR = os.path.join(PROJECT_ROOT, "topology")
TEST_CASES_DIR = os.path.join(TOPO_DIR, "test_cases")
OVERVIEW_XML = os.path.join(TOPO_DIR, "overview", "all_pod.xml")
SCREENSHOT_DIR = "/tmp/topo_test_screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Test case definitions
OVERVIEW_TESTS = [
    {
        "name": "4pods_grid",
        "xml": os.path.join(TEST_CASES_DIR, "overview_4pods.xml"),
        "expected_nodes": 4,
        "expected_edges": 5,  # 4 sides + 1 diagonal
        "pod_ids": ["POD#1", "POD#2", "POD#3", "POD#4"],
    },
    {
        "name": "5pods_pentagon",
        "xml": os.path.join(TEST_CASES_DIR, "overview_5pods.xml"),
        "expected_nodes": 5,
        "expected_edges": 5,  # ring
        "pod_ids": ["POD#1", "POD#2", "POD#3", "POD#4", "POD#5"],
    },
    {
        "name": "6pods_grid",
        "xml": os.path.join(TEST_CASES_DIR, "overview_6pods.xml"),
        "expected_nodes": 6,
        "expected_edges": 7,  # 2 rows of 2 + 3 vertical
        "pod_ids": ["POD#1", "POD#2", "POD#3", "POD#4", "POD#5", "POD#6"],
    },
    {
        "name": "7pods_star",
        "xml": os.path.join(TEST_CASES_DIR, "overview_7pods.xml"),
        "expected_nodes": 7,
        "expected_edges": 6,  # center to 6 outer
        "pod_ids": ["POD#1", "POD#2", "POD#3", "POD#4", "POD#5", "POD#6", "POD#7"],
    },
]

POD_DETAIL_TESTS = [
    {
        "name": "pod_large",
        "xml": os.path.join(TEST_CASES_DIR, "pod_large.xml"),
        "expected_nodes": 16,  # 1 OXC + 3 SPINE + 4 LEAF + 8 SERVER
        "expected_edges": 15,  # 3 OXC-SPINE + 4 SPINE-LEAF + 8 LEAF-SERVER
    },
    {
        "name": "pod_minimal",
        "xml": os.path.join(TEST_CASES_DIR, "pod_minimal.xml"),
        "expected_nodes": 4,  # 1 SPINE + 1 LEAF + 2 SERVER
        "expected_edges": 3,  # 1 SPINE-LEAF + 2 LEAF-SERVER
    },
]


def clear_cache():
    """Clear the topology service cache by restarting the backend."""
    # Kill existing backend
    subprocess.run("lsof -ti:5001 | xargs kill -9 2>/dev/null", shell=True)
    time.sleep(1)
    # Start new backend
    env = os.environ.copy()
    env["SIMAI_SERVER_PORT"] = "5001"
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.app"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for backend to be ready
    for _ in range(20):
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:5001/api/topology/overview", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    return proc


def run_playwright_test(test_name, test_js):
    """Run a Playwright test script and return stdout."""
    result = subprocess.run(
        ["node", "-e", test_js],
        cwd=DASHBOARD_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout, result.stderr, result.returncode


def test_overview(test_case):
    """Test an overview topology."""
    name = test_case["name"]
    print(f"\n{'='*60}")
    print(f"TESTING OVERVIEW: {name}")
    print(f"{'='*60}")

    # Swap XML
    shutil.copy2(test_case["xml"], OVERVIEW_XML)
    print(f"  Swapped overview XML to {name}")

    # Restart backend to clear cache
    proc = clear_cache()

    # Verify API response first
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://localhost:5001/api/topology/overview", timeout=5)
        data = json.loads(resp.read())["data"]
        api_nodes = len(data["nodes"])
        api_edges = len(data["edges"])
        print(f"  API: {api_nodes} nodes, {api_edges} edges")

        if api_nodes != test_case["expected_nodes"]:
            print(f"  FAIL: Expected {test_case['expected_nodes']} nodes, got {api_nodes}")
        if api_edges != test_case["expected_edges"]:
            print(f"  WARN: Expected {test_case['expected_edges']} edges, got {api_edges}")

        # Check sourceMetrics/targetMetrics present
        for edge in data["edges"]:
            if "sourceMetrics" not in edge:
                print(f"  FAIL: Edge {edge['id']} missing sourceMetrics")
            if "targetMetrics" not in edge:
                print(f"  FAIL: Edge {edge['id']} missing targetMetrics")
            else:
                sm = edge["sourceMetrics"]
                tm = edge["targetMetrics"]
                if sm == tm:
                    print(f"  WARN: Edge {edge['id']} has identical source/target metrics")
    except Exception as e:
        print(f"  FAIL: API error: {e}")

    # Playwright test
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"overview_{name}.png")
    js = f"""
const {{ chromium }} = require('@playwright/test');
(async () => {{
  const browser = await chromium.launch();
  const ctx = await browser.newContext({{ viewport: {{ width: 1920, height: 1080 }} }});
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', msg => {{ if (msg.type() === 'error') errors.push(msg.text()); }});

  await page.goto('http://localhost:4173/', {{ waitUntil: 'networkidle' }});
  await page.waitForTimeout(4000);

  const nodes = await page.locator('.react-flow__node').count();
  const edges = await page.locator('.react-flow__edge').count();
  const labels = await page.locator('.react-flow__edgelabel-renderer .text-center').all();

  const labelData = [];
  for (let i = 0; i < labels.length; i++) {{
    const text = await labels[i].textContent();
    const box = await labels[i].boundingBox();
    labelData.push({{ text: text?.replace(/\\s+/g, ' ').trim(), x: box?.x, y: box?.y, w: box?.width, h: box?.height }});
  }}

  // Check for overlaps between labels
  const overlaps = [];
  for (let i = 0; i < labelData.length; i++) {{
    for (let j = i + 1; j < labelData.length; j++) {{
      const a = labelData[i], b = labelData[j];
      if (a.x != null && b.x != null) {{
        const overlapX = a.x < b.x + b.w && a.x + a.w > b.x;
        const overlapY = a.y < b.y + b.h && a.y + a.h > b.y;
        if (overlapX && overlapY) overlaps.push([i, j]);
      }}
    }}
  }}

  await page.screenshot({{ path: '{screenshot_path}', fullPage: true }});

  console.log(JSON.stringify({{
    nodes, edges, labelCount: labels.length, labels: labelData, overlaps, errors
  }}));
  await browser.close();
}})();
"""
    stdout, stderr, rc = run_playwright_test(name, js)
    if rc != 0:
        print(f"  FAIL: Playwright error: {stderr[:500]}")
        return False

    try:
        result = json.loads(stdout.strip())
    except Exception:
        print(f"  FAIL: Could not parse Playwright output: {stdout[:500]}")
        return False

    print(f"  UI: {result['nodes']} nodes, {result['edges']} edges, {result['labelCount']} labels")

    ok = True
    if result["nodes"] != test_case["expected_nodes"]:
        print(f"  FAIL: Expected {test_case['expected_nodes']} nodes, got {result['nodes']}")
        ok = False
    if result["edges"] != test_case["expected_edges"]:
        print(f"  WARN: Expected {test_case['expected_edges']} edges, got {result['edges']}")

    expected_labels = test_case["expected_edges"] * 2  # 2 labels per edge
    if result["labelCount"] != expected_labels:
        print(f"  WARN: Expected {expected_labels} labels, got {result['labelCount']}")

    if result["overlaps"]:
        print(f"  WARN: {len(result['overlaps'])} label overlaps detected: {result['overlaps']}")
        for pair in result["overlaps"]:
            i, j = pair
            print(f"    Labels {i} & {j}: '{result['labels'][i]['text']}' vs '{result['labels'][j]['text']}'")

    if result["errors"]:
        print(f"  WARN: Console errors: {result['errors'][:3]}")

    print(f"  Screenshot: {screenshot_path}")
    return ok


def test_pod_detail(test_case):
    """Test a pod detail topology."""
    name = test_case["name"]
    print(f"\n{'='*60}")
    print(f"TESTING POD DETAIL: {name}")
    print(f"{'='*60}")

    # Swap POD#1 XML
    pod1_xml = os.path.join(TOPO_DIR, "pods", "POD#1.xml")
    shutil.copy2(test_case["xml"], pod1_xml)
    print(f"  Swapped POD#1 XML to {name}")

    # Restart backend
    proc = clear_cache()

    # Verify API
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://localhost:5001/api/topology/pod/POD%231", timeout=5)
        data = json.loads(resp.read())["data"]
        api_nodes = len(data["nodes"])
        api_edges = len(data["edges"])
        print(f"  API: {api_nodes} nodes, {api_edges} edges")

        if api_nodes != test_case["expected_nodes"]:
            print(f"  FAIL: Expected {test_case['expected_nodes']} nodes, got {api_nodes}")
        if api_edges != test_case["expected_edges"]:
            print(f"  WARN: Expected {test_case['expected_edges']} edges, got {api_edges}")
    except Exception as e:
        print(f"  FAIL: API error: {e}")

    # Playwright test
    screenshot_path = os.path.join(SCREENSHOT_DIR, f"pod_{name}.png")
    js = f"""
const {{ chromium }} = require('@playwright/test');
(async () => {{
  const browser = await chromium.launch();
  const ctx = await browser.newContext({{ viewport: {{ width: 1920, height: 1080 }} }});
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', msg => {{ if (msg.type() === 'error') errors.push(msg.text()); }});

  await page.goto('http://localhost:4173/pod/POD%231', {{ waitUntil: 'networkidle' }});
  await page.waitForTimeout(4000);

  const nodes = await page.locator('.react-flow__node').count();
  const edges = await page.locator('.react-flow__edge').count();
  const labels = await page.locator('.react-flow__edgelabel-renderer .tabular-nums').all();

  const labelData = [];
  for (let i = 0; i < labels.length; i++) {{
    const text = await labels[i].textContent();
    const box = await labels[i].boundingBox();
    labelData.push({{ text: text?.trim(), x: box?.x, y: box?.y, w: box?.width, h: box?.height }});
  }}

  // Check for overlaps
  const overlaps = [];
  for (let i = 0; i < labelData.length; i++) {{
    for (let j = i + 1; j < labelData.length; j++) {{
      const a = labelData[i], b = labelData[j];
      if (a.x != null && b.x != null) {{
        const overlapX = a.x < b.x + b.w && a.x + a.w > b.x;
        const overlapY = a.y < b.y + b.h && a.y + a.h > b.y;
        if (overlapX && overlapY) overlaps.push([i, j]);
      }}
    }}
  }}

  // Check node bounding boxes for label-node overlaps
  const nodeBoxes = [];
  const nodeEls = await page.locator('.react-flow__node').all();
  for (const n of nodeEls) {{
    const box = await n.boundingBox();
    if (box) nodeBoxes.push(box);
  }}

  const nodeOverlaps = [];
  for (let i = 0; i < labelData.length; i++) {{
    const l = labelData[i];
    if (l.x == null) continue;
    for (let j = 0; j < nodeBoxes.length; j++) {{
      const n = nodeBoxes[j];
      const overlapX = l.x < n.x + n.width && l.x + l.w > n.x;
      const overlapY = l.y < n.y + n.height && l.y + l.h > n.y;
      if (overlapX && overlapY) nodeOverlaps.push([i, j]);
    }}
  }}

  await page.screenshot({{ path: '{screenshot_path}', fullPage: true }});

  console.log(JSON.stringify({{
    nodes, edges, labelCount: labels.length, labels: labelData,
    overlaps, nodeOverlaps, errors
  }}));
  await browser.close();
}})();
"""
    stdout, stderr, rc = run_playwright_test(name, js)
    if rc != 0:
        print(f"  FAIL: Playwright error: {stderr[:500]}")
        return False

    try:
        result = json.loads(stdout.strip())
    except Exception:
        print(f"  FAIL: Could not parse Playwright output: {stdout[:500]}")
        return False

    print(f"  UI: {result['nodes']} nodes, {result['edges']} edges, {result['labelCount']} labels")

    ok = True
    if result["nodes"] != test_case["expected_nodes"]:
        print(f"  FAIL: Expected {test_case['expected_nodes']} nodes, got {result['nodes']}")
        ok = False

    if result["overlaps"]:
        print(f"  WARN: {len(result['overlaps'])} label-label overlaps")
        for pair in result["overlaps"][:5]:
            i, j = pair
            print(f"    Labels {i} & {j}: '{result['labels'][i]['text']}' vs '{result['labels'][j]['text']}'")

    if result["nodeOverlaps"]:
        print(f"  WARN: {len(result['nodeOverlaps'])} label-node overlaps")

    if result["errors"]:
        print(f"  WARN: Console errors: {result['errors'][:3]}")

    print(f"  Screenshot: {screenshot_path}")
    return ok


def main():
    results = {}

    # Test overview topologies
    for tc in OVERVIEW_TESTS:
        ok = test_overview(tc)
        results[f"overview_{tc['name']}"] = "PASS" if ok else "FAIL"

    # Test pod detail topologies
    for tc in POD_DETAIL_TESTS:
        ok = test_pod_detail(tc)
        results[f"pod_{tc['name']}"] = "PASS" if ok else "FAIL"

    # Restore originals
    print(f"\n{'='*60}")
    print("RESTORING ORIGINAL FILES")
    print(f"{'='*60}")
    shutil.copy2(OVERVIEW_XML + ".orig", OVERVIEW_XML)
    shutil.copy2(os.path.join(TOPO_DIR, "pods", "POD#1.xml.orig"),
                 os.path.join(TOPO_DIR, "pods", "POD#1.xml"))
    clear_cache()
    print("  Originals restored and backend restarted")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n  {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
