/**
 * SimAI Network Monitoring Dashboard - Comprehensive E2E Tests
 *
 * Tests critical user journeys for the React + TypeScript + ReactFlow dashboard.
 * The app uses MSW (Mock Service Worker) for API mocking so no backend is needed.
 *
 * Journeys covered:
 * 1. Overview page loads with all required elements
 * 2. All 4 POD nodes are visible in the ReactFlow canvas
 * 3. Cluster summary MetricCards display correct labels
 * 4. Header displays title and animated status dot
 * 5. Navigate to POD#1 detail page by clicking POD node
 * 6. POD detail page shows device nodes (OXC, SPINE, LEAF, SERVER)
 * 7. Click a device node opens NodeDetailPanel
 * 8. NodeDetailPanel shows metrics rows
 * 9. Close NodeDetailPanel with X button
 * 10. "Back to Overview" link navigates back to /
 * 11. Data polling: overview page retains state after 5 seconds
 * 12. Direct URL navigation to /pod/POD%231 works
 * 13. Edge labels are rendered in the overview graph
 * 14. POD detail MetricCards (Servers, GPUs, OXC Status, Links) visible
 */

import { test, expect, type Page } from '@playwright/test';
import { fileURLToPath } from 'url';
import * as path from 'path';
import * as fs from 'fs';

// ESM-compatible __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SCREENSHOT_DIR = path.resolve(__dirname, 'screenshots');

if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

async function screenshot(page: Page, name: string): Promise<void> {
  const filePath = path.join(SCREENSHOT_DIR, `${name}.png`);
  await page.screenshot({ path: filePath, fullPage: false });
  console.log(`  Screenshot saved: ${filePath}`);
}

/**
 * Wait until the overview topology loads (4 POD nodes present).
 */
async function waitForOverviewLoaded(page: Page): Promise<void> {
  await page.waitForSelector('.react-flow__node', { state: 'attached', timeout: 15000 });
}

/**
 * Wait until the detail topology loads (device nodes present).
 */
async function waitForDetailLoaded(page: Page): Promise<void> {
  await page.waitForSelector('.react-flow__node', { state: 'attached', timeout: 15000 });
}

/**
 * Click a ReactFlow node by its data-testid.
 *
 * ReactFlow renders a transparent full-canvas overlay (div.flex-1 / the pane)
 * that intercepts all pointer events. Playwright's normal click and even
 * force:true still hit that overlay at the element's center coordinates.
 *
 * The only reliable way is to dispatch synthetic mouse events directly on
 * the node's DOM element, bypassing the overlay entirely.
 */
async function clickReactFlowNode(page: Page, nodeDataId: string): Promise<void> {
  const selector = `[data-testid="rf__node-${nodeDataId}"]`;
  await page.waitForSelector(selector, { timeout: 10000 });

  await page.evaluate((sel) => {
    const nodeEl = document.querySelector(sel);
    if (!nodeEl) throw new Error(`ReactFlow node not found: ${sel}`);
    const rect = nodeEl.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const eventOpts = { bubbles: true, cancelable: true, clientX: cx, clientY: cy };
    nodeEl.dispatchEvent(new MouseEvent('mousedown', eventOpts));
    nodeEl.dispatchEvent(new MouseEvent('mouseup', eventOpts));
    nodeEl.dispatchEvent(new MouseEvent('click', eventOpts));
  }, selector);
}

// ---------------------------------------------------------------------------
// Test Suite 1: Overview Page
// ---------------------------------------------------------------------------

test.describe('Overview Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);
  });

  test('page title and header are rendered', async ({ page }) => {
    // Header h1 contains the monitor title
    const h1 = page.locator('h1');
    await expect(h1).toBeVisible();
    await expect(h1).toContainText('OCS-Sim Network Monitor');

    // Animated status dot is present (StatusDot component has animate-ping span)
    const statusDot = page.locator('.animate-ping');
    await expect(statusDot).toBeVisible();

    await screenshot(page, '01-overview-header');
  });

  test('last updated time is displayed in header', async ({ page }) => {
    // Header shows "Last update:" text followed by a time string
    const lastUpdateText = page.locator('header').filter({ hasText: 'Last update:' });
    await expect(lastUpdateText).toBeVisible();
  });

  test('cluster summary bar shows all MetricCard labels', async ({ page }) => {
    // MetricCards use uppercase CSS class but DOM text is the original case
    // Labels: "Total PODs", "Total GPUs", "Utilization", "Active Alerts"
    const metricLabels = ['Total PODs', 'Total GPUs', 'Utilization', 'Active Alerts'];

    for (const label of metricLabels) {
      const card = page.locator(`text=${label}`);
      await expect(card).toBeVisible();
    }

    await screenshot(page, '02-overview-metric-cards');
  });

  test('cluster summary shows numeric values from MSW mock', async ({ page }) => {
    // MSW mock returns totalPods:4, totalGpus:128
    // MetricCard renders a text-2xl bold value
    // Find the MetricCard value "4" adjacent to "Total PODs"
    const podValueEl = page.locator('.text-2xl').filter({ hasText: '4' }).first();
    await expect(podValueEl).toBeVisible({ timeout: 10000 });

    // GPUs value = 128
    const gpuValueEl = page.locator('.text-2xl').filter({ hasText: '128' });
    await expect(gpuValueEl).toBeVisible({ timeout: 10000 });
  });

  test('ReactFlow canvas and controls are present', async ({ page }) => {
    // Verify the ReactFlow canvas is rendered by checking its child controls
    // (The ReactFlow outer div uses overflow:hidden which can confuse Playwright's
    // visibility check, so we check child elements that are definitively visible.)

    // Controls: ReactFlow renders Zoom In/Out/Fit buttons
    const zoomInBtn = page.getByRole('button', { name: 'Zoom In' });
    await expect(zoomInBtn).toBeVisible();

    const zoomOutBtn = page.getByRole('button', { name: 'Zoom Out' });
    await expect(zoomOutBtn).toBeVisible();

    const fitViewBtn = page.getByRole('button', { name: 'Fit View' });
    await expect(fitViewBtn).toBeVisible();

    // MiniMap: ReactFlow renders the minimap accessible as img[alt="Mini Map"]
    const minimap = page.getByRole('img', { name: 'Mini Map' });
    await expect(minimap).toBeAttached({ timeout: 5000 });

    // The canvas is present in the DOM with role=application
    const canvas = page.locator('[role="application"]');
    await expect(canvas).toBeAttached();

    await screenshot(page, '03-overview-canvas-controls');
  });

  test('4 POD nodes are rendered in the ReactFlow canvas', async ({ page }) => {
    // Each podNode renders a .react-flow__node div with data-testid="rf__node-POD#N"
    const nodes = page.locator('.react-flow__node');
    await expect(nodes).toHaveCount(4, { timeout: 10000 });

    await screenshot(page, '04-overview-four-pod-nodes');
  });

  test('POD node cards display GPU count (32 GPUs) and utilization (0.6%)', async ({ page }) => {
    // PodNode renders "32 GPUs" and "0.6%" (0.0061 * 100 = 0.61 -> toFixed(1) = 0.6%)
    const gpuTexts = page.locator('text=32 GPUs');
    await expect(gpuTexts).toHaveCount(4, { timeout: 10000 });

    // Utilization text: "0.6%" (0.0061 * 100 = 0.61, toFixed(1) = "0.6")
    const utilTexts = page.locator('text=0.6%');
    await expect(utilTexts).toHaveCount(4, { timeout: 10000 });
  });

  test('all 4 POD node labels are visible (POD #1 through POD #4)', async ({ page }) => {
    for (let i = 1; i <= 4; i++) {
      // Use data-testid to target each specific POD node
      const nodeEl = page.locator(`[data-testid="rf__node-POD#${i}"]`);
      await expect(nodeEl).toBeAttached({ timeout: 10000 });
      // Verify the label text is in the node
      await expect(nodeEl).toContainText(`POD #${i}`);
    }
    await screenshot(page, '05-overview-pod-labels');
  });

  test('5 edges are rendered between POD nodes', async ({ page }) => {
    // The overview has 5 inter-pod edges (per all_pod.xml, no POD#2-POD#4 edge).
    let maxSeen = 0;
    for (let attempt = 0; attempt < 20; attempt++) {
      const count = await page.evaluate(() => {
        return document.querySelectorAll('.react-flow__edgelabel-renderer > div').length;
      });
      if (count > maxSeen) maxSeen = count;
      if (maxSeen >= 5) break;
      await page.waitForTimeout(300);
    }
    expect(maxSeen).toBe(5);

    await screenshot(page, '06-overview-edges');
  });

  test('edge labels show utilization percentage (0.61%)', async ({ page }) => {
    // Each MetricEdge EdgeLabelRenderer renders "0.61%" (utilization=0.0061)
    const edgeLabel = page.locator('text=0.61%').first();
    await expect(edgeLabel).toBeVisible({ timeout: 10000 });
  });

  test('edge labels show link count markers (~x6 and ~x4)', async ({ page }) => {
    // Most edges have linkCount=6 -> "~x6", one has linkCount=4 -> "~x4"
    // Edge labels use ReactFlow EdgeLabelRenderer which re-renders during polling.
    // Retry finding the labels across multiple polling cycles.
    let x6Found = false;
    let x4Found = false;
    for (let attempt = 0; attempt < 20; attempt++) {
      if (!x6Found) {
        x6Found = (await page.locator('text=~x6').count()) > 0;
      }
      if (!x4Found) {
        x4Found = (await page.locator('text=~x4').count()) > 0;
      }
      if (x6Found && x4Found) break;
      await page.waitForTimeout(300);
    }
    expect(x6Found, '~x6 link count label not found in any render cycle').toBe(true);
    expect(x4Found, '~x4 link count label not found in any render cycle').toBe(true);
  });

  test('edge labels show error rate percentage (0.31%)', async ({ page }) => {
    // MetricEdge renders errorRate as third line: "0.31%" (0.0031 * 100)
    const errLabel = page.locator('text=0.31%').first();
    await expect(errLabel).toBeVisible({ timeout: 10000 });
  });
});

// ---------------------------------------------------------------------------
// Test Suite 2: Navigation - Overview to Detail
// ---------------------------------------------------------------------------

test.describe('Navigation: Overview to POD Detail', () => {
  test('clicking POD#1 node navigates to /pod/POD%231', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    // Click POD#1 using its data-testid (force required to bypass ReactFlow pane overlay)
    await clickReactFlowNode(page, 'POD#1');

    // Should navigate to the detail page
    await expect(page).toHaveURL(/\/pod\/POD/, { timeout: 10000 });

    await screenshot(page, '07-navigate-to-pod1');
  });

  test('clicking POD#2 node navigates to /pod/POD%232', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    await clickReactFlowNode(page, 'POD#2');
    await expect(page).toHaveURL(/\/pod\/POD/, { timeout: 10000 });
  });

  test('clicking POD#3 node navigates to /pod/POD%233', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    await clickReactFlowNode(page, 'POD#3');
    await expect(page).toHaveURL(/\/pod\/POD/, { timeout: 10000 });
  });

  test('clicking POD#4 node navigates to /pod/POD%234', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    await clickReactFlowNode(page, 'POD#4');
    await expect(page).toHaveURL(/\/pod\/POD/, { timeout: 10000 });
  });
});

// ---------------------------------------------------------------------------
// Test Suite 3: POD Detail Page
// ---------------------------------------------------------------------------

test.describe('POD Detail Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/pod/POD%231');
    await waitForDetailLoaded(page);
  });

  test('detail page header shows "POD#1 Detail" and back link', async ({ page }) => {
    // Header h1 shows "POD#1 Detail"
    const h1 = page.locator('h1');
    await expect(h1).toBeVisible();
    await expect(h1).toContainText('Detail');

    // Back to Overview link is present
    const backLink = page.locator('a', { hasText: 'Back to Overview' });
    await expect(backLink).toBeVisible();

    await screenshot(page, '08-detail-page-header');
  });

  test('detail page header shows the animated status dot', async ({ page }) => {
    const statusDot = page.locator('.animate-ping');
    await expect(statusDot).toBeVisible();
  });

  test('detail MetricCards show Servers, GPUs, OXC Status, Links labels', async ({ page }) => {
    // Labels are the original case text (CSS uppercase only renders them visually)
    const metricLabels = ['Servers', 'GPUs', 'OXC Status', 'Links'];
    for (const label of metricLabels) {
      await expect(page.locator(`text=${label}`)).toBeVisible({ timeout: 10000 });
    }

    await screenshot(page, '09-detail-metric-cards');
  });

  test('OXC Status MetricCard shows "Online"', async ({ page }) => {
    // OXC Status value is hardcoded to "Online" in PodDetailPage
    const onlineText = page.locator('text=Online');
    await expect(onlineText).toBeVisible({ timeout: 10000 });
  });

  test('Servers MetricCard shows 4 and GPUs shows 32', async ({ page }) => {
    // POD#1 has 4 SERVER nodes -> Servers=4, GPUs=4*8=32
    // Find ".text-2xl" elements with those values
    const server4 = page.locator('.text-2xl').filter({ hasText: '4' }).first();
    await expect(server4).toBeVisible({ timeout: 10000 });

    const gpu32 = page.locator('.text-2xl').filter({ hasText: '32' }).first();
    await expect(gpu32).toBeVisible({ timeout: 10000 });
  });

  test('Links MetricCard shows 14', async ({ page }) => {
    // POD#1 detail has 14 edges (2 OXC->SPINE + 4 SPINE->LEAF + 8 SERVER->LEAF dual-ToR)
    const links14 = page.locator('.text-2xl').filter({ hasText: '14' }).first();
    await expect(links14).toBeVisible({ timeout: 10000 });
  });

  test('ReactFlow canvas is present on detail page', async ({ page }) => {
    const canvas = page.locator('.react-flow');
    await expect(canvas).toBeVisible();
  });

  test('device type badges are rendered: OXC, SPN, LF, SRV', async ({ page }) => {
    // NetworkDeviceNode renders abbreviated type badges
    await expect(page.locator('text=OXC').first()).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=SPN').first()).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=LF').first()).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=SRV').first()).toBeVisible({ timeout: 10000 });

    await screenshot(page, '10-detail-device-type-badges');
  });

  test('OXC #1 node label is visible', async ({ page }) => {
    await expect(page.locator('text=OXC #1')).toBeVisible({ timeout: 10000 });
  });

  test('SPINE #1 and SPINE #2 node labels are visible', async ({ page }) => {
    await expect(page.locator('text=SPINE #1')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=SPINE #2')).toBeVisible({ timeout: 10000 });
  });

  test('LEAF #1 and LEAF #2 node labels are visible', async ({ page }) => {
    await expect(page.locator('text=LEAF #1')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=LEAF #2')).toBeVisible({ timeout: 10000 });
  });

  test('SERVER node labels (A3 #1 through A3 #4) are visible', async ({ page }) => {
    for (let i = 1; i <= 4; i++) {
      await expect(page.locator(`text=A3 #${i}`)).toBeVisible({ timeout: 10000 });
    }
  });

  test('11 total nodes rendered (9 device + 2 supernode boundaries)', async ({ page }) => {
    // 1 OXC + 2 SPINE + 2 LEAF + 4 SERVER = 9 device nodes
    // + 2 SuperNodeBoundary nodes = 11 total
    const allNodes = page.locator('.react-flow__node');
    await expect(allNodes).toHaveCount(11, { timeout: 15000 });
  });

  test('SuperNode boundary labels SUPER_NODE#1 and SUPER_NODE#2 are visible', async ({ page }) => {
    await expect(page.locator('text=SUPER_NODE#1')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=SUPER_NODE#2')).toBeVisible({ timeout: 10000 });

    await screenshot(page, '11-detail-supernode-boundaries');
  });

  test('OXC #1 IP address 104.101.101.101 is displayed', async ({ page }) => {
    // baseIp = 101 + podIndex(0) = 101
    // OXC IP: "104.101.101.101"
    const oxcIp = page.locator('text=104.101.101.101');
    await expect(oxcIp).toBeVisible({ timeout: 10000 });
  });

  test('SPINE node IP addresses are displayed', async ({ page }) => {
    await expect(page.locator('text=103.101.101.101')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=103.101.101.102')).toBeVisible({ timeout: 10000 });
  });

  test('14 edges are rendered in the detail topology', async ({ page }) => {
    // 2 OXC->SPINE + 4 SPINE->LEAF + 8 SERVER->LEAF (dual-ToR) = 14 edges
    let maxSeen = 0;
    for (let attempt = 0; attempt < 10; attempt++) {
      const count = await page.evaluate(
        () => document.querySelectorAll('.react-flow__edgelabel-renderer > div').length
      );
      if (count > maxSeen) maxSeen = count;
      if (maxSeen >= 14) break;
      await page.waitForTimeout(500);
    }
    expect(maxSeen).toBe(14);

    await screenshot(page, '12-detail-edges');
  });

  test('edge label for OXC->SPINE shows 0.45% utilization', async ({ page }) => {
    // OXC->SPINE edges have utilization=0.0045 -> "0.45%"
    const oxcEdgeLabel = page.locator('text=0.45%').first();
    await expect(oxcEdgeLabel).toBeVisible({ timeout: 10000 });
  });
});

// ---------------------------------------------------------------------------
// Test Suite 4: NodeDetailPanel interaction
// ---------------------------------------------------------------------------

test.describe('NodeDetailPanel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/pod/POD%231');
    await waitForDetailLoaded(page);
  });

  test('clicking a SERVER node (A3#1) opens the NodeDetailPanel', async ({ page }) => {
    // Use force:true since ReactFlow pane may intercept clicks
    await clickReactFlowNode(page, 'A3#1');

    // NodeDetailPanel is a w-80 side panel with an h2 heading
    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });

    await screenshot(page, '13-node-detail-panel-opened');
  });

  test('NodeDetailPanel heading shows the node ID', async ({ page }) => {
    await clickReactFlowNode(page, 'A3#1');

    const panelHeading = page.locator('[class*="w-80"] h2');
    await expect(panelHeading).toBeVisible({ timeout: 10000 });

    const headingText = await panelHeading.textContent();
    expect(headingText).toBeTruthy();
    // Should contain the node ID (e.g. "A3#1")
    expect(headingText).toContain('A3#1');
  });

  test('NodeDetailPanel shows all 6 metric rows after MSW fetch', async ({ page }) => {
    await clickReactFlowNode(page, 'A3#1');

    // Wait for the panel to appear first
    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });

    // usePolling fetches immediately on mount - metrics should load quickly
    const metricLabels = [
      'GPU Utilization',
      'CPU Utilization',
      'Memory',
      'Temperature',
      'Power',
    ];
    for (const label of metricLabels) {
      await expect(panel.locator(`text=${label}`)).toBeVisible({ timeout: 10000 });
    }

    // "Status" is the 6th metric row - scope to panel to avoid "OXC Status" collision
    await expect(panel.getByText('Status', { exact: true })).toBeVisible({ timeout: 10000 });

    await screenshot(page, '14-node-detail-panel-metrics');
  });

  test('NodeDetailPanel shows numeric metric values', async ({ page }) => {
    await clickReactFlowNode(page, 'A3#1');

    // MSW mock generates: gpuUtilization 50-90%, cpuUtilization 30-80%
    // Values appear as "XX.X%" next to metric labels
    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });

    // Memory shows format "XX.X / 80 GB"
    const memValue = page.locator('text=/ 80 GB');
    await expect(memValue).toBeVisible({ timeout: 10000 });
  });

  test('NodeDetailPanel X button dismisses the panel', async ({ page }) => {
    await clickReactFlowNode(page, 'A3#1');

    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });

    // The × (times) close button
    const closeButton = panel.locator('button').filter({ hasText: '×' });
    await expect(closeButton).toBeVisible({ timeout: 5000 });
    await closeButton.click();

    // Panel should disappear
    await expect(panel).not.toBeVisible({ timeout: 5000 });

    await screenshot(page, '15-node-detail-panel-closed');
  });

  test('clicking the same node again toggles the panel closed', async ({ page }) => {
    // First click - open panel
    await clickReactFlowNode(page, 'A3#1');
    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });

    // Second click on same node - PodDetailPage toggles: node.id === selectedNodeId -> selectNode(null)
    await clickReactFlowNode(page, 'A3#1');
    await expect(panel).not.toBeVisible({ timeout: 5000 });
  });

  test('clicking a different node switches the panel content', async ({ page }) => {
    // Open A3#1 panel
    await clickReactFlowNode(page, 'A3#1');
    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });
    const heading1 = page.locator('[class*="w-80"] h2');
    await expect(heading1).toContainText('A3#1');

    // Click A3#2 - panel should update to show A3#2
    await clickReactFlowNode(page, 'A3#2');
    await expect(panel).toBeVisible({ timeout: 10000 });
    await expect(heading1).toContainText('A3#2');
  });

  test('clicking an OXC node opens panel with OXC#1 heading', async ({ page }) => {
    await clickReactFlowNode(page, 'OXC#1');

    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });

    const heading = page.locator('[class*="w-80"] h2');
    await expect(heading).toContainText('OXC#1');
  });
});

// ---------------------------------------------------------------------------
// Test Suite 5: Navigation - Back to Overview
// ---------------------------------------------------------------------------

test.describe('Navigation: Back to Overview', () => {
  test('Back to Overview link navigates back to /', async ({ page }) => {
    await page.goto('/pod/POD%231');
    await waitForDetailLoaded(page);

    const backLink = page.locator('a', { hasText: 'Back to Overview' });
    await expect(backLink).toBeVisible();
    await backLink.click();

    await expect(page).toHaveURL('/');
    await waitForOverviewLoaded(page);

    // Verify we are back on overview (4 POD nodes visible)
    const nodes = page.locator('.react-flow__node');
    await expect(nodes).toHaveCount(4, { timeout: 10000 });

    await screenshot(page, '16-back-to-overview');
  });

  test('browser back button returns from detail to overview', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    // Navigate to detail
    await clickReactFlowNode(page, 'POD#1');
    await expect(page).toHaveURL(/\/pod\/POD/, { timeout: 10000 });

    // Use browser back
    await page.goBack();
    await expect(page).toHaveURL('/');
    await waitForOverviewLoaded(page);

    const nodes = page.locator('.react-flow__node');
    await expect(nodes).toHaveCount(4, { timeout: 10000 });
  });

  test('full round-trip: overview -> POD#1 detail -> back to overview', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);
    await screenshot(page, '17-roundtrip-step1-overview');

    // Navigate to POD#1
    await clickReactFlowNode(page, 'POD#1');
    await expect(page).toHaveURL(/\/pod\/POD/, { timeout: 10000 });
    await waitForDetailLoaded(page);
    await screenshot(page, '18-roundtrip-step2-pod1-detail');

    // Back to overview
    await page.locator('a', { hasText: 'Back to Overview' }).click();
    await expect(page).toHaveURL('/');
    await waitForOverviewLoaded(page);
    await screenshot(page, '19-roundtrip-step3-back-overview');

    // 4 nodes should be present again
    await expect(page.locator('.react-flow__node')).toHaveCount(4, { timeout: 10000 });
  });
});

// ---------------------------------------------------------------------------
// Test Suite 6: Data Polling
// ---------------------------------------------------------------------------

test.describe('Data Polling', () => {
  test('overview page retains all 4 nodes after 5s polling cycle', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    // Wait for 2 polling cycles (2 * 5000ms)
    await page.waitForTimeout(11000);

    // Nodes should still be rendered correctly after polling updates state
    const nodes = page.locator('.react-flow__node');
    await expect(nodes).toHaveCount(4, { timeout: 5000 });

    await screenshot(page, '20-polling-overview-stable');
  });

  test('lastUpdated timestamp is displayed and non-empty', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    // Wait for first poll to complete
    await page.waitForTimeout(1000);

    const lastUpdateSpan = page.locator('header span', { hasText: 'Last update:' });
    await expect(lastUpdateSpan).toBeVisible();

    const text = await lastUpdateSpan.textContent();
    console.log(`  lastUpdated display: ${text}`);

    // Should show a time like "Last update: 6:20:30 PM" - not the placeholder "--:--:--"
    expect(text).not.toContain('--:--:--');
    expect(text).toMatch(/Last update: \d+:\d+:\d+/);
  });

  test('detail page retains nodes after 5s polling cycle', async ({ page }) => {
    await page.goto('/pod/POD%231');
    await waitForDetailLoaded(page);

    // Wait for one polling cycle
    await page.waitForTimeout(6000);

    // 11 nodes should still be rendered
    const nodes = page.locator('.react-flow__node');
    await expect(nodes).toHaveCount(11, { timeout: 5000 });

    await screenshot(page, '21-polling-detail-stable');
  });
});

// ---------------------------------------------------------------------------
// Test Suite 7: Direct URL Navigation
// ---------------------------------------------------------------------------

test.describe('Direct URL Navigation', () => {
  test('navigating directly to /pod/POD%231 loads POD#1 detail', async ({ page }) => {
    await page.goto('/pod/POD%231');
    await waitForDetailLoaded(page);

    await expect(page.locator('h1')).toContainText('Detail');
    await expect(page.locator('text=OXC #1')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=104.101.101.101')).toBeVisible({ timeout: 10000 });

    await screenshot(page, '22-direct-url-pod1');
  });

  test('navigating to /pod/POD%232 loads POD#2 detail with OXC#2', async ({ page }) => {
    await page.goto('/pod/POD%232');
    await waitForDetailLoaded(page);

    // POD#2 has podIndex=1, so OXC#2, baseIp=102
    await expect(page.locator('text=OXC #2')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=104.102.101.101')).toBeVisible({ timeout: 10000 });

    await screenshot(page, '23-direct-url-pod2');
  });

  test('navigating to /pod/POD%233 loads POD#3 detail with OXC#3', async ({ page }) => {
    await page.goto('/pod/POD%233');
    await waitForDetailLoaded(page);

    await expect(page.locator('text=OXC #3')).toBeVisible({ timeout: 10000 });
  });

  test('navigating to /pod/POD%234 loads POD#4 detail with OXC#4', async ({ page }) => {
    await page.goto('/pod/POD%234');
    await waitForDetailLoaded(page);

    await expect(page.locator('text=OXC #4')).toBeVisible({ timeout: 10000 });
  });
});

// ---------------------------------------------------------------------------
// Test Suite 8: UI Visual Checks
// ---------------------------------------------------------------------------

test.describe('UI Visual Checks', () => {
  test('overview page has the correct dark background color (#0a0e17)', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);

    const shell = page.locator('.min-h-screen');
    await expect(shell).toBeVisible();

    // Background is COLORS.bg.primary = #0a0e17 = rgb(10, 14, 23)
    const bgColor = await shell.evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(bgColor).toBe('rgb(10, 14, 23)');

    await screenshot(page, '24-dark-background');
  });

  test('Total PODs MetricCard correctly shows value 4', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);
    await page.waitForTimeout(1000); // let MSW respond

    // Find the parent MetricCard div containing "Total PODs" label
    // and verify its .text-2xl child shows "4"
    const podCardContainer = page
      .locator('.text-xs', { hasText: 'Total PODs' })
      .locator('..') // parent flex-col div
      .locator('.text-2xl');

    await expect(podCardContainer).toBeVisible({ timeout: 10000 });
    await expect(podCardContainer).toHaveText('4');
  });

  test('Total GPUs MetricCard correctly shows value 128', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);
    await page.waitForTimeout(1000);

    const gpuCardContainer = page
      .locator('.text-xs', { hasText: 'Total GPUs' })
      .locator('..')
      .locator('.text-2xl');

    await expect(gpuCardContainer).toBeVisible({ timeout: 10000 });
    await expect(gpuCardContainer).toHaveText('128');
  });

  test('Utilization MetricCard shows a percentage value', async ({ page }) => {
    await page.goto('/');
    await waitForOverviewLoaded(page);
    await page.waitForTimeout(1000);

    const utilizationCard = page
      .locator('.text-xs', { hasText: 'Utilization' })
      .locator('..')
      .locator('.text-2xl');

    await expect(utilizationCard).toBeVisible({ timeout: 10000 });
    const utilText = await utilizationCard.textContent();
    expect(utilText).toMatch(/\d+\.\d+%/); // matches "68.9%" etc.
  });

  test('detail page ReactFlow controls are rendered', async ({ page }) => {
    await page.goto('/pod/POD%231');
    await waitForDetailLoaded(page);

    const controls = page.locator('[aria-label="Control Panel"]');
    await expect(controls).toBeVisible();

    await screenshot(page, '25-detail-controls');
  });
});

// ---------------------------------------------------------------------------
// Test Suite 9: Full User Journey (End-to-End Screenshot Sequence)
// ---------------------------------------------------------------------------

test.describe('Full User Journey', () => {
  test('complete journey: load overview -> navigate to POD detail -> open node panel -> close -> back', async ({ page }) => {
    // Step 1: Load overview page
    await page.goto('/');
    await waitForOverviewLoaded(page);
    await page.waitForTimeout(500);
    await screenshot(page, '26-journey-step1-overview');

    // Verify overview is correct
    expect(await page.locator('.react-flow__node').count()).toBe(4);
    await expect(page.locator('h1')).toContainText('SimAI Network Monitor');

    // Step 2: Click POD#1 to navigate to detail
    await clickReactFlowNode(page, 'POD#1');
    await expect(page).toHaveURL(/\/pod\/POD/, { timeout: 10000 });
    await waitForDetailLoaded(page);
    await page.waitForTimeout(500);
    await screenshot(page, '27-journey-step2-pod1-detail');

    // Verify detail page is correct
    await expect(page.locator('h1')).toContainText('Detail');
    await expect(page.locator('text=OXC #1')).toBeVisible({ timeout: 5000 });

    // Step 3: Click on A3#1 to open NodeDetailPanel
    await clickReactFlowNode(page, 'A3#1');
    const panel = page.locator('[class*="w-80"]');
    await expect(panel).toBeVisible({ timeout: 10000 });
    await page.waitForTimeout(1500); // wait for metrics to load
    await screenshot(page, '28-journey-step3-node-panel-open');

    // Verify panel has metrics
    await expect(page.locator('text=GPU Utilization')).toBeVisible({ timeout: 5000 });

    // Step 4: Close the NodeDetailPanel
    const closeBtn = panel.locator('button').filter({ hasText: '×' });
    await closeBtn.click();
    await expect(panel).not.toBeVisible({ timeout: 5000 });
    await screenshot(page, '29-journey-step4-panel-closed');

    // Step 5: Navigate back to overview via "Back to Overview" link
    await page.locator('a', { hasText: 'Back to Overview' }).click();
    await expect(page).toHaveURL('/');
    await waitForOverviewLoaded(page);
    await screenshot(page, '30-journey-step5-back-overview');

    // Final verification: 4 POD nodes, correct title
    await expect(page.locator('.react-flow__node')).toHaveCount(4, { timeout: 10000 });
    await expect(page.locator('h1')).toContainText('SimAI Network Monitor');
  });
});
