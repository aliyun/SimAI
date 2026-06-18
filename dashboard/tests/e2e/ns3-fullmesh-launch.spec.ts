import { test, expect } from '@playwright/test';

const BASE_API = 'http://localhost:5001';

interface SessionInfo {
  readonly token: string;
  readonly username: string;
}

async function login(): Promise<SessionInfo> {
  const res = await fetch(`${BASE_API}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'e2etest' }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`Login failed: ${JSON.stringify(data)}`);
  return { token: data.token, username: data.username };
}

async function apiPost(token: string, path: string, body: Record<string, unknown>) {
  const res = await fetch(`${BASE_API}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Session-Token': token,
    },
    body: JSON.stringify(body),
  });
  return { status: res.status, data: await res.json() };
}

function generateFullMeshTopology(): string {
  const npus = 8;
  const leaf = npus; // node 8
  const totalNodes = npus + 1; // 9
  const npuPerServer = 8;
  const nvSwitchNum = 0;
  const switchNodes = 1;

  const links: string[] = [];
  // Full-mesh NPU↔NPU intra links at 2400Gbps
  for (let i = 0; i < npus; i++) {
    for (let j = i + 1; j < npus; j++) {
      links.push(`${i} ${j} 2400Gbps 0.000025ms 0`);
    }
  }
  // NPU→Leaf uplinks at 100Gbps
  for (let i = 0; i < npus; i++) {
    links.push(`${i} ${leaf} 100Gbps 0.0005ms 0`);
  }

  const header = `${totalNodes} ${npuPerServer} ${nvSwitchNum} ${switchNodes} ${links.length} A100`;
  return [header, `${leaf} `, ...links].join('\n');
}

function generateMinimalWorkload(): string {
  const header = 'HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: 8 checkpoints: 0 checkpoint_initiates: 0';
  const layers = [
    'layer_0 -1 1000 ALLREDUCE 1024 1000 NONE 0 1000 NONE 0 0',
    'layer_1 -1 1000 ALLREDUCE 1024 1000 NONE 0 1000 NONE 0 0',
  ];
  return [header, String(layers.length), ...layers].join('\n');
}

function generateRanktable(): object {
  const ranktable = {
    version: '2.0',
    status: 'completed',
    rank_count: 8,
    rank_list: [] as object[],
  };
  const eidBase = BigInt('0x000000000000002000100000DF001001');
  for (let i = 0; i < 8; i++) {
    const eidStr = (eidBase + BigInt(i)).toString(16).padStart(32, '0');
    ranktable.rank_list.push({
      rank_id: i,
      device_id: i,
      local_id: i,
      level_list: [{
        net_layer: 0,
        net_instance_id: 'rack_0',
        net_type: 'TOPO_FILE_DESC',
        net_attr: '',
        rank_addr_list: [{
          addr_type: 'EID',
          addr: eidStr,
          ports: ['0/0'],
          plane_id: 'plane0',
        }],
      }],
    });
  }
  return ranktable;
}

function getNetworkStorage() {
  const network = {
    id: 'net_fm8g',
    name: 'FullMesh-8G',
    topologyDir: 'topo_8g.txt',
    npuPerServer: 8,
    npuType: 'A100',
    intraBw: '2400Gbps',
    bandwidth: '100Gbps',
    serverIps: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  return {
    state: {
      networks: [network],
      activeNetworkId: network.id,
    },
    version: 0,
  };
}

test.describe('NS3 full-mesh end-to-end launch', () => {
  test.setTimeout(600_000); // 10 minutes

  test('launch 2-layer NS3 workload via dashboard and verify completion', async ({ page }) => {
    // ---------- API setup ----------
    const session = await login();

    // Save topology
    const topoRes = await apiPost(session.token, '/api/files/save', {
      filename: 'topo_8g.txt',
      content: generateFullMeshTopology(),
    });
    expect(topoRes.status).toBe(200);

    // Save workload
    const wlRes = await apiPost(session.token, '/api/files/save', {
      filename: 'workload.txt',
      content: generateMinimalWorkload(),
    });
    expect(wlRes.status).toBe(200);

    // Save ranktable
    const rtRes = await apiPost(session.token, '/api/files/save', {
      filename: 'ranktable.json',
      content: JSON.stringify(generateRanktable(), null, 2),
    });
    expect(rtRes.status).toBe(200);

    // ---------- Frontend ----------
    await page.goto('http://localhost:3000/');
    await page.waitForLoadState('networkidle');

    // Inject auth token and network into localStorage
    const storageValue = JSON.stringify(getNetworkStorage());
    await page.evaluate(({ token, networks }) => {
      localStorage.setItem('ocs_sim_token', token);
      localStorage.setItem('ocs-sim-networks', networks);
    }, { token: session.token, networks: storageValue });

    // Reload so zustand and api client pick up the injected state
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Navigate to launch page
    await page.goto('http://localhost:3000/deploy/launch');
    await page.waitForLoadState('networkidle');

    // Verify network is auto-selected and trigger onChange so topologyPath is set
    const networkSelect = page.locator('select').first();
    const selectedOption = await networkSelect.inputValue();
    expect(selectedOption).toBe('net_fm8g');
    // Re-select to trigger handleNetworkChange and populate launchConfig.topologyPath
    await networkSelect.selectOption('net_fm8g');
    await page.waitForTimeout(500);

    // Debug: log console messages and network requests
    const consoleLogs: string[] = [];
    page.on('console', (msg) => {
      consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
    });
    const requests: string[] = [];
    page.on('request', (req) => {
      if (req.method() === 'POST') requests.push(`${req.method()} ${req.url()}`);
    });
    page.on('response', (resp) => {
      if (resp.request().method() === 'POST') requests.push(`RESP ${resp.status()} ${resp.url()}`);
    });

    // Select NS3 Simulation mode
    const ns3Button = page.getByRole('button', { name: /NS3 Simulation/ });
    await expect(ns3Button).toBeVisible({ timeout: 5000 });
    await ns3Button.click();

    // Verify NS3 mode is selected (button has active border class)
    await expect(ns3Button).toHaveClass(/border-\[var\(--color-accent-blue\)\]/);

    // Click launch
    const launchButton = page.getByRole('button', { name: /启动仿真/ });
    await expect(launchButton).toBeVisible({ timeout: 5000 });
    await expect(launchButton).toBeEnabled({ timeout: 5000 });

    // Intercept the launch response to capture error details
    const launchPromise = page.waitForResponse(
      (resp) => resp.url().includes('/api/process/launch'),
      { timeout: 15000 }
    );
    await launchButton.click();
    const launchResp = await launchPromise;
    const launchData = await launchResp.json();
    console.log('Launch response status:', launchResp.status());
    console.log('Launch response body:', JSON.stringify(launchData));

    // Wait a moment then query API for the most recent running process
    await page.waitForTimeout(3000);

    // Take debug screenshot
    await page.screenshot({ path: 'test-results/ns3-fullmesh-after-launch.png', fullPage: true });

    console.log('Console logs after launch:', consoleLogs.join('\n'));
    console.log('Network requests after launch:', requests.join('\n'));

    const listRes = await fetch(`${BASE_API}/api/process/list`, {
      headers: { 'X-Session-Token': session.token },
    });
    expect(listRes.ok).toBe(true);
    const listData = await listRes.json();
    expect(Array.isArray(listData.processes)).toBe(true);
    console.log('Processes after launch:', JSON.stringify(listData.processes.slice(0, 5), null, 2));

    // Find the most recent process that uses SimAI_simulator (any status)
    const simProc = listData.processes
      .filter((p: any) => p.command && p.command.includes('SimAI_simulator'))
      .sort((a: any, b: any) => b.pid - a.pid)[0];
    expect(simProc).toBeTruthy();
    const pid = simProc.pid as number;

    // Wait for the specific PID to appear in the UI
    await expect(page.locator(`text=PID ${pid}`)).toBeVisible({ timeout: 15000 });

    // Poll for completion via API (up to 9 minutes)
    const maxWaitMs = 9 * 60 * 1000;
    const pollInterval = 5000;
    const startTime = Date.now();
    let completed = false;

    while (Date.now() - startTime < maxWaitMs) {
      const statusRes = await fetch(`${BASE_API}/api/process/logs/${pid}`, {
        headers: { 'X-Session-Token': session.token },
      });
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        console.log(`PID ${pid} status: ${statusData.status}`);
        if (statusData.status === 'finished' || statusData.status === 'exited') {
          completed = true;
          break;
        }
        if (statusData.status === 'error' || statusData.status === 'dead') {
          throw new Error(`Simulation failed with status: ${statusData.status}`);
        }
      }
      await page.waitForTimeout(pollInterval);
    }

    expect(completed).toBe(true);

    // Verify UI shows completion
    await expect(page.locator('text=仿真完成！')).toBeVisible({ timeout: 10000 });

    // Navigate to results page
    await page.click('text=查看仿真结果');
    await page.waitForURL('http://localhost:3000/results', { timeout: 10000 });
    await page.waitForLoadState('networkidle');

    // Verify results page shows some data
    await expect(page.locator('text=/layer_0|layer_1|EndToEnd/i').first()).toBeVisible({ timeout: 15000 });

    // Screenshot for evidence
    await page.screenshot({ path: 'test-results/ns3-fullmesh-results.png', fullPage: true });

    console.log(`NS3 full-mesh E2E test passed (PID ${pid})`);
  });
});
