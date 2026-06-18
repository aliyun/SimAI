import type {
  PodOverviewTopology,
  PodDetailTopology,
  TopologyNode,
  TopologyEdge,
  EdgeMetrics,
  SuperNodeGroup,
} from '../types/topology';

// --- Jitter utility: adds +-20% random variation to simulate real-time updates ---

function jitter(base: number, ratio = 0.2): number {
  return base * (1 + (Math.random() * 2 - 1) * ratio);
}

function jitterMetrics(m: EdgeMetrics): EdgeMetrics {
  return {
    ...m,
    utilization: jitter(m.utilization),
    errorRate: jitter(m.errorRate),
    latencyUs: m.latencyUs != null ? jitter(m.latencyUs) : undefined,
    uplinkErrorRange: m.uplinkErrorRange,
    downlinkErrorRange: m.downlinkErrorRange,
  };
}

function jitterEdges(edges: readonly TopologyEdge[]): TopologyEdge[] {
  return edges.map((e) => ({ ...e, metrics: jitterMetrics(e.metrics) }));
}

// --- POD Overview (dynamic: supports any number of PODs) ---

export const MOCK_POD_COUNT = 2;

function generatePodOverviewNodes(count: number): readonly TopologyNode[] {
  const centerY = 300;
  const spacing = 350;
  const totalWidth = (count - 1) * spacing;
  const startX = 500 - totalWidth / 2;
  return Array.from({ length: count }, (_, i) => ({
    id: `POD#${i + 1}`,
    type: 'POD' as const,
    label: `POD#${i + 1}`,
    position: {
      x: Math.round(startX + i * spacing),
      y: centerY,
    },
    metadata: {},
  }));
}

function createInterPodEdge(source: string, target: string, linkCount: number): TopologyEdge {
  return {
    id: `${source}-${target}`,
    source,
    target,
    metrics: { utilization: 0.0000, linkCount, errorRate: 0.0000, bandwidthGbps: 400, latencyUs: 3.2 },
  };
}

function generatePodOverviewEdges(count: number): readonly TopologyEdge[] {
  const edges: TopologyEdge[] = [];
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      const src = `POD#${i + 1}`;
      const tgt = `POD#${j + 1}`;
      edges.push(createInterPodEdge(src, tgt, 8));
    }
  }
  return edges;
}

export function getMockPodOverview(): PodOverviewTopology {
  return {
    nodes: generatePodOverviewNodes(MOCK_POD_COUNT),
    edges: jitterEdges(generatePodOverviewEdges(MOCK_POD_COUNT)),
  };
}

// --- POD Detail (derived from draw.io XML) ---
// Topology: SSW_182 (SPINE top) → SSW_181 (SPINE mid) → LSW_195/LSW_198 (LEAF) → worker-172/worker-192 (SERVER)
// Layout is dynamically computed: centered, symmetric.

const POD_DETAIL_LAYOUT = {
  centerX: 400,
  pairSpacing: 500,
  layerY: { spineTop: 20, spineMid: 270, leaf: 530, server: 790 },
} as const;

function createPodDetailNodes(_podId: string, _podIndex: number): readonly TopologyNode[] {
  const cx = POD_DETAIL_LAYOUT.centerX;
  const half = POD_DETAIL_LAYOUT.pairSpacing / 2;
  const y = POD_DETAIL_LAYOUT.layerY;
  // SPINE nodes are wide (500px), centered at cx
  // LEAF nodes are 224px wide, SERVER nodes are 160px wide
  const spineX = cx - 250;
  const leafLeftX = cx - half;
  const leafRightX = cx + half - 224;
  const srvLeftX = cx - half + 32;
  const srvRightX = cx + half - 192;

  return [
    { id: 'SSW_182', type: 'SPINE', label: 'E0319_SSW_182\nSPINE', position: { x: spineX, y: y.spineTop }, metadata: { nodeIp: '10.44.231.182' } },
    { id: 'SSW_181', type: 'SPINE', label: 'E0319_SSW_181\nSPINE', position: { x: spineX, y: y.spineMid }, metadata: { nodeIp: '10.44.231.181' } },
    { id: 'LSW_195', type: 'LEAF', label: 'E0319_LSW_195\nLEAF', position: { x: leafLeftX, y: y.leaf }, metadata: { nodeIp: '10.44.231.195' } },
    { id: 'LSW_198', type: 'LEAF', label: 'E0319_LSW_198\nLEAF', position: { x: leafRightX, y: y.leaf }, metadata: { nodeIp: '10.44.231.198' } },
    { id: 'worker-172', type: 'SERVER', label: 'worker-172\nSERVER', position: { x: srvLeftX, y: y.server }, metadata: { nodeIp: '10.44.231.172' } },
    { id: 'worker-192', type: 'SERVER', label: 'worker-192\nSERVER', position: { x: srvRightX, y: y.server }, metadata: { nodeIp: '10.44.231.192' } },
  ];
}

function createPodDetailEdges(_podIndex: number): readonly TopologyEdge[] {
  return [
    // SSW_182 ↔ SSW_181
    {
      id: 'SSW_181-SSW_182',
      source: 'SSW_181',
      target: 'SSW_182',
      metrics: { utilization: 0.0012, linkCount: 1, errorRate: 0.0012, bandwidthGbps: 400, latencyUs: 2.0,
        uplinkErrorRange: '0.01%~0.12%', downlinkErrorRange: '0.01%~0.12%' },
    },
    // LSW_195 ↔ SSW_181
    {
      id: 'LSW_195-SSW_181',
      source: 'LSW_195',
      target: 'SSW_181',
      metrics: { utilization: 0.0008, linkCount: 1, errorRate: 0.0008, bandwidthGbps: 400, latencyUs: 2.8,
        uplinkErrorRange: '0.01%~0.08%', downlinkErrorRange: '0.01%~0.08%' },
    },
    // LSW_198 ↔ SSW_181
    {
      id: 'LSW_198-SSW_181',
      source: 'LSW_198',
      target: 'SSW_181',
      metrics: { utilization: 0.0012, linkCount: 1, errorRate: 0.0012, bandwidthGbps: 400, latencyUs: 2.8,
        uplinkErrorRange: '0.01%~0.12%', downlinkErrorRange: '0.01%~0.08%' },
    },
    // worker-172 ↔ LSW_195
    {
      id: 'worker-172-LSW_195',
      source: 'worker-172',
      target: 'LSW_195',
      metrics: { utilization: 0.0008, linkCount: 1, errorRate: 0.0008, bandwidthGbps: 400, latencyUs: 3.2,
        uplinkErrorRange: '0.08%~0.08%', downlinkErrorRange: '0.08%~0.08%' },
    },
    // worker-172 ↔ LSW_198
    {
      id: 'worker-172-LSW_198',
      source: 'worker-172',
      target: 'LSW_198',
      metrics: { utilization: 0.0008, linkCount: 1, errorRate: 0.0008, bandwidthGbps: 400, latencyUs: 3.2,
        uplinkErrorRange: '0.08%~0.08%', downlinkErrorRange: '0.08%~0.08%' },
    },
    // worker-192 ↔ LSW_195
    {
      id: 'worker-192-LSW_195',
      source: 'worker-192',
      target: 'LSW_195',
      metrics: { utilization: 0.0017, linkCount: 1, errorRate: 0.0017, bandwidthGbps: 400, latencyUs: 3.2,
        uplinkErrorRange: '0.17%~0.17%', downlinkErrorRange: '0.17%~0.17%' },
    },
    // worker-192 ↔ LSW_198
    {
      id: 'worker-192-LSW_198',
      source: 'worker-192',
      target: 'LSW_198',
      metrics: { utilization: 0.0000, linkCount: 1, errorRate: 0.0000, bandwidthGbps: 400, latencyUs: 3.2,
        uplinkErrorRange: '0.00%~0.00%', downlinkErrorRange: '0.00%~0.00%' },
    },
  ];
}

function createPodDetailSuperNodes(_podIndex: number): readonly SuperNodeGroup[] {
  // No super node grouping in the new topology
  return [];
}

export function getMockPodDetail(podId: string): PodDetailTopology {
  const podIndex = parseInt(podId.replace(/\D/g, ''), 10) - 1;
  const idx = isNaN(podIndex) ? 0 : Math.max(0, podIndex);
  return {
    podId,
    paraPlaneId: podId,
    nodes: createPodDetailNodes(podId, idx),
    edges: jitterEdges(createPodDetailEdges(idx)),
    superNodes: createPodDetailSuperNodes(idx),
  };
}

export const MOCK_POD1_DETAIL = getMockPodDetail('POD#1');
