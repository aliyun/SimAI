import type { Node, Edge } from '@xyflow/react';
import type { EdgGraph, EdgDiff, EdgLeafLeafEdge } from '../../types/edg';
import type { DiffType } from './DiffEdge';

// ===== Layout constants =====
const SERVER_Y = 40;
const LEAF_Y = 150;
const OXC_TOP_Y = 260;
const PORT_Y_OFFSET = 32;           // first port row y, relative to OXC top
const PORT_ROW_GAP = 4;
const OXC_BODY_PADDING_X = 18;
const OXC_BODY_PADDING_BOTTOM = 80; // room for the dip of port-port arcs

const SERVER_SPACING = 72;
const LEAF_SPACING = 180;
const PORT_W = 30;
const PORT_H = 22;
const PORT_GAP = 4;
const PORTS_PER_ROW = 2;

const GROUP_COLORS = ['#3b82f6', '#a855f7', '#f59e0b', '#10b981', '#ec4899', '#06b6d4'];

function pairKey(a: string, b: string): string {
  return [a, b].sort().join('::');
}

function shortIp(ip: string): string {
  const parts = ip.split('.');
  return parts.length === 4 ? `${parts[2]}.${parts[3]}` : ip;
}

function classifyDiff(
  tuple: EdgLeafLeafEdge,
  origin: 'current' | 'removed',
  addedSet: ReadonlySet<string>,
  removedSet: ReadonlySet<string>,
): DiffType {
  const k = pairKey(tuple[0], tuple[1]);
  if (origin === 'removed' || removedSet.has(k)) return 'removed';
  if (addedSet.has(k)) return 'added';
  return 'unchanged';
}

interface GroupLayout {
  readonly leafIp: string;
  readonly ports: readonly string[];       // ordered port ids
  readonly groupCenterX: number;
  readonly groupWidth: number;
  readonly numRows: number;
  readonly color: string;
}

function layoutPortGroups(
  oxcIp: string,
  portMap: Readonly<Record<string, string>>,
  leafIpToX: ReadonlyMap<string, number>,
): GroupLayout[] {
  // Group ports by their parent leaf
  const leafToPorts = new Map<string, string[]>();
  for (const [port, leafIp] of Object.entries(portMap)) {
    const arr = leafToPorts.get(leafIp) ?? [];
    arr.push(port);
    leafToPorts.set(leafIp, arr);
  }
  // Sort ports numerically within each group for a stable layout
  for (const [, arr] of leafToPorts) {
    arr.sort((a, b) => Number(a) - Number(b));
  }

  // Order groups by their parent leaf's x-position so ports read L→R
  const ordered = Array.from(leafToPorts.entries()).sort((a, b) => {
    const xa = leafIpToX.get(a[0]) ?? 0;
    const xb = leafIpToX.get(b[0]) ?? 0;
    return xa - xb;
  });

  return ordered.map(([leafIp, ports], idx) => {
    const groupWidth = PORTS_PER_ROW * PORT_W + (PORTS_PER_ROW - 1) * PORT_GAP;
    return {
      leafIp,
      ports,
      groupCenterX: leafIpToX.get(leafIp) ?? 0,
      groupWidth,
      numRows: Math.ceil(ports.length / PORTS_PER_ROW),
      color: GROUP_COLORS[idx % GROUP_COLORS.length],
    };
  });
}

function portPositionWithinGroup(
  group: GroupLayout,
  portIndex: number,
): { x: number; y: number } {
  const col = portIndex % PORTS_PER_ROW;
  const row = Math.floor(portIndex / PORTS_PER_ROW);
  const startX = group.groupCenterX - group.groupWidth / 2;
  const x = startX + col * (PORT_W + PORT_GAP);
  const y = OXC_TOP_Y + PORT_Y_OFFSET + row * (PORT_H + PORT_ROW_GAP);
  return { x, y };
}

export function buildEdgFlowElements(
  graph: EdgGraph,
  diff: EdgDiff | null,
  participatingIps?: ReadonlySet<string>,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // === Diff lookups ===
  const addedSet = new Set<string>();
  const removedSet = new Set<string>();
  if (diff) {
    for (const e of diff.added) addedSet.add(pairKey(e[0], e[1]));
    for (const e of diff.removed) removedSet.add(pairKey(e[0], e[1]));
  }

  // === Fallback: if backend didn't supply oxc_port_map, derive the subset of
  // ports referenced by baseline/removed crosses so we still render something. ===
  const portMapByOxc: Record<string, Record<string, string>> = {};
  if (graph.oxc_port_map) {
    for (const [oxcIp, pm] of Object.entries(graph.oxc_port_map)) {
      portMapByOxc[oxcIp] = { ...pm };
    }
  }
  const ensurePort = (oxcIp: string, port: string, leafIp: string) => {
    const pm = portMapByOxc[oxcIp] ?? {};
    if (!(port in pm)) pm[port] = leafIp;
    portMapByOxc[oxcIp] = pm;
  };
  for (const [leafA, leafB, oxcIp, portA, portB] of graph.leaf_leaf_edges) {
    ensurePort(oxcIp, portA, leafA);
    ensurePort(oxcIp, portB, leafB);
  }
  for (const e of diff?.removed ?? []) {
    ensurePort(e[2], e[3], e[0]);
    ensurePort(e[2], e[4], e[1]);
  }

  // === Leaves (their x positions drive port-group placement) ===
  const leafCount = graph.leaves.length;
  const totalLeafWidth = Math.max(0, (leafCount - 1) * LEAF_SPACING);
  const leafStartX = -totalLeafWidth / 2;
  const leafIpToX = new Map<string, number>();
  graph.leaves.forEach((leaf, i) => {
    leafIpToX.set(leaf.ip, leafStartX + i * LEAF_SPACING);
  });

  // === Per-OXC port group layouts ===
  const perOxcLayout: Array<{
    oxcIp: string;
    groups: GroupLayout[];
    bodyLeft: number;
    bodyRight: number;
    bodyTop: number;
    bodyBottom: number;
  }> = [];
  for (const oxcIp of Object.keys(portMapByOxc).sort()) {
    const groups = layoutPortGroups(oxcIp, portMapByOxc[oxcIp], leafIpToX);
    if (groups.length === 0) continue;
    const leftmost = Math.min(...groups.map((g) => g.groupCenterX - g.groupWidth / 2));
    const rightmost = Math.max(...groups.map((g) => g.groupCenterX + g.groupWidth / 2));
    const maxRows = Math.max(...groups.map((g) => g.numRows));
    const bodyTop = OXC_TOP_Y;
    const bodyBottom =
      OXC_TOP_Y + PORT_Y_OFFSET + maxRows * (PORT_H + PORT_ROW_GAP) + OXC_BODY_PADDING_BOTTOM;
    perOxcLayout.push({
      oxcIp,
      groups,
      bodyLeft: leftmost - OXC_BODY_PADDING_X,
      bodyRight: rightmost + OXC_BODY_PADDING_X,
      bodyTop,
      bodyBottom,
    });
  }

  // === Render OXC body (added FIRST so it renders behind ports) ===
  for (const l of perOxcLayout) {
    const width = l.bodyRight - l.bodyLeft;
    const height = l.bodyBottom - l.bodyTop;
    nodes.push({
      id: `oxc-body-${l.oxcIp}`,
      type: 'default',
      draggable: false,
      selectable: false,
      width,
      height,
      position: { x: l.bodyLeft, y: l.bodyTop },
      data: { label: `OXC  ${shortIp(l.oxcIp)}` },
      style: {
        width,
        height,
        background: 'rgba(146, 64, 14, 0.08)',
        border: '1.5px solid #d97706',
        borderRadius: '14px',
        color: '#fbbf24',
        fontSize: '11px',
        fontWeight: 700,
        padding: '6px 12px',
        textAlign: 'left' as const,
        zIndex: 0,
      },
    });
  }

  // === Render servers (top) ===
  const leafServerGroups = new Map<string, typeof graph.servers[number][]>();
  for (const srv of graph.servers) {
    const arr = leafServerGroups.get(srv.leaf_ip ?? '') ?? [];
    arr.push(srv);
    leafServerGroups.set(srv.leaf_ip ?? '', arr);
  }
  let srvIdx = 0;
  for (const [leafIp, servers] of leafServerGroups) {
    const leafX = leafIpToX.get(leafIp) ?? 0;
    const groupWidth = (servers.length - 1) * SERVER_SPACING;
    const startX = leafX - groupWidth / 2;
    servers.forEach((srv, i) => {
      srvIdx++;
      const isParticipating = participatingIps?.has(srv.ip) ?? true;
      nodes.push({
        id: `srv-${srv.ip}`,
        type: 'default',
        position: { x: startX + i * SERVER_SPACING, y: SERVER_Y },
        data: { label: `Srv ${srvIdx}\n${shortIp(srv.ip)}` },
        style: {
          background: isParticipating ? '#0e3a2d' : '#1a1a2e',
          color: isParticipating ? '#6ee7b7' : '#64748b',
          border: isParticipating ? '2px solid #10b981' : '1px solid #334155',
          borderRadius: '8px',
          fontSize: '10px',
          padding: '6px 10px',
          width: 'auto',
          textAlign: 'center' as const,
          whiteSpace: 'pre-line' as const,
          fontWeight: isParticipating ? 600 : 400,
        },
      });
    });
  }

  // === Render leaves ===
  graph.leaves.forEach((leaf, i) => {
    const x = leafStartX + i * LEAF_SPACING;
    nodes.push({
      id: `leaf-${leaf.ip}`,
      type: 'default',
      position: { x, y: LEAF_Y },
      data: { label: `Leaf ${i + 1}\n${shortIp(leaf.ip)}` },
      style: {
        background: '#1e293b',
        color: '#cbd5e1',
        border: '1px solid #64748b',
        borderRadius: '8px',
        fontSize: '11px',
        padding: '8px 14px',
        width: 'auto',
        textAlign: 'center' as const,
        whiteSpace: 'pre-line' as const,
        fontWeight: 600,
      },
    });
  });

  // === Render ports (after body, so they stack on top) + leaf→port fiber ===
  // Also build a lookup: (oxc, port) → node id, for cross edges later.
  const portNodeId = (oxcIp: string, port: string) => `port-${oxcIp}-${port}`;
  for (const l of perOxcLayout) {
    for (const g of l.groups) {
      g.ports.forEach((port, idx) => {
        const { x, y } = portPositionWithinGroup(g, idx);
        const nodeId = portNodeId(l.oxcIp, port);
        nodes.push({
          id: nodeId,
          type: 'default',
          width: PORT_W,
          height: PORT_H,
          position: { x, y },
          data: { label: port },
          style: {
            width: PORT_W,
            height: PORT_H,
            background: '#0b1220',
            color: g.color,
            border: `1.5px solid ${g.color}`,
            borderRadius: '6px',
            fontSize: '10px',
            padding: '2px 0',
            textAlign: 'center' as const,
            fontWeight: 700,
            zIndex: 2,
          },
        });
        // Fixed fiber: leaf → port (always gray, never part of diff)
        edges.push({
          id: `fiber-${l.oxcIp}-${port}`,
          source: `leaf-${g.leafIp}`,
          target: nodeId,
          type: 'diffEdge',
          data: { diffType: 'server-leaf' as DiffType },
        });
      });
    }
  }

  // === Server → Leaf edges ===
  for (const [srvIp, leafIp] of graph.server_leaf_edges) {
    edges.push({
      id: `sle-${srvIp}-${leafIp}`,
      source: `srv-${srvIp}`,
      target: `leaf-${leafIp}`,
      type: 'diffEdge',
      data: { diffType: 'server-leaf' as DiffType },
    });
  }

  // === Internal OXC crosses: port ↔ port arcs, diff-colored ===
  const pushCross = (tuple: EdgLeafLeafEdge, origin: 'current' | 'removed') => {
    const [, , oxcIp, portA, portB] = tuple;
    const srcId = portNodeId(oxcIp, portA);
    const dstId = portNodeId(oxcIp, portB);
    const diffType = classifyDiff(tuple, origin, addedSet, removedSet);
    edges.push({
      id: `cross-${origin}-${oxcIp}-${portA}-${portB}`,
      source: srcId,
      target: dstId,
      type: 'crossEdge',
      data: { diffType, portA, portB },
    });
  };
  for (const e of graph.leaf_leaf_edges) pushCross(e, 'current');
  for (const e of diff?.removed ?? []) pushCross(e, 'removed');

  return { nodes, edges };
}
