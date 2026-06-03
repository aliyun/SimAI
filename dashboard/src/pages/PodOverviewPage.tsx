import { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useNavigate } from 'react-router-dom';

import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { MetricCard } from '../components/widgets/MetricCard';
import { PodNode } from '../components/nodes/PodNode';
import { MetricEdge } from '../components/edges/MetricEdge';
import { useTopologyStore } from '../stores/topology-store';
import { usePolling } from '../hooks/use-polling';
import { fetchOverviewTopology, fetchClusterSummary } from '../api/topology-api';

const NODE_TYPES = { podNode: PodNode };
const EDGE_TYPES = { metricEdge: MetricEdge };

export function PodOverviewPage() {
  const navigate = useNavigate();
  const { overview, clusterSummary, lastUpdated, setOverview, setClusterSummary, setError } =
    useTopologyStore();

  usePolling(
    async () => {
      const [topo, summary] = await Promise.all([
        fetchOverviewTopology(),
        fetchClusterSummary(),
      ]);
      setOverview(topo);
      setClusterSummary(summary);
    },
    { intervalMs: 5000, onError: (e) => setError(e.message) },
  );

  const rfNodes: Node[] = useMemo(() => {
    if (!overview) return [];
    // Scale positions to increase spacing between PODs for label readability.
    // XML positions are tightly packed (250px apart); we center and spread them.
    const xs = overview.nodes.map((n) => n.position.x);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const SPREAD = 3;
    return overview.nodes.map((n) => ({
      id: n.id,
      type: 'podNode',
      position: {
        x: centerX + (n.position.x - centerX) * SPREAD,
        y: n.position.y,
      },
      data: {
        label: n.label,
        status: 'healthy' as const,
        npuCount: 32,
        utilization: 0.0061,
      },
    }));
  }, [overview]);

  const rfEdges: Edge[] = useMemo(() => {
    if (!overview) return [];
    return overview.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: 'metricEdge',
      data: {
        metrics: e.metrics,
        sourceMetrics: e.sourceMetrics,
        targetMetrics: e.targetMetrics,
        portMapping: e.portMapping,
        bandwidthGbps: e.metrics.bandwidthGbps,
        latencyUs: e.metrics.latencyUs,
      },
    }));
  }, [overview]);

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      navigate(`/monitor/pod/${encodeURIComponent(node.id)}`);
    },
    [navigate],
  );

  return (
    <DashboardShell>
      <Header title="OCS-Sim Network Monitoring" lastUpdated={lastUpdated} />

      <div className="px-6 py-3 flex gap-4 bg-[var(--color-bg-secondary)] border-b border-[var(--color-bg-hover)]">
        <MetricCard label="Total PODs" value={clusterSummary?.totalPods ?? '--'} />
        <MetricCard label="Total NPUs" value={clusterSummary?.totalNpus ?? '--'} />
        <MetricCard
          label="Utilization"
          value={clusterSummary ? `${(clusterSummary.overallUtilization * 100).toFixed(1)}%` : '--'}
          status={clusterSummary && clusterSummary.overallUtilization > 0.8 ? 'warning' : 'healthy'}
        />
        <MetricCard
          label="Active Alerts"
          value={clusterSummary?.activeAlerts ?? 0}
          status={clusterSummary && clusterSummary.activeAlerts > 0 ? 'warning' : 'healthy'}
        />
      </div>

      <div className="flex-1">
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          onNodeClick={onNodeClick}
          fitView
          proOptions={{ hideAttribution: true }}
          minZoom={0.5}
          maxZoom={2}
        >
          <Background color="#1f2937" gap={24} />
          <Controls />
          <MiniMap
            nodeColor="#3b82f6"
            maskColor="rgba(10, 14, 23, 0.8)"
            style={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '8px' }}
          />
        </ReactFlow>
      </div>
    </DashboardShell>
  );
}
