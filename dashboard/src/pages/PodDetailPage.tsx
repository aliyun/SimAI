import { useCallback, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { MetricCard } from '../components/widgets/MetricCard';
import { NetworkDeviceNode } from '../components/nodes/NetworkDeviceNode';
import { SuperNodeBoundary } from '../components/nodes/SuperNodeBoundary';
import { MetricEdge } from '../components/edges/MetricEdge';
import { NodeDetailPanel } from '../components/widgets/NodeDetailPanel';
import { EdgeDetailPanel } from '../components/widgets/EdgeDetailPanel';
import { useTopologyStore } from '../stores/topology-store';
import { usePolling } from '../hooks/use-polling';
import { fetchPodDetail } from '../api/topology-api';

const NODE_TYPES = {
  networkDevice: NetworkDeviceNode,
  superNodeBoundary: SuperNodeBoundary,
};
const EDGE_TYPES = { metricEdge: MetricEdge };

export function PodDetailPage() {
  const { podId } = useParams<{ podId: string }>();
  const decodedPodId = podId ? decodeURIComponent(podId) : '';
  const { podDetail, selectedNodeId, selectedEdgeId, lastUpdated, setPodDetail, selectNode, selectEdge, setError } =
    useTopologyStore();

  usePolling(
    async () => {
      if (!decodedPodId) return;
      const detail = await fetchPodDetail(decodedPodId);
      setPodDetail(detail);
    },
    { intervalMs: 5000, enabled: !!decodedPodId, onError: (e) => setError(e.message) },
  );

  const rfNodes: Node[] = useMemo(() => {
    if (!podDetail) return [];

    const boundaryNodes: Node[] = podDetail.superNodes.map((sn) => ({
      id: sn.id,
      type: 'superNodeBoundary',
      position: { x: sn.boundingBox.x, y: sn.boundingBox.y },
      data: { label: sn.id, width: sn.boundingBox.width, height: sn.boundingBox.height },
      selectable: false,
      draggable: false,
      zIndex: 1,
    }));

    const deviceNodes: Node[] = podDetail.nodes.map((n) => ({
      id: n.id,
      type: 'networkDevice',
      position: n.position,
      data: {
        label: n.label,
        nodeType: n.type,
        nodeIp: n.metadata.nodeIp,
        serverType: n.metadata.serverType,
        status: 'healthy' as const,
        width: n.size?.width,
        height: n.size?.height,
      },
      zIndex: 10,
    }));

    return [...boundaryNodes, ...deviceNodes];
  }, [podDetail]);

  const rfEdges: Edge[] = useMemo(() => {
    if (!podDetail) return [];
    return podDetail.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: 'metricEdge',
      data: {
        metrics: e.metrics,
        portMapping: e.portMapping,
        bandwidthGbps: e.metrics.bandwidthGbps,
        latencyUs: e.metrics.latencyUs,
        portDetails: e.portDetails,
      },
    }));
  }, [podDetail]);

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      selectNode(node.id === selectedNodeId ? null : node.id);
      if (node.id !== selectedNodeId) {
        selectEdge(null);
      }
    },
    [selectNode, selectedNodeId, selectEdge],
  );

  const onEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      selectEdge(edge.id === selectedEdgeId ? null : edge.id);
      if (edge.id !== selectedEdgeId) {
        selectNode(null);
      }
    },
    [selectEdge, selectedEdgeId, selectNode],
  );

  const selectedEdge = useMemo(() => {
    if (!selectedEdgeId || !podDetail) return null;
    return podDetail.edges.find((e) => e.id === selectedEdgeId) ?? null;
  }, [selectedEdgeId, podDetail]);

  const serverCount = podDetail?.nodes.filter((n) => n.type === 'SERVER').length ?? 0;

  return (
    <DashboardShell>
      <Header title={`${decodedPodId} Detail`} backLink="/monitor" lastUpdated={lastUpdated} />

      <div className="px-6 py-3 flex gap-4 bg-[var(--color-bg-secondary)] border-b border-[var(--color-bg-hover)]">
        <MetricCard label="Servers" value={serverCount || '--'} />
        <MetricCard label="Spines" value={podDetail?.nodes.filter((n) => n.type === 'SPINE').length ?? '--'} />
        <MetricCard label="Leaves" value={podDetail?.nodes.filter((n) => n.type === 'LEAF').length ?? '--'} />
        <MetricCard label="Links" value={podDetail?.edges.length ?? '--'} />
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1">
          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            fitView
            proOptions={{ hideAttribution: true }}
            minZoom={0.3}
            maxZoom={2}
          >
            <Background color="#1f2937" gap={24} />
            <Controls />
          </ReactFlow>
        </div>

        {selectedNodeId && (
          <NodeDetailPanel nodeId={selectedNodeId} onClose={() => selectNode(null)} />
        )}

        {selectedEdge && (
          <EdgeDetailPanel
            edgeId={selectedEdge.id}
            source={selectedEdge.source}
            target={selectedEdge.target}
            metrics={selectedEdge.metrics}
            portDetails={selectedEdge.portDetails ? [...selectedEdge.portDetails] : undefined}
            onClose={() => selectEdge(null)}
          />
        )}
      </div>
    </DashboardShell>
  );
}
