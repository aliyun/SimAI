import { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from '@xyflow/react';
import type { MouseEvent as ReactMouseEvent } from 'react';
import '@xyflow/react/dist/style.css';

import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { MetricCard } from '../components/widgets/MetricCard';
import { PodNode } from '../components/nodes/PodNode';
import { MetricEdge } from '../components/edges/MetricEdge';
import { useNetworkStore } from '../stores/network-store';
import { useTopologyStore } from '../stores/topology-store';
import { usePolling } from '../hooks/use-polling';
import { scanTopologyDir, setTopologyDir } from '../api/simulation-api';
import { fetchOverviewTopology, fetchClusterSummary } from '../api/topology-api';
import type { TopologyFileEntry } from '../types/wizard';

const NODE_TYPES = { podNode: PodNode };
const EDGE_TYPES = { metricEdge: MetricEdge };
type Tab = 'monitor' | 'files';

export function NetworkDetailPage() {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const { getNetwork, updateNetwork, setActiveNetwork } = useNetworkStore();
  const network = id ? getNetwork(id) : undefined;

  const {
    overview, clusterSummary,
    setOverview, setClusterSummary, setError,
  } = useTopologyStore();

  const [activeTab, setActiveTab] = useState<Tab>('monitor');
  const [files, setFiles] = useState<readonly TopologyFileEntry[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [editingName, setEditingName] = useState(false);
  const [editName, setEditName] = useState('');

  // Set active network on mount
  useEffect(() => {
    if (network) {
      setActiveNetwork(network.id);
      setEditName(network.name);
    }
  }, [network, setActiveNetwork]);

  // Sync topologyDir to backend when switching to monitor tab
  const refreshTopology = useCallback(async () => {
    if (!network) return;
    try {
      await setTopologyDir(network.topologyDir);
    } catch {
      // ignore — monitor will show error via polling
    }
  }, [network]);

  useEffect(() => {
    if (activeTab === 'monitor' && network) {
      refreshTopology();
    }
  }, [activeTab, network, refreshTopology]);

  usePolling(
    async () => {
      if (!network) return;
      const [topo, summary] = await Promise.all([
        fetchOverviewTopology(),
        fetchClusterSummary(),
      ]);
      setOverview(topo);
      setClusterSummary(summary);
    },
    { intervalMs: 5000, onError: (e) => setError(e.message), enabled: activeTab === 'monitor' },
  );

  const handleScan = useCallback(async () => {
    if (!network) return;
    setIsScanning(true);
    setScanError(null);
    try {
      await setTopologyDir(network.topologyDir);
      const scan = await scanTopologyDir();
      setFiles(scan.files);
    } catch (err) {
      setScanError(err instanceof Error ? err.message : '扫描失败');
    } finally {
      setIsScanning(false);
    }
  }, [network]);

  useEffect(() => {
    if (network && activeTab === 'files') {
      handleScan();
    }
  }, [network, activeTab]);

  const handleSaveName = () => {
    if (!id || !editName.trim()) return;
    updateNetwork(id, { name: editName.trim() });
    setEditingName(false);
  };

  const rfNodes: Node[] = useMemo(() => {
    if (!overview) return [];
    const xs = overview.nodes.map((n) => n.position.x);
    const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const SPREAD = 3;
    return overview.nodes.map((n) => ({
      id: n.id,
      type: 'podNode',
      position: { x: centerX + (n.position.x - centerX) * SPREAD, y: n.position.y },
      data: { label: n.label, status: 'healthy' as const, npuCount: 32, utilization: 0.0061 },
    }));
  }, [overview]);

  const onNodeClick = useCallback(
    (_event: ReactMouseEvent, node: Node) => {
      navigate(`/monitor/pod/${encodeURIComponent(node.id)}`);
    },
    [navigate],
  );

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

  if (!network) {
    return (
      <DashboardShell>
        <Header title="组网" backLink="/" />
        <div className="flex-1 flex items-center justify-center">
          <p className="text-[var(--color-text-muted)]">组网不存在</p>
        </div>
      </DashboardShell>
    );
  }

  return (
    <DashboardShell>
      <Header title={network.name} backLink="/" />

      {/* Metrics bar — always visible in monitor tab */}
      {activeTab === 'monitor' && (
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
      )}

      {/* Content area */}
      <div className="flex-1 flex overflow-hidden flex-col">
        {/* Top bar: name + tabs */}
        <div className="px-6 py-3 border-b border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)] flex items-center justify-between">
          <div className="flex items-center gap-3">
            {editingName ? (
              <>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="px-3 py-1.5 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-sm font-semibold text-[var(--color-text-primary)]"
                  autoFocus
                  onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
                />
                <button onClick={handleSaveName} className="px-3 py-1.5 rounded-lg bg-[var(--color-accent-cyan)] text-white text-xs font-medium">
                  保存
                </button>
                <button onClick={() => setEditingName(false)} className="px-3 py-1.5 rounded-lg border border-[var(--color-bg-hover)] text-[var(--color-text-secondary)] text-xs">
                  取消
                </button>
              </>
            ) : (
              <button
                onClick={() => setEditingName(true)}
                className="text-sm font-semibold text-[var(--color-text-primary)] hover:text-[var(--color-accent-cyan)] transition-colors"
                title="点击修改名称"
              >
                {network.name}
              </button>
            )}
            <span className="text-xs text-[var(--color-text-muted)] truncate max-w-xs">{network.topologyDir}</span>
          </div>

          {/* Tabs */}
          <div className="flex gap-1">
            {([
              { value: 'monitor' as const, label: '监控' },
              { value: 'files' as const, label: '拓扑文件' },
            ] as const).map((tab) => (
              <button
                key={tab.value}
                onClick={() => setActiveTab(tab.value)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === tab.value
                    ? 'bg-[var(--color-accent-cyan)]/15 text-[var(--color-accent-cyan)]'
                    : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Monitor tab: ReactFlow */}
        {activeTab === 'monitor' && (
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
        )}

        {/* Files tab */}
        {activeTab === 'files' && (
          <div className="flex-1 overflow-y-auto px-6 py-6">
            <div className="max-w-4xl mx-auto space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
                  拓扑文件 ({files.length})
                </h3>
                <button
                  onClick={handleScan}
                  disabled={isScanning}
                  className="text-xs text-[var(--color-accent-cyan)] hover:underline disabled:opacity-50"
                >
                  {isScanning ? '扫描中...' : '刷新'}
                </button>
              </div>

              {scanError && <p className="text-xs text-[var(--color-accent-red)]">{scanError}</p>}

              {files.length > 0 ? (
                <div className="rounded-lg border border-[var(--color-bg-hover)] divide-y divide-[var(--color-bg-hover)]">
                  {files.map((file) => (
                    <div key={file.path} className="px-4 py-2.5 flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          file.type === 'directory'
                            ? 'bg-[var(--color-accent-cyan)]/10 text-[var(--color-accent-cyan)]'
                            : 'bg-[var(--color-bg-tertiary)] text-[var(--color-text-muted)]'
                        }`}>
                          {file.type === 'directory' ? 'DIR' : 'FILE'}
                        </span>
                        <span className="text-[var(--color-text-primary)]">{file.name}</span>
                      </div>
                      <span className="text-xs text-[var(--color-text-muted)]">
                        {file.size != null ? `${(file.size / 1024).toFixed(1)} KB` : ''}
                      </span>
                    </div>
                  ))}
                </div>
              ) : !isScanning && (
                <p className="text-sm text-[var(--color-text-muted)] text-center py-8 border border-dashed border-[var(--color-bg-hover)] rounded-lg">
                  点击刷新扫描拓扑目录
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </DashboardShell>
  );
}
