import { useState } from 'react';
import { usePolling } from '../../hooks/use-polling';
import { fetchNodeMetrics } from '../../api/topology-api';
import type { NodeLiveMetrics } from '../../types/metrics';

interface NodeDetailPanelProps {
  readonly nodeId: string;
  readonly onClose: () => void;
}

export function NodeDetailPanel({ nodeId, onClose }: NodeDetailPanelProps) {
  const [metrics, setMetrics] = useState<NodeLiveMetrics | null>(null);

  usePolling(
    async () => {
      const m = await fetchNodeMetrics(nodeId);
      setMetrics(m);
    },
    { intervalMs: 3000 },
  );

  return (
    <div className="w-80 bg-[var(--color-bg-secondary)] border-l border-[var(--color-bg-hover)] p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">{nodeId}</h2>
        <button onClick={onClose} className="text-[var(--color-text-secondary)] hover:text-white text-xl leading-none">
          &times;
        </button>
      </div>

      {metrics ? (
        <div className="space-y-3">
          <MetricRow label="NPU Utilization" value={`${(metrics.npuUtilization * 100).toFixed(1)}%`} />
          <MetricRow label="CPU Utilization" value={`${(metrics.cpuUtilization * 100).toFixed(1)}%`} />
          <MetricRow label="Memory" value={`${metrics.memoryUsedGb.toFixed(1)} / ${metrics.memoryTotalGb} GB`} />
          <MetricRow label="Temperature" value={`${metrics.temperature.toFixed(0)} C`} />
          <MetricRow label="Power" value={`${metrics.powerWatts.toFixed(0)} W`} />
          <MetricRow label="Status" value={metrics.status} />
        </div>
      ) : (
        <div className="text-[var(--color-text-muted)] text-center mt-8">Loading...</div>
      )}
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center text-sm">
      <span className="text-[var(--color-text-secondary)]">{label}</span>
      <span className="text-white">{value}</span>
    </div>
  );
}
