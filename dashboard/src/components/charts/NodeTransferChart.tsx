import type { NodeTransfer } from '../../types/wizard';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

interface NodeTransferChartProps {
  readonly nodes: readonly NodeTransfer[];
}

export function NodeTransferChart({ nodes }: NodeTransferChartProps) {
  if (!nodes || nodes.length === 0) {
    return (
      <div className="rounded-lg border border-[var(--color-bg-hover)] p-6 bg-[var(--color-bg-secondary)] text-center">
        <p className="text-sm text-[var(--color-text-muted)]">
          暂无 Node Transfer 数据 — 请上传 Console Output 文件
        </p>
      </div>
    );
  }

  const data = nodes.map((n) => ({
    name: `Node ${n.id}`,
    sent_mb: n.sent / (1024 * 1024),
    received_mb: n.received / (1024 * 1024),
  }));

  return (
    <div className="rounded-lg border border-[var(--color-bg-hover)] p-4 bg-[var(--color-bg-secondary)]">
      <h3 className="text-sm font-medium text-[var(--color-text-primary)] mb-4">
        Data Transfer per Node
      </h3>
      <ResponsiveContainer width="100%" height={Math.max(250, nodes.length * 35)}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 11 }} />
          <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} label={{ value: 'MB', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="sent_mb" name="Sent (MB)" fill="#3b82f6" />
          <Bar dataKey="received_mb" name="Received (MB)" fill="#10b981" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
