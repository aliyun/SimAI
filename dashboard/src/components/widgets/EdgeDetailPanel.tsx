import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { PortDetail, EdgeMetrics } from '../../types/topology';

interface EdgeDetailPanelProps {
  readonly edgeId: string;
  readonly source: string;
  readonly target: string;
  readonly metrics?: EdgeMetrics;
  readonly portDetails?: readonly PortDetail[];
  readonly onClose: () => void;
}

const COLOR_IN = '#3b82f6';   // blue-500
const COLOR_OUT = '#f59e0b';  // amber-500

function parseRate(s: string): number {
  const v = parseFloat(s.replace('%', ''));
  return isNaN(v) ? 0 : v;
}

/** Use full port name as-is from data */
function portLabel(port: string): string {
  return port;
}

export function EdgeDetailPanel({ edgeId, source, target, metrics, portDetails, onClose }: EdgeDetailPanelProps) {
  const ports = portDetails ?? [];
  const hasPorts = ports.length > 0;

  const chartData = useMemo(
    () =>
      ports.map((p) => ({
        port: portLabel(p.port),
        fullPort: p.port,
        inRate: parseRate(p.in_rate),
        outRate: parseRate(p.out_rate),
      })),
    [ports],
  );

  const maxRate = useMemo(() => {
    if (!chartData.length) return 0.2;
    const m = Math.max(...chartData.flatMap((d) => [d.inRate, d.outRate]));
    return Math.max(m * 1.3, 0.05);
  }, [chartData]);

  const totalInRate = chartData.reduce((s, d) => s + d.inRate, 0);
  const totalOutRate = chartData.reduce((s, d) => s + d.outRate, 0);
  const avgInRate = hasPorts ? totalInRate / ports.length : 0;
  const avgOutRate = hasPorts ? totalOutRate / ports.length : 0;

  return (
    <div className="w-[480px] bg-[var(--color-bg-secondary)] border-l border-[var(--color-bg-hover)] flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-[var(--color-bg-hover)]">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
            Link Detail
          </h2>
          <button
            onClick={onClose}
            className="text-[var(--color-text-secondary)] hover:text-white text-lg leading-none p-1 rounded hover:bg-[var(--color-bg-hover)] transition-colors"
          >
            &times;
          </button>
        </div>
        <div className="text-base font-medium text-white truncate" title={edgeId}>
          {source} → {target}
        </div>
      </div>

      {/* Summary stats */}
      <div className="px-4 py-3 border-b border-[var(--color-bg-hover)] grid grid-cols-3 gap-3">
        <div>
          <div className="text-xs text-[var(--color-text-secondary)]">Ports</div>
          <div className="text-lg font-semibold text-white">{hasPorts ? ports.length : '--'}</div>
        </div>
        {hasPorts ? (
          <>
            <div>
              <div className="text-xs text-[var(--color-text-secondary)]">Avg In</div>
              <div className="text-lg font-semibold tabular-nums" style={{ color: COLOR_IN }}>
                {avgInRate.toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--color-text-secondary)]">Avg Out</div>
              <div className="text-lg font-semibold tabular-nums" style={{ color: COLOR_OUT }}>
                {avgOutRate.toFixed(2)}%
              </div>
            </div>
          </>
        ) : metrics ? (
          <>
            <div>
              <div className="text-xs text-[var(--color-text-secondary)]">Utilization</div>
              <div className="text-lg font-semibold text-white tabular-nums">
                {(metrics.utilization * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--color-text-secondary)]">Links</div>
              <div className="text-lg font-semibold text-white">{metrics.linkCount}</div>
            </div>
          </>
        ) : null}
      </div>

      {/* Chart */}
      {hasPorts ? (
        <div className="flex-1 overflow-y-auto px-2 py-3">
          <ResponsiveContainer width="100%" height={Math.max(ports.length * 48 + 40, 200)}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 40, left: 10, bottom: 4 }}
              barCategoryGap="20%"
              barGap={2}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, maxRate]}
                tickFormatter={(v: number) => `${v.toFixed(2)}%`}
                tick={{ fill: '#9ca3af', fontSize: 10 }}
                axisLine={{ stroke: '#4b5563' }}
                tickLine={{ stroke: '#4b5563' }}
              />
              <YAxis
                type="category"
                dataKey="port"
                width={105}
                tick={{ fill: '#d1d5db', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={{ stroke: '#4b5563' }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 4 }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={((value: unknown, name: unknown) => [
                  `${Number(value ?? 0).toFixed(2)}%`,
                  name === 'inRate' ? 'In Rate' : 'Out Rate',
                ]) as any}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                labelFormatter={((_label: unknown, payload: readonly any[]) =>
                  payload?.[0]?.payload?.fullPort ?? String(_label ?? '')
                ) as any}
                cursor={{ fill: 'rgba(59,130,246,0.08)' }}
              />
              <Legend
                verticalAlign="top"
                height={28}
                formatter={(value: string) => (
                  <span style={{ color: '#d1d5db', fontSize: 11 }}>
                    {value === 'inRate' ? 'In Rate' : 'Out Rate'}
                  </span>
                )}
              />
              <Bar dataKey="inRate" name="inRate" radius={[0, 3, 3, 0]} maxBarSize={16}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={COLOR_IN} />
                ))}
              </Bar>
              <Bar dataKey="outRate" name="outRate" radius={[0, 3, 3, 0]} maxBarSize={16}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={COLOR_OUT} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-[var(--color-text-secondary)] text-sm px-4">
          <p className="text-center">No port-level data available for this link.</p>
        </div>
      )}
    </div>
  );
}
