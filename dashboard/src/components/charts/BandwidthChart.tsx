import { useState, useMemo } from 'react';
import type { LayerResult } from '../../types/wizard';
import { Pagination } from '../widgets/Pagination';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

const CHART_THRESHOLD = 15;
const TABLE_PAGE_SIZE = 50;
const CHART_MAX_BARS = 60;

interface BandwidthChartProps {
  readonly layers: readonly LayerResult[];
}

export function BandwidthChart({ layers }: BandwidthChartProps) {
  const [showTable, setShowTable] = useState(layers.length > CHART_THRESHOLD);
  const [tablePage, setTablePage] = useState(1);
  const [chartPage, setChartPage] = useState(1);

  if (!layers || layers.length === 0) return null;

  const processed = layers.map((l) => {
    const fwd = (l.fwd_algbw ?? 0) + (l.fwd_busbw ?? 0);
    const wg = (l.wg_algbw ?? 0) + (l.wg_busbw ?? 0);
    const ig = (l.ig_algbw ?? 0) + (l.ig_busbw ?? 0);
    return { name: l.layer_name, fwd, wg, ig, total: fwd + wg + ig };
  });

  const sorted = [...processed].sort((a, b) => b.total - a.total);

  const pagedTableRows = useMemo(() => {
    const start = (tablePage - 1) * TABLE_PAGE_SIZE;
    return sorted.slice(start, start + TABLE_PAGE_SIZE);
  }, [sorted, tablePage]);

  const pagedChartData = useMemo(() => {
    const start = (chartPage - 1) * CHART_MAX_BARS;
    return sorted.slice(start, start + CHART_MAX_BARS);
  }, [sorted, chartPage]);

  const fmt = (v: number | null | undefined) => {
    if (v == null || isNaN(v) || v === 0) return '—';
    return v.toFixed(2);
  };

  return (
    <div className="rounded-lg border border-[var(--color-bg-hover)] p-4 bg-[var(--color-bg-secondary)]">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-[var(--color-text-primary)]">
          Bandwidth Profile (algBW + busBW per phase)
        </h3>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setShowTable(false)}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              !showTable
                ? 'bg-[var(--color-accent-blue)]/20 text-[var(--color-accent-blue)]'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
            }`}
          >
            Chart
          </button>
          <button
            type="button"
            onClick={() => setShowTable(true)}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              showTable
                ? 'bg-[var(--color-accent-blue)]/20 text-[var(--color-accent-blue)]'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
            }`}
          >
            Table ({layers.length})
          </button>
        </div>
      </div>

      {showTable ? (
        <div className="overflow-auto max-h-96 text-xs">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-[var(--color-bg-tertiary)]">
              <tr>
                <th className="text-left px-3 py-2 text-[var(--color-text-muted)] font-medium">#</th>
                <th className="text-left px-3 py-2 text-[var(--color-text-muted)] font-medium">Layer</th>
                <th className="text-right px-3 py-2 text-[var(--color-text-muted)] font-medium">FWD</th>
                <th className="text-right px-3 py-2 text-[var(--color-text-muted)] font-medium">WG</th>
                <th className="text-right px-3 py-2 text-[var(--color-text-muted)] font-medium">IG</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-bg-hover)]">
              {pagedTableRows.map((l, i) => (
                <tr key={i} className="hover:bg-[var(--color-bg-hover)]/30">
                  <td className="px-3 py-1.5 text-[var(--color-text-muted)]">{(tablePage - 1) * TABLE_PAGE_SIZE + i + 1}</td>
                  <td className="px-3 py-1.5 text-[var(--color-text-primary)] font-mono max-w-[200px] truncate">{l.name}</td>
                  <td className="text-right px-3 py-1.5 text-[#2563eb]">{fmt(l.fwd)}</td>
                  <td className="text-right px-3 py-1.5 text-[#059669]">{fmt(l.wg)}</td>
                  <td className="text-right px-3 py-1.5 text-[#7c3aed]">{fmt(l.ig)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pagination currentPage={tablePage} totalItems={sorted.length} pageSize={TABLE_PAGE_SIZE} onPageChange={setTablePage} />
        </div>
      ) : (
        <div>
        <ResponsiveContainer width="100%" height={Math.min(600, Math.max(300, pagedChartData.length * 28))}>
          <BarChart
            data={pagedChartData}
            layout="horizontal"
            margin={{ left: 10, right: 20, top: 5, bottom: Math.max(80, pagedChartData.length * 0.5) }}
            barCategoryGap="4%"
            barGap={2}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal vertical />
            <XAxis
              type="category"
              dataKey="name"
              stroke="#9ca3af"
              tick={{ fontSize: 9 }}
              angle={-45}
              textAnchor="end"
              height={80}
              interval={0}
            />
            <YAxis
              type="number"
              stroke="#9ca3af"
              tick={{ fontSize: 10 }}
              label={{ value: 'Bandwidth (GB/s)', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
              formatter={(value: unknown) => {
                const v = value as number;
                if (v === 0) return ['—', ''];
                return [`${v.toFixed(2)} GB/s`, ''];
              }}
              labelStyle={{ color: '#e5e7eb', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}
            />
            <Legend
              wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
              formatter={(value: string) => <span style={{ color: '#9ca3af', fontSize: 10 }}>{value}</span>}
            />
            <Bar dataKey="fwd" stackId="a" fill="#2563eb" name="FWD" />
            <Bar dataKey="wg" stackId="a" fill="#059669" name="WG" />
            <Bar dataKey="ig" stackId="a" fill="#7c3aed" name="IG" />
          </BarChart>
        </ResponsiveContainer>
        <Pagination currentPage={chartPage} totalItems={sorted.length} pageSize={CHART_MAX_BARS} onPageChange={setChartPage} />
        </div>
      )}
    </div>
  );
}
