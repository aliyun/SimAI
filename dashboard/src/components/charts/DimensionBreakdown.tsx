import type { DimensionEntry } from '../../types/wizard';

interface DimensionBreakdownProps {
  readonly dimensions: Record<string, DimensionEntry>;
}

const DIMENSION_CONFIG = [
  { key: 'Expose DP comm', label: 'DP Comm', color: '#ef4444', alwaysShow: true },
  { key: 'Expose DP_EP comm', label: 'DP+EP Comm', color: '#dc2626', alwaysShow: true },
  { key: 'Expose TP comm', label: 'TP Comm', color: '#3b82f6', alwaysShow: true },
  { key: 'Expose_EP_comm', label: 'EP Comm', color: '#8b5cf6', alwaysShow: true },
  { key: 'Expose_PP_comm', label: 'PP Comm', color: '#f59e0b', alwaysShow: true },
  { key: 'bubble time', label: 'Bubble', color: '#6b7280', alwaysShow: false },
  { key: 'total comp', label: 'Compute', color: '#10b981', alwaysShow: false },
] as const;

export function DimensionBreakdown({ dimensions }: DimensionBreakdownProps) {
  if (!dimensions || Object.keys(dimensions).length === 0) return null;

  // Always show the four parallelism dimensions (even 0) so a missing EP/PP
  // slot is visually explicit rather than silently filtered out.
  const chartData = DIMENSION_CONFIG
    .filter(({ key, alwaysShow }) => alwaysShow || (dimensions[key] && dimensions[key].value > 0))
    .map(({ key, label, color }) => ({
      name: label,
      value: dimensions[key]?.value ?? 0,
      percentage: dimensions[key]?.percentage ?? 0,
      fill: color,
    }));

  if (chartData.length === 0) return null;

  const total = chartData.reduce((s, d) => s + d.value, 0);

  return (
    <div className="rounded-lg border border-[var(--color-bg-hover)] p-4 bg-[var(--color-bg-secondary)]">
      <h3 className="text-sm font-medium text-[var(--color-text-primary)] mb-3">
        Time Dimension Breakdown (DP/TP/EP/PP)
      </h3>
      <div className="space-y-2">
        {chartData.map((d) => (
          <div key={d.name}>
            <div className="flex justify-between text-xs mb-0.5">
              <span className="text-[var(--color-text-secondary)]">{d.name}</span>
              <span className="text-[var(--color-text-muted)]">
                {d.value.toLocaleString(undefined, { maximumFractionDigits: 0 })} us &middot; {d.percentage.toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-[var(--color-bg-primary)] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${d.percentage}%`,
                  backgroundColor: d.fill,
                }}
              />
            </div>
          </div>
        ))}
        <div className="pt-1 border-t border-[var(--color-bg-hover)] flex justify-between text-xs font-medium">
          <span className="text-[var(--color-text-secondary)]">Total</span>
          <span className="text-[var(--color-text-primary)]">{total.toLocaleString(undefined, { maximumFractionDigits: 0 })} us</span>
        </div>
      </div>
    </div>
  );
}
