import type { NodeStatus } from '../../types/topology';

interface MetricCardProps {
  readonly label: string;
  readonly value: string | number;
  readonly unit?: string;
  readonly status?: NodeStatus;
}

const STATUS_BORDER: Record<NodeStatus, string> = {
  healthy: 'border-l-[var(--color-status-healthy)]',
  warning: 'border-l-[var(--color-status-warning)]',
  critical: 'border-l-[var(--color-status-critical)]',
  unknown: 'border-l-[var(--color-status-unknown)]',
};

export function MetricCard({ label, value, unit, status = 'healthy' }: MetricCardProps) {
  return (
    <div className={`bg-[var(--color-bg-tertiary)] rounded-lg px-4 py-3 border-l-4 ${STATUS_BORDER[status]} flex flex-col gap-1 min-w-[140px]`}>
      <span className="text-xs text-[var(--color-text-muted)] uppercase tracking-wider">{label}</span>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold">{value}</span>
        {unit && <span className="text-sm text-[var(--color-text-secondary)]">{unit}</span>}
      </div>
    </div>
  );
}
