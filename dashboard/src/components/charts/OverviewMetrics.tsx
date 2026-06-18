interface OverviewMetricsProps {
  readonly totalTime: number | null;
  readonly totalCompute: number | null;
  readonly totalExposed: number | null;
  readonly finishTime: number | null;
  readonly streamsInjected: number | null;
  readonly streamsFinished: number | null;
}

export function OverviewMetrics({
  totalTime, totalCompute, totalExposed,
  finishTime, streamsInjected, streamsFinished,
}: OverviewMetricsProps) {
  const metrics = [
    { label: 'Total Time', value: totalTime, unit: 'us', color: 'blue' },
    { label: 'Compute', value: totalCompute, unit: 'us', color: 'green' },
    { label: 'Exposed Comm', value: totalExposed, unit: 'us', color: 'red' },
    { label: 'Finish Time', value: finishTime, unit: 'us', color: 'purple' },
    { label: 'Streams Injected', value: streamsInjected, unit: '', color: 'cyan' },
    { label: 'Streams Finished', value: streamsFinished, unit: '', color: 'yellow' },
  ];

  const colorMap: Record<string, string> = {
    blue: 'text-[var(--color-accent-blue)]',
    green: 'text-[var(--color-accent-green)]',
    red: 'text-[var(--color-accent-red)]',
    purple: 'text-[var(--color-accent-purple)]',
    cyan: 'text-[var(--color-accent-cyan)]',
    yellow: 'text-[var(--color-accent-yellow)]',
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      {metrics.map((m) => (
        <div
          key={m.label}
          className="rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)] p-3"
        >
          <div className="text-xs text-[var(--color-text-muted)]">{m.label}</div>
          <div className={`text-lg font-bold mt-0.5 ${colorMap[m.color] ?? ''}`}>
            {m.value != null ? (
              <>
                {typeof m.value === 'number' && m.unit ? m.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : m.value}
                {m.unit && <span className="text-xs font-normal ml-1 text-[var(--color-text-muted)]">{m.unit}</span>}
              </>
            ) : (
              <span className="text-[var(--color-text-muted)]">--</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
