import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
} from 'recharts';

interface ComputeCommBreakdownProps {
  readonly totalCompute: number;
  readonly totalExposed: number;
}

const COLORS = ['#3b82f6', '#ef4444'];

export function ComputeCommBreakdown({ totalCompute = 0, totalExposed = 0 }: ComputeCommBreakdownProps) {
  if (totalCompute === 0 && totalExposed === 0) return null;

  const data = [
    { name: 'Compute', value: totalCompute },
    { name: 'Exposed Comm', value: totalExposed },
  ];

  const totalTime = totalCompute + totalExposed;

  return (
    <div className="rounded-lg border border-[var(--color-bg-hover)] p-4 bg-[var(--color-bg-secondary)]">
      <h3 className="text-sm font-medium text-[var(--color-text-primary)] mb-4">
        Compute vs Communication ({totalTime.toFixed(0)} us total)
      </h3>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={90}
            paddingAngle={2}
            dataKey="value"
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            label={(props: any) => `${props.name ?? ''} ${(((props.percent as number) ?? 0) * 100).toFixed(1)}%`}
          >
            {data.map((_, idx) => (
              <Cell key={idx} fill={COLORS[idx]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={((value: unknown) => [`${Number(value ?? 0).toFixed(2)} us`, '']) as any}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
