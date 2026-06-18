import { getStraightPath, type EdgeProps } from '@xyflow/react';

export type DiffType = 'added' | 'removed' | 'unchanged' | 'server-leaf';

interface DiffEdgeData {
  readonly diffType: DiffType;
  readonly oxcIp?: string;
  readonly portA?: string;
  readonly portB?: string;
}

const STYLE_MAP: Record<DiffType, { stroke: string; strokeWidth: number; dasharray?: string; glow?: boolean }> = {
  added:       { stroke: '#10b981', strokeWidth: 3, glow: true },
  removed:     { stroke: '#ef4444', strokeWidth: 2, dasharray: '6 4' },
  unchanged:   { stroke: '#4b5563', strokeWidth: 1.5 },
  'server-leaf': { stroke: '#374151', strokeWidth: 1 },
};

export function DiffEdge({
  id, sourceX, sourceY, targetX, targetY, data,
}: EdgeProps) {
  const edgeData = data as DiffEdgeData | undefined;
  const diffType = edgeData?.diffType ?? 'unchanged';
  const style = STYLE_MAP[diffType];

  const [edgePath] = getStraightPath({ sourceX, sourceY, targetX, targetY });

  return (
    <g>
      {style.glow && (
        <defs>
          <filter id={`glow-${id}`}>
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      )}
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={style.stroke}
        strokeWidth={style.strokeWidth}
        strokeDasharray={style.dasharray}
        filter={style.glow ? `url(#glow-${id})` : undefined}
      />
    </g>
  );
}
