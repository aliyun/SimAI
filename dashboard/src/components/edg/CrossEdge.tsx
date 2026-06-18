import { type EdgeProps } from '@xyflow/react';
import type { DiffType } from './DiffEdge';

interface CrossEdgeData {
  readonly diffType: DiffType;
  readonly portA?: string;
  readonly portB?: string;
}

const STYLE_MAP: Record<DiffType, { stroke: string; strokeWidth: number; dasharray?: string; glow?: boolean }> = {
  added:       { stroke: '#10b981', strokeWidth: 2.5, glow: true },
  removed:     { stroke: '#ef4444', strokeWidth: 2, dasharray: '5 4' },
  unchanged:   { stroke: '#64748b', strokeWidth: 1.5 },
  'server-leaf': { stroke: '#374151', strokeWidth: 1 },
};

// OXC internal port-to-port cross: a quadratic bezier dipping downward,
// keeping the arc entirely within the OXC body.
export function CrossEdge({
  id, sourceX, sourceY, targetX, targetY, data,
}: EdgeProps) {
  const d = data as CrossEdgeData | undefined;
  const diffType = d?.diffType ?? 'unchanged';
  const style = STYLE_MAP[diffType];

  const midX = (sourceX + targetX) / 2;
  const span = Math.abs(targetX - sourceX);
  const dipY = Math.max(sourceY, targetY) + 24 + span * 0.25;
  const path = `M ${sourceX} ${sourceY} Q ${midX} ${dipY} ${targetX} ${targetY}`;

  return (
    <g>
      {style.glow && (
        <defs>
          <filter id={`glow-${id}`}>
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      )}
      <path
        id={id}
        d={path}
        fill="none"
        stroke={style.stroke}
        strokeWidth={style.strokeWidth}
        strokeDasharray={style.dasharray}
        filter={style.glow ? `url(#glow-${id})` : undefined}
      />
    </g>
  );
}
