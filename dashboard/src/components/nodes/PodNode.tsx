import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { COLORS } from '../../theme/tokens';

export interface PodNodeData {
  label: string;
  status: 'healthy' | 'warning' | 'critical';
  npuCount: number;
  utilization: number;
  [key: string]: unknown;
}

function PodNodeComponent({ data, selected }: NodeProps) {
  const nodeData = data as PodNodeData;
  const statusColors: Record<string, string> = {
    healthy: COLORS.status.healthy,
    warning: COLORS.status.warning,
    critical: COLORS.status.critical,
  };
  const ringColor = statusColors[nodeData.status] || COLORS.status.unknown;
  const utilPercent = (nodeData.utilization * 100).toFixed(1);

  return (
    <>
      <Handle
        type="source"
        position={Position.Top}
        className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0"
        style={{ left: '50%', top: '50%' }}
      />
      <Handle
        type="target"
        position={Position.Top}
        id="top-tgt"
        className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0"
        style={{ left: '50%', top: '50%' }}
      />

      <div
        className={`w-32 h-32 rounded-full flex flex-col items-center justify-center cursor-pointer transition-all duration-200 ${selected ? 'ring-2 ring-blue-400 ring-offset-2 ring-offset-[#0a0e17]' : ''}`}
        style={{
          background: 'radial-gradient(circle at 30% 30%, #1f2937, #111827)',
          border: `3px solid ${ringColor}`,
          boxShadow: `0 0 20px ${ringColor}40`,
        }}
      >
        <span className="text-sm font-bold text-white">{nodeData.label}</span>
        <span className="text-xs text-gray-400 mt-1">{nodeData.npuCount} NPUs</span>
        <span className="text-lg font-bold mt-1" style={{ color: ringColor }}>
          {utilPercent}%
        </span>
      </div>
    </>
  );
}

export const PodNode = memo(PodNodeComponent);
