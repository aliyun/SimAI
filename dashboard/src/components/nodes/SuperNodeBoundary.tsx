import { memo } from 'react';
import type { NodeProps } from '@xyflow/react';

export interface SuperNodeBoundaryData {
  label: string;
  width: number;
  height: number;
  [key: string]: unknown;
}

function SuperNodeBoundaryComponent({ data }: NodeProps) {
  const nodeData = data as SuperNodeBoundaryData;
  return (
    <div
      style={{
        width: nodeData.width,
        height: nodeData.height,
        border: '1px dashed rgba(59, 130, 246, 0.3)',
        borderRadius: '8px',
        backgroundColor: 'rgba(59, 130, 246, 0.05)',
        position: 'relative',
        overflow: 'visible',
      }}
    >
      <span className="absolute left-1/2 -translate-x-1/2 text-xs text-blue-400/60 whitespace-nowrap" style={{ bottom: '-20px' }}>
        {nodeData.label}
      </span>
    </div>
  );
}

export const SuperNodeBoundary = memo(SuperNodeBoundaryComponent);
