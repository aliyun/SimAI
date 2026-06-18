import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { NodeType, NodeStatus } from '../../types/topology';
import { COLORS } from '../../theme/tokens';

export interface NetworkDeviceNodeData {
  label: string;
  nodeType: NodeType;
  nodeIp?: string;
  status: NodeStatus;
  serverType?: string;
  [key: string]: unknown;
}

const TYPE_STYLES: Record<string, { bg: string; icon: string; width: string }> = {
  OXC: { bg: COLORS.node.oxc, icon: 'OXC', width: 'w-[650px]' },
  SPINE: { bg: COLORS.node.spine, icon: 'SPN', width: 'w-[650px]' },
  LEAF: { bg: COLORS.node.leaf, icon: 'LF', width: 'w-56' },
  SERVER: { bg: COLORS.node.server, icon: 'SRV', width: 'w-56' },
};

const STATUS_COLORS: Record<NodeStatus, string> = {
  healthy: COLORS.status.healthy,
  warning: COLORS.status.warning,
  critical: COLORS.status.critical,
  unknown: COLORS.status.unknown,
};

function NetworkDeviceNodeComponent({ data, selected }: NodeProps) {
  const nodeData = data as NetworkDeviceNodeData;
  const style = TYPE_STYLES[nodeData.nodeType] || TYPE_STYLES.SERVER;

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0"
      />

      <div
        className={`${style.width} rounded-lg p-3 border transition-all duration-200 cursor-pointer ${selected ? 'ring-2 ring-blue-400' : ''}`}
        style={{
          backgroundColor: COLORS.bg.tertiary,
          borderColor: `${style.bg}60`,
          borderLeftWidth: '4px',
          borderLeftColor: style.bg,
        }}
      >
        <div className="flex items-center gap-2 mb-1 justify-center">
          <span
            className="text-xs font-bold px-1.5 py-0.5 rounded"
            style={{ backgroundColor: `${style.bg}30`, color: style.bg }}
          >
            {style.icon}
          </span>
          <span className="text-sm font-semibold text-white text-center whitespace-pre-line">{nodeData.label}</span>
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: STATUS_COLORS[nodeData.status] }}
          />
        </div>
        <div className="text-center">
          {nodeData.nodeIp && (
            <span className="text-xs text-gray-400">{nodeData.nodeIp}</span>
          )}
          {nodeData.serverType && (
            <span className="text-xs text-gray-500 ml-2">{nodeData.serverType}</span>
          )}
        </div>
      </div>
    </>
  );
}

export const NetworkDeviceNode = memo(NetworkDeviceNodeComponent);
