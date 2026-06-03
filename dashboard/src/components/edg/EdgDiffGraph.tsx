import { useMemo } from 'react';
import { ReactFlow, Background, Controls } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { DiffEdge } from './DiffEdge';
import { CrossEdge } from './CrossEdge';
import { buildEdgFlowElements } from './edg-layout';
import type { EdgGraph, EdgDiff } from '../../types/edg';

const EDGE_TYPES = { diffEdge: DiffEdge, crossEdge: CrossEdge };

interface EdgDiffGraphProps {
  readonly baselineGraph: EdgGraph;
  readonly adjustedGraph: EdgGraph | null;
  readonly diff: EdgDiff | null;
  readonly participatingServerIps?: ReadonlySet<string>;
  readonly height?: string;
}

export function EdgDiffGraph({ baselineGraph, adjustedGraph, diff, participatingServerIps, height = '420px' }: EdgDiffGraphProps) {
  const showComparison = adjustedGraph !== null && diff !== null;

  const baselineElements = useMemo(
    () => buildEdgFlowElements(baselineGraph, null, participatingServerIps),
    [baselineGraph, participatingServerIps],
  );

  const adjustedElements = useMemo(
    () => showComparison ? buildEdgFlowElements(baselineGraph, diff, participatingServerIps) : null,
    [baselineGraph, diff, participatingServerIps, showComparison],
  );

  if (showComparison && adjustedElements) {
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-3">
          <GraphPanel title="调节前" elements={baselineElements} height={height} />
          <GraphPanel title="调节后" elements={adjustedElements} height={height} />
        </div>
        <Legend diff={diff} />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <GraphPanel title="当前拓扑" elements={baselineElements} height={height} showControls />
    </div>
  );
}

function GraphPanel({
  title, elements, height, showControls = false,
}: {
  title: string;
  elements: { nodes: ReturnType<typeof buildEdgFlowElements>['nodes']; edges: ReturnType<typeof buildEdgFlowElements>['edges'] };
  height: string;
  showControls?: boolean;
}) {
  return (
    <div className="rounded-lg border border-[var(--color-bg-hover)] overflow-hidden">
      <div className="px-3 py-1.5 bg-[var(--color-bg-tertiary)] border-b border-[var(--color-bg-hover)]">
        <span className="text-xs font-medium text-[var(--color-text-secondary)]">{title}</span>
      </div>
      <div style={{ height }}>
        <ReactFlow
          nodes={elements.nodes}
          edges={elements.edges}
          edgeTypes={EDGE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.12 }}
          nodesDraggable={false}
          nodesConnectable={false}
          proOptions={{ hideAttribution: true }}
          minZoom={0.15}
          maxZoom={2}
        >
          <Background color="#1f2937" gap={24} />
          {showControls && <Controls />}
        </ReactFlow>
      </div>
    </div>
  );
}

function Legend({ diff }: { diff: EdgDiff | null }) {
  if (!diff) return null;

  const items = [
    { color: '#10b981', label: `新增连接 (${diff.added.length})`, dash: false },
    { color: '#ef4444', label: `删除连接 (${diff.removed.length})`, dash: true },
    { color: '#4b5563', label: `不变 (${diff.unchanged.length})`, dash: false },
  ];

  return (
    <div className="flex items-center justify-center gap-6 px-3 py-2 rounded-lg bg-[var(--color-bg-secondary)] border border-[var(--color-bg-hover)]">
      {items.map((item) => (
        <div key={item.color} className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
          <svg width="24" height="4">
            <line
              x1="0" y1="2" x2="24" y2="2"
              stroke={item.color}
              strokeWidth={item.dash ? 2 : 3}
              strokeDasharray={item.dash ? '4 3' : undefined}
            />
          </svg>
          {item.label}
        </div>
      ))}
      <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
        <span className="inline-block w-3 h-3 rounded border-2 border-[#10b981] bg-[#0e3a2d]" />
        参与调节的服务器
      </div>
    </div>
  );
}
