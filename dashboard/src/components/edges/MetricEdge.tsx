import { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getStraightPath,
  type EdgeProps,
} from '@xyflow/react';
import type { EdgeMetrics, PortMapping, PortDetail } from '../../types/topology';

export interface MetricEdgeData {
  metrics: EdgeMetrics;
  sourceMetrics?: EdgeMetrics;
  targetMetrics?: EdgeMetrics;
  portMapping?: PortMapping;
  bandwidthGbps?: number;
  latencyUs?: number;
  portDetails?: readonly PortDetail[];
  [key: string]: unknown;
}

function getEdgeColor(util: number): string {
  if (util > 0.008) return '#ef4444';
  if (util > 0.005) return '#f59e0b';
  return '#4b5563';
}

function formatPorts(ports: readonly number[]): string {
  return ports.join(', ');
}

function EdgeTooltip({
  anchorRef,
  bandwidthGbps,
  latencyUs,
  linkCount,
  portMapping,
}: {
  anchorRef: React.RefObject<HTMLDivElement | null>;
  bandwidthGbps?: number;
  latencyUs?: number;
  linkCount: number;
  portMapping?: PortMapping;
}) {
  const rect = anchorRef.current?.getBoundingClientRect();
  if (!rect) return null;

  return createPortal(
    <div
      style={{
        position: 'fixed',
        left: rect.left + rect.width / 2,
        top: rect.bottom + 4,
        transform: 'translateX(-50%)',
        zIndex: 99999,
      }}
      className="bg-[#0f172a] border border-blue-500/30 rounded-lg px-3 py-2 shadow-lg shadow-blue-900/20 w-max pointer-events-none"
    >
      <div className="text-[10px] font-semibold text-blue-400 mb-1.5 border-b border-gray-700 pb-1">
        Link Detail
      </div>
      <table className="text-[10px] w-full">
        <tbody>
          {bandwidthGbps != null && (
            <tr>
              <td className="text-gray-500 pr-3 py-0.5">Bandwidth</td>
              <td className="text-gray-200 text-right whitespace-nowrap">{bandwidthGbps} Gbps</td>
            </tr>
          )}
          {latencyUs != null && (
            <tr>
              <td className="text-gray-500 pr-3 py-0.5 whitespace-nowrap">Latency</td>
              <td className="text-gray-200 text-right whitespace-nowrap">{latencyUs.toFixed(1)} us</td>
            </tr>
          )}
          <tr>
            <td className="text-gray-500 pr-3 py-0.5 whitespace-nowrap">Links</td>
            <td className="text-gray-200 text-right whitespace-nowrap">{linkCount}</td>
          </tr>
          {portMapping && (
            <>
              <tr>
                <td className="text-gray-500 pr-3 py-0.5 whitespace-nowrap">Src Ports</td>
                <td className="text-gray-200 text-right whitespace-nowrap">{formatPorts(portMapping.sourcePorts)}</td>
              </tr>
              <tr>
                <td className="text-gray-500 pr-3 py-0.5 whitespace-nowrap">Dst Ports</td>
                <td className="text-gray-200 text-right whitespace-nowrap">{formatPorts(portMapping.targetPorts)}</td>
              </tr>
            </>
          )}
        </tbody>
      </table>
    </div>,
    document.body,
  );
}

/** Interpolate along the edge. t=0 → source, t=1 → target. */
function lerp(sx: number, sy: number, tx: number, ty: number, t: number) {
  return { x: sx + (tx - sx) * t, y: sy + (ty - sy) * t };
}

export function MetricEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
  selected,
}: EdgeProps) {
  const [hovered, setHovered] = useState(false);
  const labelRef = useRef<HTMLDivElement>(null);
  const edgeData = data as MetricEdgeData | undefined;
  const [edgePath] = getStraightPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });

  const metrics = edgeData?.metrics;
  const hasMetrics = metrics != null && (metrics.utilization > 0 || metrics.errorRate > 0 || metrics.linkCount > 1
    || metrics.uplinkErrorRange || metrics.downlinkErrorRange);
  const linkCount = metrics?.linkCount || 1;
  const uplinkLabel = metrics?.uplinkErrorRange;
  const downlinkLabel = metrics?.downlinkErrorRange;
  const edgeColor = hasMetrics ? getEdgeColor(metrics!.utilization) : '#4b5563';

  const bandwidthGbps = edgeData?.bandwidthGbps ?? metrics?.bandwidthGbps;
  const latencyUs = edgeData?.latencyUs ?? metrics?.latencyUs;
  const portMapping = edgeData?.portMapping;
  const sourceMetrics = edgeData?.sourceMetrics;
  const targetMetrics = edgeData?.targetMetrics;

  const hasDualLabels = !!(uplinkLabel || downlinkLabel);

  // Fixed relative label positions: 30% near source, 60% near target.
  // Consistent across all edge types and lengths.
  const srcLabelPos = lerp(sourceX, sourceY, targetX, targetY, 0.30);
  const tgtLabelPos = lerp(sourceX, sourceY, targetX, targetY, 0.70);

  // For dual labels: "up" label at position closer to top (smaller Y),
  // "down" label at position closer to bottom (larger Y).
  const srcIsTop = srcLabelPos.y < tgtLabelPos.y;
  const topLabelX = srcIsTop ? srcLabelPos.x : tgtLabelPos.x;
  const topLabelY = srcIsTop ? srcLabelPos.y : tgtLabelPos.y;
  const bottomLabelX = srcIsTop ? tgtLabelPos.x : srcLabelPos.x;
  const bottomLabelY = srcIsTop ? tgtLabelPos.y : srcLabelPos.y;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: edgeColor,
          strokeWidth: selected ? 3 : 2,
          filter: selected ? `drop-shadow(0 0 6px ${edgeColor})` : undefined,
          transition: 'stroke 0.6s ease',
        }}
      />
      {hasMetrics && (
        <>
          <EdgeLabelRenderer>
            {hasDualLabels ? (
              <>
                {/* Uplink label at top (smaller Y) */}
                {uplinkLabel && (
                  <div
                    ref={labelRef}
                    style={{
                      position: 'absolute',
                      transform: `translate(-50%, -50%) translate(${topLabelX}px,${topLabelY}px)`,
                      pointerEvents: 'all',
                      zIndex: 1000,
                    }}
                    onMouseEnter={() => setHovered(true)}
                    onMouseLeave={() => setHovered(false)}
                  >
                    <div className="bg-[#111827] border border-gray-700 rounded px-1.5 py-0.5 text-[10px] text-gray-300 tabular-nums whitespace-nowrap">
                      {uplinkLabel}
                    </div>
                  </div>
                )}
                {/* Downlink label at bottom (larger Y) */}
                {downlinkLabel && (
                  <div
                    style={{
                      position: 'absolute',
                      transform: `translate(-50%, -50%) translate(${bottomLabelX}px,${bottomLabelY}px)`,
                      pointerEvents: 'all',
                      zIndex: 1000,
                    }}
                    onMouseEnter={() => setHovered(true)}
                    onMouseLeave={() => setHovered(false)}
                  >
                    <div className="bg-[#111827] border border-gray-700 rounded px-1.5 py-0.5 text-[10px] text-gray-300 tabular-nums whitespace-nowrap">
                      {downlinkLabel}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <>
                {/* Legacy format: per-side labels near source and target */}
                <div
                  ref={labelRef}
                  style={{
                    position: 'absolute',
                    transform: `translate(-50%, -50%) translate(${srcLabelPos.x}px,${srcLabelPos.y}px)`,
                    pointerEvents: 'all',
                  }}
                  onMouseEnter={() => setHovered(true)}
                  onMouseLeave={() => setHovered(false)}
                >
                  <div className="bg-[#111827] border border-gray-700 rounded px-1.5 py-0.5 text-center">
                    <div className="text-[10px] text-white tabular-nums">{((sourceMetrics?.utilization ?? metrics!.utilization) * 100).toFixed(2)}%</div>
                    <div className="text-[9px] text-gray-400">x{sourceMetrics?.linkCount ?? linkCount}</div>
                    <div className="text-[10px] text-gray-300 tabular-nums">{((sourceMetrics?.errorRate ?? metrics!.errorRate) * 100).toFixed(2)}%</div>
                  </div>
                </div>
                <div
                  style={{
                    position: 'absolute',
                    transform: `translate(-50%, -50%) translate(${tgtLabelPos.x}px,${tgtLabelPos.y}px)`,
                    pointerEvents: 'all',
                  }}
                  onMouseEnter={() => setHovered(true)}
                  onMouseLeave={() => setHovered(false)}
                >
                  <div className="bg-[#111827] border border-gray-700 rounded px-1.5 py-0.5 text-center">
                    <div className="text-[10px] text-white tabular-nums">{((targetMetrics?.utilization ?? metrics!.utilization) * 100).toFixed(2)}%</div>
                    <div className="text-[9px] text-gray-400">x{targetMetrics?.linkCount ?? linkCount}</div>
                    <div className="text-[10px] text-gray-300 tabular-nums">{((targetMetrics?.errorRate ?? metrics!.errorRate) * 100).toFixed(2)}%</div>
                  </div>
                </div>
              </>
            )}
          </EdgeLabelRenderer>

          {hovered && (
            <EdgeTooltip
              anchorRef={labelRef}
              bandwidthGbps={bandwidthGbps}
              latencyUs={latencyUs}
              linkCount={linkCount}
              portMapping={portMapping}
            />
          )}
        </>
      )}
    </>
  );
}
