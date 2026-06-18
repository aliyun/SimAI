// --- Node Types ---

export type NodeType = 'POD' | 'OXC' | 'SPINE' | 'LEAF' | 'SERVER' | 'SUPER_NODE';

export type NodeStatus = 'healthy' | 'warning' | 'critical' | 'unknown';

export interface Position {
  readonly x: number;
  readonly y: number;
}

export interface NodeMetadata {
  readonly nodeIp?: string;
  readonly serverType?: string;
  readonly superNodeId?: string;
  readonly paraPlaneId?: string;
}

export interface NodeSize {
  readonly width: number;
  readonly height: number;
}

export interface TopologyNode {
  readonly id: string;
  readonly type: NodeType;
  readonly label: string;
  readonly position: Position;
  readonly size?: NodeSize;
  readonly metadata: NodeMetadata;
}

// --- Edge Types ---

export interface PortMapping {
  readonly sourcePorts: readonly number[];
  readonly targetPorts: readonly number[];
}

export interface EdgeMetrics {
  readonly utilization: number;
  readonly linkCount: number;
  readonly errorRate: number;
  readonly bandwidthGbps?: number;
  readonly latencyUs?: number;
  /** Uplink error rate range (source→target), e.g. "0.01%~0.08%" */
  readonly uplinkErrorRange?: string;
  /** Downlink error rate range (target→source), e.g. "0.01%~0.08%" */
  readonly downlinkErrorRange?: string;
}

export interface TopologyEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly portMapping?: PortMapping;
  readonly metrics: EdgeMetrics;
  /** Per-side metrics: label near source node (overview) */
  readonly sourceMetrics?: EdgeMetrics;
  /** Per-side metrics: label near target node (overview) */
  readonly targetMetrics?: EdgeMetrics;
  /** Detailed per-port data from link rate CSV */
  readonly portDetails?: readonly PortDetail[];
}

export interface PortDetail {
  readonly edge_id: string;
  readonly lower_node: string;
  readonly upper_node: string;
  readonly port: string;
  readonly in_rate: string;
  readonly out_rate: string;
}

// --- Topology Containers ---

export interface BoundingBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface SuperNodeGroup {
  readonly id: string;
  readonly serverIds: readonly string[];
  readonly boundingBox: BoundingBox;
}

export interface PodOverviewTopology {
  readonly nodes: readonly TopologyNode[];
  readonly edges: readonly TopologyEdge[];
}

export interface PodDetailTopology {
  readonly podId: string;
  readonly paraPlaneId: string;
  readonly nodes: readonly TopologyNode[];
  readonly edges: readonly TopologyEdge[];
  readonly superNodes: readonly SuperNodeGroup[];
}
