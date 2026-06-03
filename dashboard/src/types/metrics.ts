import type { NodeStatus } from './topology';

export interface NodeLiveMetrics {
  readonly nodeId: string;
  readonly cpuUtilization: number;
  readonly npuUtilization: number;
  readonly memoryUsedGb: number;
  readonly memoryTotalGb: number;
  readonly temperature: number;
  readonly powerWatts: number;
  readonly status: NodeStatus;
  readonly timestamp: number;
}

export interface EdgeLiveMetrics {
  readonly edgeId: string;
  readonly txBytesPerSec: number;
  readonly rxBytesPerSec: number;
  readonly utilization: number;
  readonly packetDropRate: number;
  readonly errorRate: number;
  readonly timestamp: number;
}

export interface ClusterSummary {
  readonly totalPods: number;
  readonly totalServers: number;
  readonly totalNpus: number;
  readonly overallUtilization: number;
  readonly activeAlerts: number;
  readonly timestamp: number;
}
