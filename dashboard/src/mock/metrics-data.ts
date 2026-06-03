import type { NodeLiveMetrics, ClusterSummary } from '../types/metrics';
import type { NodeStatus } from '../types/topology';
import { MOCK_POD_COUNT } from './topology-data';

const SERVERS_PER_POD = 2;
const NPUS_PER_SERVER = 8;

export function generateClusterSummary(): ClusterSummary {
  return {
    totalPods: MOCK_POD_COUNT,
    totalServers: MOCK_POD_COUNT * SERVERS_PER_POD,
    totalNpus: MOCK_POD_COUNT * SERVERS_PER_POD * NPUS_PER_SERVER,
    overallUtilization: 0.68 + Math.random() * 0.1,
    activeAlerts: Math.floor(Math.random() * 5),
    timestamp: Date.now(),
  };
}

export function generateNodeMetrics(nodeId: string): NodeLiveMetrics {
  const status: NodeStatus = Math.random() > 0.95 ? 'warning' : 'healthy';
  return {
    nodeId,
    cpuUtilization: 0.3 + Math.random() * 0.5,
    gpuUtilization: 0.5 + Math.random() * 0.4,
    memoryUsedGb: 60 + Math.random() * 20,
    memoryTotalGb: 80,
    temperature: 55 + Math.random() * 25,
    powerWatts: 200 + Math.random() * 300,
    status,
    timestamp: Date.now(),
  };
}

export function jitterMetrics(baseUtil: number, jitter: number = 0.001): number {
  return Math.max(0, Math.min(1, baseUtil + (Math.random() - 0.5) * jitter));
}
