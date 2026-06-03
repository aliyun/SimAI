import apiClient from './client';
import type { PodOverviewTopology, PodDetailTopology } from '../types/topology';
import type { ClusterSummary, NodeLiveMetrics } from '../types/metrics';
import type { ApiResponse } from '../types/api';

export async function fetchOverviewTopology(): Promise<PodOverviewTopology> {
  const { data } = await apiClient.get<ApiResponse<PodOverviewTopology>>('/api/topology/overview');
  if (data.error || !data.data) {
    throw new Error(data.error || 'No topology data');
  }
  return data.data;
}

export async function fetchPodDetail(podId: string): Promise<PodDetailTopology> {
  const { data } = await apiClient.get<ApiResponse<PodDetailTopology>>(
    `/api/topology/pod/${encodeURIComponent(podId)}`,
  );
  if (data.error || !data.data) {
    throw new Error(data.error || 'No pod detail data');
  }
  return data.data;
}

export async function fetchClusterSummary(): Promise<ClusterSummary> {
  const { data } = await apiClient.get<ApiResponse<ClusterSummary>>('/api/metrics/cluster');
  if (data.error || !data.data) {
    throw new Error(data.error || 'No cluster summary');
  }
  return data.data;
}

export async function fetchNodeMetrics(nodeId: string): Promise<NodeLiveMetrics> {
  const { data } = await apiClient.get<ApiResponse<NodeLiveMetrics>>(
    `/api/metrics/node/${encodeURIComponent(nodeId)}`,
  );
  if (data.error || !data.data) {
    throw new Error(data.error || 'No node metrics');
  }
  return data.data;
}
