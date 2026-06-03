import apiClient from './client';
import type { EdgInitResponse, EdgTaskResponse, EdgBaselineGraphResponse } from '../types/edg';

export async function initNetwork(lld: Record<string, unknown>, topologyDir?: string): Promise<EdgInitResponse> {
  const { data } = await apiClient.post<EdgInitResponse>('/api/edg/init', {
    lld,
    topology_dir: topologyDir,
  });
  return data;
}

export interface EdgTopoParams {
  readonly npu_per_server?: string;
  readonly npu_type?: string;
  readonly intra_bw?: string;
  readonly bandwidth?: string;
}

export async function fetchBaselineGraph(
  serverIps: readonly string[],
  npuPerServer: number,
): Promise<EdgBaselineGraphResponse> {
  const { data } = await apiClient.post<EdgBaselineGraphResponse>('/api/edg/baseline-graph', {
    server_ips: [...serverIps],
    npu_per_server: npuPerServer,
  });
  return data;
}

export async function registerTask(
  npuMatch: Record<string, unknown>,
  taskId: string,
  topoParams?: EdgTopoParams,
): Promise<EdgTaskResponse> {
  const { data } = await apiClient.post<EdgTaskResponse>('/api/edg/register-task', {
    npu_match: npuMatch,
    task_id: taskId,
    topo_params: topoParams,
  });
  return data;
}
