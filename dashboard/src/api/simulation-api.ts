import apiClient from './client';
import type {
  WorkloadConfig,
  WorkloadLayer,
  CustomLayerConfig,
  RankTableConfig,
  RankTable,
  RankRackMap,
  ModelPreset,
  EndToEndData,
  ConsoleData,
  TopologyFileEntry,
} from '../types/wizard';

// ---- Workload ----

interface ModelInfo {
  readonly id: string;
  readonly num_layers: number;
  readonly hidden_size: number;
  readonly num_experts: number;
}

interface ModelFamiliesResponse {
  readonly families: Record<string, readonly ModelInfo[]>;
}

export async function fetchModelFamilies(): Promise<Record<string, readonly ModelInfo[]>> {
  const { data } = await apiClient.get<ModelFamiliesResponse>(
    '/api/simulation/workload/models',
  );
  return data.families;
}

interface GeneratePresetResponse {
  readonly content: string;
  readonly layers: readonly WorkloadLayer[];
  readonly model_size: string;
}

export async function generatePresetWorkload(
  config: WorkloadConfig,
  aiob_enable = true,
): Promise<GeneratePresetResponse> {
  const { data } = await apiClient.post<GeneratePresetResponse>(
    '/api/simulation/workload/generate-preset',
    { ...config, aiob_enable, ga: config.ga, vpp: config.vpp },
  );
  return data;
}

interface GenerateCustomResponse {
  readonly content: string;
}

export async function generateCustomWorkload(params: {
  tp_size: number;
  dp_size: number;
  pp_size: number;
  ep_size: number;
  all_gpus: number;
  ga: number;
  vpp: number;
  layers_config: readonly CustomLayerConfig[];
}): Promise<GenerateCustomResponse> {
  const { data } = await apiClient.post<GenerateCustomResponse>(
    '/api/simulation/workload/generate-custom',
    params,
  );
  return data;
}

interface ParseWorkloadResponse {
  readonly config: Record<string, unknown>;
}

export async function parseWorkload(content: string): Promise<ParseWorkloadResponse> {
  const { data } = await apiClient.post<ParseWorkloadResponse>(
    '/api/simulation/workload/parse',
    { content },
  );
  return data;
}

interface PresetsResponse {
  readonly presets: Record<string, ModelPreset>;
}

export async function fetchWorkloadPresets(): Promise<Record<string, ModelPreset>> {
  const { data } = await apiClient.get<PresetsResponse>('/api/simulation/workload/presets');
  return data.presets;
}

export async function generateTimelinePreview(params: {
  content: string;
  maxLayers?: number;
  rank?: number;
}): Promise<string> {
  const { data } = await apiClient.post<{ html: string }>(
    '/api/simulation/workload/timeline-preview',
    {
      content: params.content,
      max_layers: params.maxLayers ?? 0,
      rank: params.rank ?? 0,
    },
  );
  return data.html;
}

// ---- RankTable ----

interface GenerateRanktableResponse {
  readonly ranktable: RankTable;
  readonly rank_rack_map: RankRackMap;
}

export async function generateRanktable(config: RankTableConfig): Promise<GenerateRanktableResponse> {
  const { data } = await apiClient.post<GenerateRanktableResponse>(
    '/api/simulation/ranktable/generate',
    config,
  );
  return data;
}

export async function generateCustomRanktable(params: {
  rank_count: number;
  topology: ReadonlyArray<{ rack_id: number; ranks: readonly number[]; superpod?: string }>;
}): Promise<GenerateRanktableResponse> {
  const { data } = await apiClient.post<GenerateRanktableResponse>(
    '/api/simulation/ranktable/generate-custom',
    params,
  );
  return data;
}

interface ValidateResponse {
  readonly valid: boolean;
  readonly errors: readonly string[];
}

export async function validateRanktable(ranktable: RankTable): Promise<ValidateResponse> {
  const { data } = await apiClient.post<ValidateResponse>(
    '/api/simulation/ranktable/validate',
    { ranktable },
  );
  return data;
}

// ---- Topology Dir ----

interface TopologyDirResponse {
  readonly path: string;
}

export async function setTopologyDir(path: string): Promise<TopologyDirResponse> {
  const { data } = await apiClient.post<TopologyDirResponse>(
    '/api/simulation/topology-dir',
    { path },
  );
  return data;
}

export async function getTopologyDir(): Promise<string> {
  const { data } = await apiClient.get<TopologyDirResponse>('/api/simulation/topology-dir');
  return data.path;
}

interface ScanResponse {
  readonly path: string;
  readonly files: readonly TopologyFileEntry[];
}

export async function scanTopologyDir(): Promise<ScanResponse> {
  const { data } = await apiClient.get<ScanResponse>('/api/simulation/topology-dir/scan');
  return data;
}

// ---- Results ----

export async function parseEndToEnd(content: string): Promise<EndToEndData> {
  const { data } = await apiClient.post<EndToEndData>(
    '/api/simulation/results/parse-endtoend',
    { content },
  );
  return data;
}

export async function parseConsoleOutput(logLines: readonly string[]): Promise<ConsoleData> {
  const { data } = await apiClient.post<ConsoleData>(
    '/api/simulation/results/parse-console',
    { log_lines: logLines },
  );
  return data;
}

interface FindFilesResponse {
  readonly files: Record<string, string>;
}

export async function findResultFiles(path?: string): Promise<Record<string, string>> {
  const params = path ? { path } : {};
  const { data } = await apiClient.get<FindFilesResponse>(
    '/api/simulation/results/find-files',
    { params },
  );
  return data.files;
}

// ---- Completed Tasks with Results ----

export interface CompletedTask {
  readonly tracking_id: number | null;
  readonly pid: number | null;
  readonly command: string | null;
  readonly status: string;
  readonly started_at: number;
  readonly finished_at: number;
  readonly return_code: number | null;
  readonly error_message: string | null;
  readonly result_prefix: string;
  readonly workload_path?: string;
  readonly result_files: Record<string, string>;
}

interface ListTasksResponse {
  readonly tasks: readonly CompletedTask[];
}

export interface SimProgress {
  readonly pid: string;
  readonly status: string;
  readonly total_layers: number;
  readonly current_layer: number;
  readonly pct: number;
  readonly elapsed_sec: number;
  readonly estimated_remaining_sec: number;
}

export async function fetchProgress(pid: number): Promise<SimProgress | null> {
  try {
    const { data } = await apiClient.get<SimProgress>(
      `/api/simulation/results/progress/${pid}`,
    );
    return data;
  } catch {
    return null;
  }
}

export async function fetchCompletedTasks(): Promise<readonly CompletedTask[]> {
  const { data } = await apiClient.get<ListTasksResponse>(
    '/api/simulation/results/list-tasks',
  );
  return data.tasks;
}

export async function loadResultByFilepath(filepath: string, workloadPath?: string): Promise<EndToEndData> {
  const { data } = await apiClient.post<EndToEndData>(
    '/api/simulation/results/parse-endtoend',
    { filepath, workload_path: workloadPath },
  );
  return data;
}

// ---- Files (reuse existing /api/files) ----

interface SaveFileResponse {
  readonly path: string;
  readonly filename: string;
}

export async function saveFile(filename: string, content: string): Promise<SaveFileResponse> {
  const { data } = await apiClient.post<SaveFileResponse>('/api/files/save', {
    filename,
    content,
  });
  return data;
}

interface LoadFileResponse {
  readonly content: string;
  readonly path: string;
  readonly filename: string;
}

export async function loadFile(filename: string): Promise<LoadFileResponse> {
  const { data } = await apiClient.get<LoadFileResponse>('/api/files/load', {
    params: { filename },
  });
  return data;
}

// ---- Process (reuse existing /api/process) ----

interface LaunchResponse {
  readonly tracking_id: string;
  readonly pid: number;
  readonly status: string;
}

export async function launchSimulation(params: Record<string, unknown>): Promise<LaunchResponse> {
  const { data } = await apiClient.post<LaunchResponse>('/api/process/launch', params);
  return data;
}

interface ProcessEntry {
  readonly pid: number;
  readonly status: string;
  readonly command: string;
  readonly started_at: string;
}

interface ListProcessesResponse {
  readonly processes: readonly ProcessEntry[];
}

export async function listProcesses(): Promise<readonly ProcessEntry[]> {
  const { data } = await apiClient.get<ListProcessesResponse>('/api/process/list');
  return data.processes;
}

export async function fetchProcessStatus(pid: number): Promise<ProcessEntry | null> {
  const { data } = await apiClient.get<ProcessEntry>('/api/process/' + pid);
  return data;
}

interface LogsResponse {
  readonly logs: readonly string[];
  readonly status: string;
  readonly pid: number;
}

export async function fetchProcessLogs(pid: number): Promise<LogsResponse> {
  const { data } = await apiClient.get<LogsResponse>(`/api/process/logs/${pid}`);
  return data;
}

export async function killProcess(pid: number): Promise<void> {
  await apiClient.delete(`/api/process/${pid}`);
}
