// ---- Wizard types for the 4-step simulation workflow ----

export type WizardStep = 'workload' | 'edg' | 'launch' | 'results';

// ---- Workload ----

export interface ModelPreset {
  readonly hidden_size: number;
  readonly num_layers: number;
  readonly num_attention_heads: number;
  readonly vocab_size: number;
  readonly seq_length: number;
}

export interface WorkloadConfig {
  readonly model_size: string;
  readonly tp_size: number;
  readonly dp_size: number;
  readonly pp_size: number;
  readonly ep_size: number;
  readonly all_gpus: number;
  /** Gradient accumulation steps (default 8) */
  readonly ga: number;
  /** Virtual pipeline stages (default 1, set per-model) */
  readonly vpp: number;
}

export interface WorkloadLayer {
  readonly name: string;
  readonly fwd_comm_type: string;
  readonly fwd_comm_size: number;
  readonly bwd_comm_type: string;
  readonly bwd_comm_size: number;
}

export interface CustomLayerConfig {
  readonly name: string;
  readonly fwd_comm_type: string;
  readonly fwd_comm_size: number;
  readonly bwd_comm_type: string;
  readonly bwd_comm_size: number;
}

// ---- RankTable ----

export interface RankTableConfig {
  readonly rank_count: number;
  readonly gpus_per_rack: number;
  readonly num_racks?: number;
  readonly superpod_prefix: string;
}

export interface RankEntry {
  readonly rank_id: number;
  readonly device_id: number;
  readonly local_id: number;
  readonly level_list: readonly RankLevel[];
}

export interface RankLevel {
  readonly net_layer: number;
  readonly net_instance_id: string;
  readonly net_type: string;
  readonly net_attr: string;
  readonly rank_addr_list: readonly RankAddr[];
}

export interface RankAddr {
  readonly addr_type: string;
  readonly addr: string;
  readonly ports: readonly string[];
  readonly plane_id: string;
}

export interface RankTable {
  readonly version: string;
  readonly status: string;
  readonly rank_count: number;
  readonly rank_list: readonly RankEntry[];
}

export interface RankRackMap {
  readonly [rankId: string]: string;
}

// ---- Topology ----

export interface TopologyFileEntry {
  readonly name: string;
  readonly type: 'file' | 'directory';
  readonly path: string;
  readonly size?: number;
  readonly children?: readonly string[];
}

// ---- Launch ----

export type SimulationMode = 'analytical' | 'ns3';

/**
 * 集合通信模式：
 * - 'nccl'     : 标准 NCCL 算法，使用传统以太网 / InfiniBand 互联
 * - 'oxc-hccl' : OXC（光学电路交换）感知的 HCCL 算法，利用光交换网络的拓扑感知调度
 */
export type CollectiveMode = 'nccl' | 'oxc-hccl';

export interface LaunchConfig {
  readonly collectiveMode: CollectiveMode;
  readonly mode: SimulationMode;
  readonly workloadPath: string;
  readonly ranktablePath: string;
  readonly topologyPath: string;
  readonly threads: number;
  readonly envVars: Record<string, string>;
  /** OXC server URL (only used when collectiveMode is 'oxc-hccl') */
  readonly oxcUrl: string;
  /** OXC scheduling algorithm (only used when collectiveMode is 'oxc-hccl') */
  readonly oxcAlgo: string;
  /** Compute-communication overlap ratio [0.0-1.0] */
  readonly tpOverlap: number;
  readonly dpOverlap: number;
  readonly epOverlap: number;
  readonly ppOverlap: number;
  /** 服务器内带宽 bandwidth GB/s (analytical mode) */
  readonly intraBw: number;
  /** NIC bandwidth GB/s (analytical mode) */
  readonly nicBw: number;
  /** NPU type for bandwidth lookup */
  readonly npuType: string;
  /** NIC type for bandwidth lookup */
  readonly nicType: string;
}

export type SimulationStatus = 'idle' | 'running' | 'completed' | 'failed';

// ---- Results ----

export interface LayerResult {
  readonly layer_name: string;
  readonly run_name: string;
  readonly fwd_compute: number | null;
  readonly wg_compute: number | null;
  readonly ig_compute: number | null;
  readonly fwd_exposed_comm: number | null;
  readonly wg_exposed_comm: number | null;
  readonly ig_exposed_comm: number | null;
  readonly fwd_total_comm: number | null;
  readonly fwd_algbw: number | null;
  readonly fwd_busbw: number | null;
  readonly wg_total_comm: number | null;
  readonly wg_algbw: number | null;
  readonly wg_busbw: number | null;
  readonly ig_total_comm: number | null;
  readonly ig_algbw: number | null;
  readonly ig_busbw: number | null;
  readonly workload_finished_at: number | null;
  // Comm group (TP/DP/EP/DP_EP) derived from workload file comm_type
  readonly fwd_group?: string;
  readonly wg_group?: string;
  readonly ig_group?: string;
}

export interface DimensionEntry {
  readonly value: number;
  readonly percentage: number;
}

export interface EndToEndData {
  readonly layers: readonly LayerResult[];
  readonly summary: Record<string, unknown>;
  readonly totals: {
    readonly total_exposed?: number | null;
    readonly total_compute?: number | null;
    readonly total_time?: number | null;
    readonly bubble_time?: number | null;
  };
  readonly dimensions: Record<string, DimensionEntry>;
  readonly run_name: string;
}

export interface NodeTransfer {
  readonly id: number;
  readonly sent: number;
  readonly received: number;
}

export interface ConsoleData {
  readonly finish_time: number | null;
  readonly streams_injected: number | null;
  readonly streams_finished: number | null;
  readonly nodes: readonly NodeTransfer[];
}
