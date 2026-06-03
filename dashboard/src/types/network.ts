// Network Management types

export interface Network {
  readonly id: string;
  readonly name: string;
  readonly topologyDir: string;
  readonly npuPerServer: number;
  readonly npuType: string;
  readonly intraBw: string;
  readonly bandwidth: string;
  readonly serverIps: readonly string[];
  readonly createdAt: number;
  readonly updatedAt: number;
}

export interface NetworkFormData {
  readonly name: string;
  readonly topologyDir: string;
  readonly npuPerServer?: number;
  readonly npuType?: string;
  readonly intraBw?: string;
  readonly bandwidth?: string;
  readonly serverIps?: readonly string[];
}
