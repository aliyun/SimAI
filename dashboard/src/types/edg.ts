// --- EDG Types ---

export interface EdgInitResponse {
  readonly lld_path: string;
  readonly crosses_count: number;
  readonly oxc_count: number;
  readonly warning?: string | null;
}

export interface EdgGraphStats {
  readonly servers: number;
  readonly leaves: number;
  readonly leaf_leaf_edges: number;
  readonly server_leaf_edges: number;
}

// --- Graph data for topology visualization ---

export interface EdgServerNode {
  readonly ip: string;
  readonly node_id: string;
  readonly server_type?: string;
  readonly leaf_ip?: string;
}

export interface EdgLeafNode {
  readonly ip: string;
  readonly node_id: string;
}

export type EdgLeafLeafEdge = readonly [string, string, string, string, string];
export type EdgServerLeafEdge = readonly [string, string, number];

export interface EdgGraph {
  readonly servers: readonly EdgServerNode[];
  readonly leaves: readonly EdgLeafNode[];
  readonly server_leaf_edges: readonly EdgServerLeafEdge[];
  readonly leaf_leaf_edges: readonly EdgLeafLeafEdge[];
  // oxc_ip -> { port -> leaf_ip }. Every OXC port physically connected to a
  // participating leaf, whether or not it currently has an internal cross.
  readonly oxc_port_map?: Readonly<Record<string, Readonly<Record<string, string>>>>;
}

export interface EdgDiff {
  readonly added: readonly EdgLeafLeafEdge[];
  readonly removed: readonly EdgLeafLeafEdge[];
  readonly unchanged: readonly EdgLeafLeafEdge[];
}

export interface EdgBaselineGraphResponse {
  readonly graph: EdgGraph;
  readonly crosses_count: number;
}

export interface EdgTaskResponse {
  readonly task_id: string;
  readonly topology_file: string;
  readonly graph_stats: EdgGraphStats;
  readonly graph: EdgGraph;
  readonly diff: EdgDiff;
  readonly ranktable_path: string;
  readonly warning?: string | null;
}
