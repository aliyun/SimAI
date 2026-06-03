import { create } from 'zustand';
import type { PodOverviewTopology, PodDetailTopology } from '../types/topology';
import type { ClusterSummary } from '../types/metrics';

interface TopologyState {
  readonly overview: PodOverviewTopology | null;
  readonly podDetail: PodDetailTopology | null;
  readonly clusterSummary: ClusterSummary | null;
  readonly selectedNodeId: string | null;
  readonly selectedEdgeId: string | null;
  readonly isLoading: boolean;
  readonly error: string | null;
  readonly lastUpdated: number | null;

  setOverview: (topology: PodOverviewTopology) => void;
  setPodDetail: (detail: PodDetailTopology) => void;
  setClusterSummary: (summary: ClusterSummary) => void;
  selectNode: (nodeId: string | null) => void;
  selectEdge: (edgeId: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useTopologyStore = create<TopologyState>((set) => ({
  overview: null,
  podDetail: null,
  clusterSummary: null,
  selectedNodeId: null,
  selectedEdgeId: null,
  isLoading: false,
  error: null,
  lastUpdated: null,

  setOverview: (topology) => set({ overview: topology, lastUpdated: Date.now(), error: null }),
  setPodDetail: (detail) => set({ podDetail: detail, lastUpdated: Date.now(), error: null }),
  setClusterSummary: (summary) => set({ clusterSummary: summary }),
  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),
  selectEdge: (edgeId) => set({ selectedEdgeId: edgeId }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error, isLoading: false }),
}));
