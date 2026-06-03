import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Network, NetworkFormData } from '../types/network';

interface NetworkState {
  readonly networks: readonly Network[];
  readonly activeNetworkId: string | null;

  createNetwork: (data: NetworkFormData) => Network;
  updateNetwork: (id: string, data: Partial<NetworkFormData>) => void;
  deleteNetwork: (id: string) => void;
  setActiveNetwork: (id: string | null) => void;
  getNetwork: (id: string) => Network | undefined;
}

export const useNetworkStore = create<NetworkState>()(
  persist(
    (set, get) => ({
      networks: [],
      activeNetworkId: null,

      createNetwork: (data) => {
        const now = Date.now();
        const network: Network = {
          id: `net_${now}`,
          name: data.name,
          topologyDir: data.topologyDir,
          npuPerServer: data.npuPerServer ?? 8,
          npuType: data.npuType ?? 'A3',
          intraBw: data.intraBw ?? '400Gbps',
          bandwidth: data.bandwidth ?? '',
          serverIps: data.serverIps ?? [],
          createdAt: now,
          updatedAt: now,
        };
        set((state) => ({ networks: [...state.networks, network] }));
        return network;
      },

      updateNetwork: (id, data) => {
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === id ? { ...n, ...data, updatedAt: Date.now() } : n,
          ),
        }));
      },

      deleteNetwork: (id) => {
        set((state) => ({
          networks: state.networks.filter((n) => n.id !== id),
          activeNetworkId: state.activeNetworkId === id ? null : state.activeNetworkId,
        }));
      },

      setActiveNetwork: (id) => set({ activeNetworkId: id }),

      getNetwork: (id) => get().networks.find((n) => n.id === id),
    }),
    { name: 'ocs-sim-networks' },
  ),
);
