import numpy as np
import torch

from vidur.profiling.collectives.collectives_impl import GraphedCollective
from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.timer_stats_store import TimerStatsStore

WARMUP_STEPS = 1
ACTIVE_STEPS = 3
GRAPH_DISABLED_STEPS = 10
DISABLE_GRAPH = True


class CollectiveWrapper:
    def __init__(
        self,
        rank: int,
        num_workers: int,
        comm_id: int,
        size: int,
        collective: str,
        devices_per_node: int,
        max_devices_per_node: int,
    ) -> None:
        self._rank = rank
        self._num_workers = num_workers
        self._size = size
        self._comm_id = comm_id
        self._collective = collective
        self._devices_per_node = devices_per_node
        self._max_devices_per_node = max_devices_per_node

        self._graphed_collective = GraphedCollective(
            num_workers, size, collective=collective, disable_graph=DISABLE_GRAPH
        )

        # TimerStatsStore是一个单例
        # TimerStatsStore is a singleton
        self.timer_stats_store = TimerStatsStore(profile_method="kineto")
        self._cuda_timer = CudaTimer(
            collective, aggregation_fn=np.median, filter_str="nccl"
        )

    def _run_collective(self):
        torch.cuda.synchronize()
        torch.distributed.barrier()

        with self._cuda_timer:
            if DISABLE_GRAPH:
                for _ in range(GRAPH_DISABLED_STEPS):
                    self._graphed_collective.launch()

            self._graphed_collective.launch()

        torch.cuda.synchronize()

    def profile(self):
        self.timer_stats_store.clear_stats()
        for _ in range(ACTIVE_STEPS):
            self._run_collective()

        return {
            "time_stats": self.timer_stats_store.get_stats(),
            "rank": self._rank,
            "num_workers": self._num_workers,
            "size": self._size * 2,  # bytes
            "collective": self._collective,
            "devices_per_node": self._devices_per_node,
            "max_devices_per_node": self._max_devices_per_node,
        }
