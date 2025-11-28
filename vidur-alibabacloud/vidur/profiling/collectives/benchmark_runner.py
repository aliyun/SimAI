import gc
import os
from typing import Optional

import ray
import torch

from vidur.logger import init_logger
from vidur.profiling.collectives.collectives_input import CollectivesInput
from vidur.profiling.collectives.collectives_wrapper import CollectiveWrapper

logger = init_logger(__name__)


@ray.remote(num_gpus=1)
class BenchmarkRunner:
    def __init__(self, gpu_id: int, max_gpus_per_node: int, head_ip: str) -> None:
        self._gpu_id = gpu_id
        self._max_devices_per_node = max_gpus_per_node
        self._set_cuda_visible_devices()
        self._last_num_workers_per_node = None
        self._last_num_workers = None
        self._head_ip = head_ip

    def _set_cuda_visible_devices(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            self._gpu_id % self._max_devices_per_node
        )
        # set additional nccl env vars
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Required for properly capturing nccl ops
        os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
        os.environ["KINETO_LOG_LEVEL"] = "5"
        os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

    def run_collective(
        self,
        collectives_input: CollectivesInput,
    ) -> Optional[dict]:
        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ) and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        rank = self._get_rank(
            collectives_input.num_workers, collectives_input.num_workers_per_node
        )
        if rank is None:
            return None

        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ):
            self._init_communication(
                collectives_input.comm_id,
                rank,
                collectives_input.num_workers,
                collectives_input.num_workers_per_node,
            )
            self._last_num_workers = collectives_input.num_workers
            self._last_num_workers_per_node = collectives_input.num_workers_per_node

        wrapper = CollectiveWrapper(
            rank,
            collectives_input.num_workers,
            collectives_input.comm_id,
            collectives_input.collective_size,
            collectives_input.collective,
            collectives_input.num_workers_per_node,
            self._max_devices_per_node,
        )
        stats = wrapper.profile()
        del wrapper
        gc.collect()
        return stats

    def _init_communication(
        self, comm_id: int, rank: int, num_workers: int, devices_per_node: int
    ):
        logger.info(
            f"Initializing gpu id: {self._gpu_id}, Rank: {rank}, num_workers: {num_workers}, comm_id: {comm_id}, "
            f"devices_per_node: {devices_per_node}, max_devices_per_node: {self._max_devices_per_node}, "
            f"ip_addr: {ray.util.get_node_ip_address()}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        # TODO > 可以改成deepep后端
        # TODO > can be changed to deepep backend
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=num_workers,
            init_method=f"tcp://{self._head_ip}:{comm_id}",
        )

    def _get_rank(self, num_workers: int, devices_per_node: int):
        assert self._max_devices_per_node >= devices_per_node
        assert self._max_devices_per_node % devices_per_node == 0
        assert num_workers % devices_per_node == 0 or num_workers < devices_per_node

        num_nodes = num_workers // devices_per_node
        current_node = self._gpu_id // self._max_devices_per_node

        if current_node >= num_nodes:
            return None

        local_gpu_id = self._gpu_id % self._max_devices_per_node

        # # scatter devices uniformly across the node
        # node_devices = list(range(self._max_devices_per_node))
        # device_offset = self._max_devices_per_node // devices_per_node

        # # selected devices for this worker
        # selected_devices = node_devices[::device_offset]

        # pack devices in order
        selected_devices = list(range(devices_per_node))

        if local_gpu_id not in selected_devices:
            return None

        # rank of this worker
        rank = current_node * devices_per_node + selected_devices.index(local_gpu_id)

        return rank
