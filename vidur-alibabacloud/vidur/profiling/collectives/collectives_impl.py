from typing import Callable

import torch

WARMUP_STEPS = 5
GRAPH_STEPS = 3


class GraphedCollective:
    def __init__(
        self,
        num_workers: int,
        size: int,
        collective: str = "all_reduce",
        disable_graph: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._size = size
        self._disable_graph = disable_graph
        self._collective_fn = self._get_collective_fn(collective)

        self._buffer = torch.empty(
            size=(size,),
            dtype=dtype,
            device="cuda",
        )
        self._gather_buffer = None
        if collective == "all_gather":
            self._gather_tensor = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )
        elif collective == "reduce_scatter":
            self._reduce_buffer = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )
        # TODO >  elif collective == "all_to_all":
        # elif collective == "all_to_all":
        #     # TODO > change _reduce_buffer to what? all to all buffer?
        #     self._reduce_buffer = torch.empty(
        #         size=(size * num_workers,),
        #         dtype=dtype,
        #         device="cuda",
        #     )
        elif collective == "all_to_all":
            # For all_to_all_single operation, we need both input and output buffers
            # with the same size. Each process sends and receives chunks of equal size.
            self._alltoall_input_buffer = torch.empty(
                size=(size,),
                dtype=dtype,
                device="cuda",
            )
            self._alltoall_output_buffer = torch.empty(
                size=(size,),
                dtype=dtype,
                device="cuda",
            )
        if not self._disable_graph:
            self._graph = self._build_graph()
            
        # >
        self._num_workers = num_workers

    def _run_all_reduce(self):
        torch.distributed.all_reduce(self._buffer)

    def _run_all_gather(self):
        torch.distributed.all_gather_into_tensor(self._gather_tensor, self._buffer)

    def _run_broadcast(self):
        torch.distributed.broadcast(self._buffer, 0)

    def _run_send_recv(self):
        if torch.distributed.get_rank() == 0:
            torch.distributed.send(self._buffer, 1)
        else:
            torch.distributed.recv(self._buffer, 0)

    def _run_reduce_scatter(self):
        # > torch.distributed function: def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
        torch.distributed.reduce_scatter_tensor(self._buffer, self._reduce_buffer)
        
    # TODO > modify according to def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        # Or use all_to_all_single first?
        # def all_to_all_single(
        #     output,
        #     input,
        #     output_split_sizes=None,
        #     input_split_sizes=None,
        #     group=None,
        #     async_op=False,
        # ):
    def _run_all_to_all(self):
        # torch.distributed.all_to_all(self._buffer, self._reduce_buffer)
        # torch.distributed.all_to_all_single(self._buffer, self._reduce_buffer)
        # torch.distributed.all_to_all_single(self._buffer, self._buffer)
        # torch.distributed.all_to_all_single(self._reduce_buffer, self._reduce_buffer)
        # torch.distributed.all_to_all_single(
        #     self._buffer, 
        #     self._reduce_buffer,
        #     output_split_sizes=[self._size]* self._num_workers,
        #     input_split_sizes=[self._size] * self._num_workers
        # )
        
        # all_to_all_single requires input and output tensors of the same size
        # Each process contributes an equal share of data
        torch.distributed.all_to_all_single(
            self._alltoall_output_buffer, 
            self._alltoall_input_buffer
        )


    def _get_collective_fn(self, collective: str) -> Callable:
        if collective == "all_reduce":
            return self._run_all_reduce
        elif collective == "all_gather":
            return self._run_all_gather
        elif collective == "broadcast":
            return self._run_broadcast
        elif collective == "send_recv":
            return self._run_send_recv
        elif collective == "reduce_scatter":
            return self._run_reduce_scatter
        elif collective == "all_to_all": # > add
            return self._run_all_to_all
        else:
            raise ValueError(f"Unknown collective: {collective}")

    def _build_graph(self) -> torch.cuda.CUDAGraph:
        # Warm up.
        for _ in range(WARMUP_STEPS):
            self._collective_fn()

        torch.cuda.synchronize()

        # Build graph.
        graph = torch.cuda.CUDAGraph()

        mempool = torch.cuda.graph_pool_handle()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
        ):
            with torch.cuda.graph(graph, mempool):
                for _ in range(GRAPH_STEPS):
                    self._collective_fn()

        torch.cuda.synchronize()
        return graph

    def launch(self) -> torch.Tensor:
        # NOTE: x must be a slice of self._buffer.
        if self._disable_graph:
            self._collective_fn()
        else:
            self._graph.replay()
