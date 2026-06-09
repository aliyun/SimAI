from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class VLLMReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        
        # > num_tokens_reserved应该大于等于 request.num_processed_tokens， 顶多小1个token； 一次iteration不能处理两个token； 等于2肯定是有token没处理完
        # num_tokens_reserved should be greater than or equal to request.num_processed_tokens, at most 1 token less;
        # one iteration cannot process two tokens; equals to 2 definitely means there are tokens not finished processing
        
        
        # print(f"> Debug: in _allocate_request: req id={request._id} request._num_processed_tokens={request._num_processed_tokens} num_tokens_required={num_tokens_required} num_tokens_reserved={num_tokens_reserved} num_tokens_required={num_tokens_required}")
        # print(f"> Debug: self._allocation_map[request.id]={self._allocation_map[request.id]} self._config.block_size={self._config.block_size}")
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        # print(f"debug: entering _get_next_batch len(self._request_queue)={len(self._request_queue)}")
        
        
        while self._request_queue:
            request = self._request_queue[0]

            next_num_tokens = self._get_request_next_num_tokens(request)

            if not self._can_allocate_request(request):
                break

            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                break

            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens

        if requests:
            # print(f"> Debug: first return vllm replica scheduler formed batch: replica_id={self._replica_id}, req count={len(requests)}, num_tokens={num_tokens}")
            return Batch(self._replica_id, requests, num_tokens)

        # Safer to sort preempted_requests to maintain FIFO order
        self._preempted_requests.sort(key=lambda r: r.arrived_at)
        # all preempted_requests will have prefill completed
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            # print(f"> Debug: vllm replica scheduler failed to form batch: replica_id={self._replica_id}, req count={len(requests)}, num_tokens={num_tokens}")
            
            return
        # print(f"> Debug: second return vllm replica scheduler formed batch: replica_id={self._replica_id}, req count={len(requests)}, num_tokens={num_tokens}")
        
        return Batch(self._replica_id, requests, num_tokens)
