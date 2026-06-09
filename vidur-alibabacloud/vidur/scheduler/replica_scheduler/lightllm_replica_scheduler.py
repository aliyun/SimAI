from typing import List, Tuple

import numpy as np

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class LightLLMReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        assert (
            self._config.block_size == 1
        ), "LightLLM scheduler only supports block size of 1."
        assert (
            self._num_stages == 1
        ), "LightLLM scheduler does not support pipeline parallel."

        self._cache_len_list = []
        self._num_waiting_iters = 0

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_tuple_tokens(self, request: Request) -> Tuple[int, int]:
        if request.scheduled:
            num_processed_tokens = request.num_processed_tokens
            remaining_tokens = (
                request.num_decode_tokens - request.num_processed_decode_tokens - 1
            )
        else:
            num_processed_tokens = request.num_prefill_tokens + 1
            remaining_tokens = request.num_decode_tokens - 1 - 1

        remaining_tokens = max(0, remaining_tokens)

        return (num_processed_tokens, remaining_tokens)

    def _can_allocate_request(self, request: Request) -> bool:
        # I have no idea what this does
        self.cache_len_list.append(self._get_tuple_tokens(request))
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()

        return need_max_token_num < self._config.num_blocks

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            self.allocate(request.id, request.num_prefill_tokens)
            return

        self.allocate(request.id, 1)

    def _get_prefill_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        self.cache_len_list = [
            self._get_tuple_tokens(request) for request in self._preempted_requests
        ]

        while self._request_queue:
            request = self._request_queue[0]

            next_num_tokens = self._get_request_next_num_tokens(request)

            if num_batch_tokens + next_num_tokens > self._config.max_tokens_in_batch:
                break

            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            if not self._can_allocate_request(request):
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens

        if requests:
            return Batch(self._replica_id, requests, num_tokens)

        return

    def _get_decode_batch(self) -> Batch:
        requests = []
        num_tokens = []

        # all preempted_requests will have prefill completed
        while self._preempted_requests:
            assert len(requests) < self._max_micro_batch_size

            request = self._preempted_requests.pop(0)

            assert self.can_allocate(1)
            self._allocate_request(request)

            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)

    def _can_decode(self):
        return self.can_allocate(len(self._preempted_requests))

    def _get_next_batch(self) -> Batch:
        if not self._preempted_requests:
            batch = self._get_prefill_batch()
            if batch:
                self._num_waiting_iters = 0
            return batch

        if self._num_waiting_iters >= self._config.max_waiting_iters:
            self._num_waiting_iters = 0
            batch = self._get_prefill_batch()
            if batch:
                return batch

        if self._can_decode():
            self._num_waiting_iters += 1
            return self._get_decode_batch()
        else:
            raise RuntimeError("OOM handling not implemented yet")
