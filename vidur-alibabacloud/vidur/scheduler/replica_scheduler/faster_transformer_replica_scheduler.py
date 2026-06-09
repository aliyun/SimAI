from vidur.entities.batch import Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class FasterTransformerReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_batches = []
        self._num_running_batches = 0
        self._pending_free_map = {}

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        if batch.all_requests_completed:
            # free memory for all requests at once
            self.free_batch(batch)
            self.free(*self._pending_free_map.pop(batch.id, []))
        else:
            self._preempted_batches.append(batch)

    def _generate_next_batch_from_preempted(self, preempted_batch: Batch) -> Batch:
        requests = []
        num_tokens = []

        for request in preempted_batch.requests:
            if request.completed:
                continue
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)

    def _get_next_batch(self) -> Batch:
        if self._preempted_batches:
            preempted_batch = self._preempted_batches.pop(0)
            return self._generate_next_batch_from_preempted(preempted_batch)

        requests = []
        num_tokens = []

        while self._request_queue:
            if len(requests) == self._max_batch_size:
                break

            if not self.can_allocate(self._max_blocks_per_sequence):
                break

            request = self._request_queue.pop(0)
            self.allocate(request.id, self._max_blocks_per_sequence)
            next_num_tokens = self._get_request_next_num_tokens(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)
