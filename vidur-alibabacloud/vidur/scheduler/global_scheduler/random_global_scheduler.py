from random import randint
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class RandomGlobalScheduler(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = randint(1, self._num_replicas) - 1
            request_mapping.append((replica_id, request))
         
        return request_mapping
