from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas

        self._num_replicas = len(self._replicas)

        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
            simulation_config=self._config
        )
        # 每个replica有自己的sheduler
        # Each replica has its own scheduler
        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get(
                config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=execution_time_predictor,
            )
            for replica_id, replica in replicas.items()
        }
        self._request_queue = []

    def sort_requests(self) -> None:
        # print(f"> {self._request_queue}")
        self._request_queue.sort(key=lambda request: request._arrived_at)
        
        # > Print all request IDs (in one line)
        request_ids = [request.id for request in self._request_queue]
        # print(f"> Debug: Request IDs in queue: {request_ids}, len(self._request_queue)={len(self._request_queue)}")

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        # print(f"> Debug: len(self._request_queue)={len(self._request_queue)} len(self._request_queue) == 0 :{len(self._request_queue) == 0}")
        # for replica_scheduler in self._replica_schedulers.values():
        #     print(f"> Debug: Replica {replica_scheduler._replica_id} replica_scheduler.is_empty()={replica_scheduler.is_empty()}")
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
