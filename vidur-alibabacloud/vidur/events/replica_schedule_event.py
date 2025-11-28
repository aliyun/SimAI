from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


# Replica之间的执行是独立的

# Execution between replicas is independent
class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id

        self._batches = []

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        # _batches中至多有PP-stages个batch
        # TODO: 这里有一点奇怪，他这样的话就是每次issue PP-stages个batch
        # _batches contains at most PP-stages batches
        # TODO: This is a bit strange, as it issues PP-stages batches each time
        self._batches = replica_scheduler.on_schedule()

        if not self._batches:
            return []

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent
        )

        for batch in self._batches:
            batch.on_schedule(self.time)

        print(f"> Debug: time={self._time} Generates {len(self._batches)} BatchStageArrivalEvents \
            from event #{self._id} {self._event_type} ")        
        return [
            # 所有PP-stages个batch到达stage 0
            # All PP-stages batches arrive at stage 0
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "batch_ids": [batch.id for batch in self._batches],
        }
