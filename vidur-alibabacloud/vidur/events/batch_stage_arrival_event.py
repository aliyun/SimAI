from typing import List

from vidur.entities.batch import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


# 一个micro-batch到达某个PP stage
class BatchStageArrivalEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, stage_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_STAGE_ARRIVAL)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

        scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        ).add_batch(self._batch)
        
        # print(f"> Debug: time={self._time} 从 第{self._id}号evnet{self._event_type}生成 1个 ReplicaStageScheduleEvent replica_id={self._replica_id} stage_id={self._stage_id}")
        print(f"> Debug: time={self._time} Event #{self._id} of type {self._event_type} Generates 1 ReplicaStageScheduleEvent replica_id={self._replica_id} stage_id={self._stage_id}")
        return [
            ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
            )
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
        }