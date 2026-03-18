from typing import List

from vidur.entities.batch import Batch
from vidur.entities.batch_stage import BatchStage
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


# 单个micro-batch在单个PP stage上执行结束，进入下一个stage
# A single micro-batch completes execution on a single PP stage and moves to the next stage
class BatchStageEndEvent(BaseEvent):
    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        batch: Batch,
        batch_stage: BatchStage,
    ):
        super().__init__(time, EventType.BATCH_STAGE_END)

        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage

        self._batch = batch
        self._batch_stage = batch_stage

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_end_event import BatchEndEvent
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent
        from vidur.events.replica_stage_schedule_event import ReplicaStageScheduleEvent

        scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        ).on_stage_end()

        self._batch_stage.on_stage_end(self.time)
        metrics_store.on_batch_stage_end(
            self._batch_stage,
            self.time,
            self._replica_id,
            self._stage_id,
        )

        next_events = [
            # 当前stage调度下一个micro-batch
            # TODO(tianhao909): odd behavior - BatchStageEndEvent triggers current stage scheduling
            # BatchStageArrivalEvent also triggers current stage scheduling
            # Although multiple scheduling doesn't cause issues, many schedules are redundant
            # (because stage_scheduler.is_busy = True or stage_scheduler.queue is empty)
            # TODO(tianhao909): 这里有点怪，BatchStageEndEvent 会触发当前 stage 的调度
            # BatchStageArrivalEvent 也会触发当前 stage 的调度
            # 虽然多次调度不会引发问题，但很多调度是冗余的
            ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
            ),
        ]
        
        # print(f"> Debug: time={self._time} from event #{self._id} {self._event_type} \
            # generating 1 ReplicaStageScheduleEvent replica_id={self._replica_id} \
            # stage_id={self._stage_id}")

        if self._is_last_stage:
            # print(f"> Debug: time={self._time} from event #{self._id} {self._event_type} \
                # generating 1 BatchEndEvent replica_id={self._replica_id}")

            # 一个micro-batch执行结束
            # A micro-batch execution completes
            return next_events + [
                BatchEndEvent(self.time, self._replica_id, self._batch)
            ]
            
        # print(f"> Debug: time={self._time} from event #{self._id} {self._event_type} \
            # generating 1 BatchStageArrivalEvent replica_id={self._replica_id} \
            # stage_id={self._stage_id +1 } batch id = {self._batch._id}" )

        return next_events + [
            # 当前micro-batch进入下一个stage
            # Current micro-batch enters the next stage 
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                self._stage_id + 1,
                self._batch,
            )
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
            "batch_stage_id": self._batch_stage.id,
            "is_last_stage": self._is_last_stage,
        }

    def to_chrome_trace(self) -> dict:
        return self._batch_stage.to_chrome_trace(self.time)
