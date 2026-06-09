from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


# 新的请求到达
# New request arrives
class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time, EventType.REQUEST_ARRIVAL)

        self._request = request

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        logger.debug(f"Request: {self._request.id} arrived at {self.time}")
        # 该请求加入调度
        # Add request to scheduler
        scheduler.add_request(self._request)
        metrics_store.on_request_arrival(self.time, self._request)
        # 触发一个全局调度事件
        # Trigger a global scheduling event
        print(f"> Debug: time={self.time} Event {self._id} of type {self._event_type} Generates 1 GlobalScheduleEvent")
        
        return [GlobalScheduleEvent(self.time)]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request": self._request.id,
        }
