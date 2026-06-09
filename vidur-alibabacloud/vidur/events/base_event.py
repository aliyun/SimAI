from abc import ABC, abstractmethod
from typing import List

from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType


class BaseEvent(ABC):
    _id = 0

    def __init__(self, time: float, event_type: EventType):
        self._time = time
        self._id = BaseEvent.generate_id()
        self._event_type = event_type
        self._priority_number = self._get_priority_number()

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id

    @property
    def id(self) -> int:
        return self._id

    @property
    def time(self):
        return self._time

    @property
    def event_type(self):
        pass

    @abstractmethod
    def handle_event(
        self,
        current_time: float,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
    ) -> List["BaseEvent"]:
        pass

    def _get_priority_number(self):
        return (self._time, self._id, self.event_type)

    def __lt__(self, other):
        if self._time == other._time:
            if self._event_type == other._event_type:
                return self._id < other._id
            return self._event_type < other._event_type
        else:
            return self._time < other._time

    def __eq__(self, other):
        return (
            self._time == other._time
            and self._event_type == other._event_type
            and self._id == other._id
        )

    def __str__(self) -> str:
        # use to_dict to get a dict representation of the object
        # and convert it to a string
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.to_dict())})"

    def to_dict(self):
        return {"time": self.time, "event_type": self.event_type}

    def to_chrome_trace(self) -> dict:
        return None
