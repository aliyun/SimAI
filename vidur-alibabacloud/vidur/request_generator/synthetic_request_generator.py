from typing import List

from vidur.config import SyntheticRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from vidur.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from vidur.types import RequestIntervalGeneratorType
from vidur.utils.random import set_seeds


class SyntheticRequestGenerator(BaseRequestGenerator):

    def __init__(self, config: SyntheticRequestGeneratorConfig):
        super().__init__(config)

        # 请求长度分布
        # Request length distribution
        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.length_generator_config.get_type(),
            self.config.length_generator_config,
        )
        # 请求间隔分布
        # Request interval distribution
        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
        )

    def _generate_next_request(self, last_arrived_at: float) -> Request:
        inter_request_time = (
            self.request_interval_generator.get_next_inter_request_time()
        )
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time

        (
            prefill_tokens,
            decode_tokens,
        ) = self.request_length_generator.get_next_num_tokens()

        if prefill_tokens is None or decode_tokens is None:
            return None

        return Request(
            arrived_at=arrived_at,
            num_prefill_tokens=int(prefill_tokens),
            num_decode_tokens=int(decode_tokens),
        )

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        # first priority is duration
        if self.config.duration is not None:
            while current_time < self.config.duration:
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        elif self.config.num_requests is not None:
            for _ in range(self.config.num_requests):
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        else:
            assert (
                self.config.interval_generator_config.get_type()
                == RequestIntervalGeneratorType.TRACE
            )

            while True:
                request = self._generate_next_request(current_time)
                if request is None:
                    break
                current_time = request.arrived_at
                requests.append(request)

        return requests

    def generate_requests(self) -> List[Request]:
        assert (
            self.config.duration
            or self.config.num_requests
            or self.config.interval_generator_config.get_type()
            == RequestIntervalGeneratorType.TRACE
        )

        set_seeds(self.config.seed)

        requests = self._generate_requests()

        # sort requests by arrival time
        requests.sort(key=lambda x: x.arrived_at)
        # remove any requests that arrived after the time limit
        if self.config.duration is not None:
            requests = [
                request
                for request in requests
                if request.arrived_at < self.config.duration
            ]

        return requests
