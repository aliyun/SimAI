import math
import random

from vidur.config import PoissonRequestIntervalGeneratorConfig
from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class PoissonRequestIntervalGenerator(BaseRequestIntervalGenerator):

    def __init__(self, config: PoissonRequestIntervalGeneratorConfig):
        super().__init__(config)

        self.qps = self.config.qps
        self.std = 1.0 / self.qps
        self.max_interval = self.std * 3.0

    def get_next_inter_request_time(self) -> float:
        next_interval = -math.log(1.0 - random.random()) / self.qps
        next_interval = min(next_interval, self.max_interval)

        return next_interval
