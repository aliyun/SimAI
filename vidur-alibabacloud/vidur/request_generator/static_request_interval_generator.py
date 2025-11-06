from vidur.request_generator.base_request_interval_generator import (
    BaseRequestIntervalGenerator,
)


class StaticRequestIntervalGenerator(BaseRequestIntervalGenerator):

    def get_next_inter_request_time(self) -> float:
        return 0
