from typing import Tuple

from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)


class FixedRequestLengthGenerator(BaseRequestLengthGenerator):

    def get_next_num_tokens(self) -> Tuple[float, float]:
        return (
            self.config.prefill_tokens,
            self.config.decode_tokens,
        )
