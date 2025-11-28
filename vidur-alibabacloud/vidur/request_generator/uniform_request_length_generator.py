import math
import random
from typing import Tuple

from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)


class UniformRequestLengthGenerator(BaseRequestLengthGenerator):

    def get_next_num_tokens(self) -> Tuple[float, float]:
        total_tokens = random.uniform(
            self.config.min_tokens,
            self.config.max_tokens,
        )

        decode_tokens = math.ceil(
            total_tokens / (1 + self.config.prefill_to_decode_ratio)
        )
        prefill_tokens = total_tokens - decode_tokens
        assert prefill_tokens > 0 and decode_tokens > 0

        return prefill_tokens, decode_tokens
