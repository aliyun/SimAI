import logging
from typing import List

import pandas as pd

from vidur.config import TraceRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


class TraceReplayRequestGenerator(BaseRequestGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] - diff_tokens
        )

        assert all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= config.max_tokens
        )

        # rescale the time to change QPS
        self.trace_df["arrived_at"] = (
            self.trace_df["arrived_at"] * config.time_scale_factor
        )

        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.debug(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

    def generate_requests(self) -> List[Request]:
        requests = []

        for _, row in self.trace_df.iterrows():
            request = Request(
                arrived_at=row["arrived_at"],
                num_prefill_tokens=row["num_prefill_tokens"],
                num_decode_tokens=row["num_decode_tokens"],
            )

            requests.append(request)

        return requests
