from abc import ABC, abstractmethod

import numpy as np

TTFT_THRESHOLD = 0.8
TBT_THRESHOLD = 0.8
LOW_BATCH_SIZE_PERCENTILE = 75
LOW_BATCH_SIZE_THRESHOLD = 0.9
LOW_MEMORY_MEMORY_THRESHOLD = 0.7
LOW_MEMORY_BATCH_SIZE_PERCENTILE = 75
LOW_MEMORY_BATCH_SIZE_THRESHOLD = 0.7
LOW_PREFILL_THROUGHPUT_PIPELINE_BUBBLES_IDLE_THRESHOLD = 0.4
HIGH_MODEL_EXECUTION_LATENCY_PERCENTILE = 25
HIGH_MODEL_EXECUTION_LATENCY_PERCENTILE_THRESHOLD = 0.75


str_to_list = lambda x: list(map(float, x[1:-1].split(",")))


class BaseBottleneckCase(ABC):
    def __init__(
        self,
        config_row,
        ttft_slo_percentile: float,
        ttft_slo_value: float,
        tbt_slo_percentile: float,
        tbt_slo_value: float,
    ) -> None:
        self.config_row = config_row
        self.ttft_slo_percentile = ttft_slo_percentile
        self.ttft_slo_value = ttft_slo_value
        self.tbt_slo_percentile = tbt_slo_percentile
        self.tbt_slo_value = tbt_slo_value

    @abstractmethod
    def is_match(self) -> bool:
        return True

    @abstractmethod
    def get_message(self) -> str:
        return ""


class TTFTViolationCase(BaseBottleneckCase):
    def is_match(self):
        super_is_match = super().is_match()
        return (
            super_is_match
            and self.config_row[f"TTFT_{self.ttft_slo_percentile}%"]
            > self.ttft_slo_value * TTFT_THRESHOLD
        )

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} TTFT (P{self.ttft_slo_percentile}) SLO violation risk."


class TBTViolationCase(BaseBottleneckCase):
    def is_match(self):
        super_is_match = super().is_match()
        return (
            super_is_match
            and self.config_row[f"TBT_{self.tbt_slo_percentile}%"]
            > self.tbt_slo_value * TBT_THRESHOLD
        )

    def get_message(self):
        super_get_message = super().get_message()
        return (
            f"{super_get_message} TBT (P{self.tbt_slo_percentile}) SLO violation risk."
        )


class TFFTViolationLowMaxBatchSizeCase(TTFTViolationCase):
    def is_match(self):
        super_is_match = super().is_match()
        # check if P{X} batch size greater than {Y} * max batch size
        batch_size_cdf = str_to_list(self.config_row["batch_size_cdf"])

        batch_size_obs = np.percentile(batch_size_cdf, LOW_BATCH_SIZE_PERCENTILE)

        max_batch_size = self.config_row["Batch Size"]

        return (
            super_is_match
            and batch_size_obs > LOW_BATCH_SIZE_THRESHOLD * max_batch_size
        )

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} Increase max batch size."


class TFFTViolationLowMemoryCase(TTFTViolationCase):
    def is_match(self):
        super_is_match = super().is_match()
        # check if memory usage is high
        memory_usage = self.config_row["memory_usage_mean"]
        mean_memory_usage = np.mean(memory_usage)
        # and batch size is lower than limit
        batch_size_cdf = str_to_list(self.config_row["batch_size_cdf"])

        batch_size_obs = np.percentile(batch_size_cdf, LOW_MEMORY_BATCH_SIZE_PERCENTILE)

        return (
            super_is_match
            and mean_memory_usage > LOW_MEMORY_MEMORY_THRESHOLD
            and batch_size_obs < LOW_MEMORY_BATCH_SIZE_THRESHOLD
        )

    def get_message(self):
        super_get_message = super().get_message()
        return (
            f"{super_get_message} Low memory availability, increase parallelism degree."
        )


class TFFTViolationLowPrefillThroughputCase(TTFTViolationCase):
    def is_match(self):
        super_is_match = super().is_match()
        # just check if none of the other cases apply
        is_low_max_batch_size = TFFTViolationLowMaxBatchSizeCase(
            self.config_row,
            self.ttft_slo_percentile,
            self.ttft_slo_value,
            self.tbt_slo_percentile,
            self.tbt_slo_value,
        ).is_match()
        is_low_memory = TFFTViolationLowMemoryCase(
            self.config_row,
            self.ttft_slo_percentile,
            self.ttft_slo_value,
            self.tbt_slo_percentile,
            self.tbt_slo_value,
        ).is_match()

        return super_is_match and not is_low_max_batch_size and not is_low_memory

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} Low prefill throughput."


class PipelineBubblesCase(BaseBottleneckCase):
    def is_match(self):
        super_is_match = super().is_match()
        # check if low efficiency due to pipeline bubbles
        idle_time = 100 - self.config_row["busy_time_percent_mean"]
        return (
            super_is_match
            and idle_time > LOW_PREFILL_THROUGHPUT_PIPELINE_BUBBLES_IDLE_THRESHOLD * 100
        )

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} High pipeline bubbles causing low performance."


class TFFTViolationLowPrefillThroughputPipelineBubblesCase(
    PipelineBubblesCase, TFFTViolationLowPrefillThroughputCase
):
    pass


class TBTViolationHighModelExecutionLatencyCase(TBTViolationCase):
    def is_match(self):
        super_is_match = super().is_match()
        low_percentile_tbt = self.config_row[
            f"TBT_{HIGH_MODEL_EXECUTION_LATENCY_PERCENTILE}%"
        ]
        return (
            super_is_match
            and low_percentile_tbt
            > self.tbt_slo_value * HIGH_MODEL_EXECUTION_LATENCY_PERCENTILE_THRESHOLD
        )

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} High model execution latency, increase parallelism degree."


class TBTViolationHighModelExecutionLatencyPipelineBubblesCase(
    PipelineBubblesCase, TBTViolationHighModelExecutionLatencyCase
):
    pass


class TBTViolationHighTailLatencyCase(TBTViolationCase):
    def is_match(self):
        super_is_match = super().is_match()
        low_percentile_tbt = self.config_row[
            f"TBT_{HIGH_MODEL_EXECUTION_LATENCY_PERCENTILE}%"
        ]
        tail_tbt = self.config_row[f"TBT_{self.tbt_slo_percentile}%"]
        return (
            super_is_match
            and low_percentile_tbt
            < self.tbt_slo_value * HIGH_MODEL_EXECUTION_LATENCY_PERCENTILE_THRESHOLD
            and tail_tbt > self.tbt_slo_value * TBT_THRESHOLD
        )

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} High tail latency."


class TBTViolationHighTailLatencySchedulerCase(TBTViolationHighTailLatencyCase):
    def is_match(self):
        super_is_match = super().is_match()
        return super_is_match and self.config_row["Scheduler"] != "Sarathi-Serve"

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} Switch to Sarathi-Serve scheduler to avoid prefill-decode interference."


class TBTViolationHighTailLatencySarathiChunkSizeCase(TBTViolationHighTailLatencyCase):
    def is_match(self):
        super_is_match = super().is_match()
        return (
            super_is_match
            and self.config_row["Scheduler"] == "Sarathi-Serve"
            and self.config_row["Sarathi Chunk Size"] > 1024
        )

    def get_message(self):
        super_get_message = super().get_message()
        return f"{super_get_message} Reduce chunk size."
