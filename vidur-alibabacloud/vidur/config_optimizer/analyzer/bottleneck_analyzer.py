from vidur.config_optimizer.analyzer.bottleneck_cases import *


class BottleneckAnalyzer:
    """
    Why I can't get more capacity out of my replica?
    C1. Violates TFFT SLO
        C1.1 Limited by  max batch size - Batch size saturated - mean memory usage is low.
        C1.2 Limited by memory availability - Memory usage is high, batch size is lower than limit.
        C1.3 Limited by prefill throughput - None of the above applies (NEED BETTER STRAT).
            C1.3.1 Low efficiency due to pipeline bubbles - high idle time and pipeline parallelism.
            X C1.3.2 Low efficiency due to network latency - high TP comm time.
            X C1.3.3 Low efficiency due to chunking overhead - lower prefill MFU and small chunk size.
            X C1.3.4 Slow prefill due to low parallelism - high prefill MFU and low TP degree.
    C2. Violates TBT SLO
        C2.1 High latency model execution - P25/P50 TBT is high.
            C2.1.1 Stalls due to pipeline bubbles - high idle time and pipeline parallelism.
            X C2.1.2 High latency due to network latency - lower MFU and high TP degree.
            X C2.1.4 Slow prefill due to low parallelism - high MFU and low TP degree.
            X C2.1.3 High latency due to chunking overhead - high chunk size.
        C2.2 High tail latency - lower percentiles are okay but tail latency is violated.
            C2.2.1 High tail latency due to scheduling problems - non-sarathi scheduler.
            C2.2.2 High tail latency due to large chunk size - sarathi scheduler and high chunk size.

    Classes marked with X are not implemented yet, and require more metrics to be implemented.
    """

    def __init__(
        self, ttft_slo_percentile, ttft_slo_value, tbt_slo_percentile, tbt_slo_value
    ):
        self.ttft_slo_percentile = ttft_slo_percentile
        self.ttft_slo_value = ttft_slo_value
        self.tbt_slo_percentile = tbt_slo_percentile
        self.tbt_slo_value = tbt_slo_value

    def analyze(self, config_row):
        """
        Analyze the config row and return the bottleneck.
        """
        params = (
            config_row,
            self.ttft_slo_percentile,
            self.ttft_slo_value,
            self.tbt_slo_percentile,
            self.tbt_slo_value,
        )

        # put child classes before parent classes
        match_seq = [
            TFFTViolationLowMaxBatchSizeCase,
            TFFTViolationLowMemoryCase,
            TFFTViolationLowPrefillThroughputPipelineBubblesCase,
            TFFTViolationLowPrefillThroughputCase,
            TBTViolationHighModelExecutionLatencyPipelineBubblesCase,
            TBTViolationHighModelExecutionLatencyCase,
            TBTViolationHighTailLatencySchedulerCase,
            TBTViolationHighTailLatencySarathiChunkSizeCase,
        ]

        # find and return the first match
        for case in match_seq:
            case_inst = case(*params)
            if case_inst.is_match():
                return case_inst.get_message()

        return "Could not interpret the bottleneck."
