import time

import torch
from torch.profiler import record_function

from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.utils import ProfileMethod


class CudaTimer:
    def __init__(
        self,
        name,
        layer_id: int = 0,  # we don't care about layer id, it is just for compatibility with sarathi cudatimer
        aggregation_fn=sum,
        filter_str=None,
    ):
        if name:
            # beautify the names we get from vllm
            name = str(name).replace("OperationMetrics.", "")
            name = name.lower()
            self.name = f"vidur_{name}"
        else:
            self.name = None

        self.timer_stats_store = TimerStatsStore()
        self.disabled = (name is None) or self.timer_stats_store.disabled

        if self.disabled:
            return

        self.aggregation_fn = aggregation_fn
        self.filter_str = filter_str

        # KINETO是torch.profier背后的性能分析工具，会生成trace
        # KINETO is the performance profiling tool behind torch.profiler, which generates traces
        if self.timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=self.handle_trace,
            )
        else:
            self.profiler = None
        self.start_event = None
        self.end_event = None
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if self.disabled:
            return

        if self.timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:
            self.profiler_function_context = record_function(self.name)
            self.profiler_function_context.__enter__()
        elif self.timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        elif self.timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler.__enter__()
        elif self.timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        else:
            raise ValueError(
                f"Unknown profile method {self.timer_stats_store.profile_method}"
            )
        return self

    def handle_trace(self, trace):
        events = trace.events()

        if self.filter_str:
            events = [e for e in events if e.name.startswith(self.filter_str)]

        total_cuda_time = self.aggregation_fn([e.cuda_time_total for e in events])
        # kineto trace的时间单位是微秒
        # The time unit of kineto trace is microseconds
        self.timer_stats_store.record_time(
            self.name, total_cuda_time * 1e-3
        )  # convert to ms

    def __exit__(self, *args):
        if self.disabled:
            return

        if self.timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:
            self.profiler_function_context.__exit__(*args)
        elif self.timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.end_event.record()
            self.timer_stats_store.record_time(
                self.name, [self.start_event, self.end_event]
            )
        elif self.timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler.__exit__(*args)
        elif self.timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:
            torch.cuda.synchronize()
            self.end_time = time.perf_counter()
            self.timer_stats_store.record_time(
                self.name, (self.end_time - self.start_time) * 1e3
            )  # convert to ms
        else:
            raise ValueError(
                f"Unknown profile method {self.timer_stats_store.profile_method}"
            )
