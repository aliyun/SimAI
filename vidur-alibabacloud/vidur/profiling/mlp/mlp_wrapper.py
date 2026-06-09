import os

import sarathi.metrics.cuda_timer
import torch

from vidur.profiling.common.cuda_timer import CudaTimer

# monkey patching the CudaTimer class to use the sarathi implementation
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.weight_utils import initialize_dummy_weights

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.mlp.mlp_impl import GPTModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2
ACTIVE_STEPS = 20


class MlpWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        profile_method: str,
        rank: int,
        output_dir: str,
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.model_config = model_config
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.profile_method = profile_method
        self.rank = rank
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.model = GPTModel(
            model_config,
            num_tensor_parallel_workers,
            (
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                else 1
            ),
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def profile(self, num_tokens: int):
        vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers
        input_ids = torch.randint(
            low=0,
            high=vocab_range,
            size=(num_tokens,),
            device="cuda",
            dtype=torch.long,
        )
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Run the model once without capturing the graph.
            # This is to make sure that the captured graph does not include the
            # kernel launches for initial benchmarking (e.g., Triton autotune).
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)

            with record_function_tracer:
                self.model(
                    input_ids,
                    positions,
                )

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            for _ in range(WARMUP_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

            time_stats = self.timer_stats_store.get_stats()

        stats = {
            "time_stats": time_stats,
            "n_head": self.model_config.num_q_heads,
            "n_kv_head": self.model_config.num_kv_heads,
            "n_embd": self.model_config.embedding_dim,
            "n_expanded_embd": self.model_config.mlp_hidden_dim,
            "vocab_size": self.model_config.vocab_size,
            "use_gated_mlp": self.model_config.use_gated_mlp,
            "num_tokens": num_tokens,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
        self.timer_stats_store.clear_stats()

        return stats
