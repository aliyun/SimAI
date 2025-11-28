from vidur.config import ReplicaConfig
from vidur.entities import BatchStage
from vidur.utils.param_counter import ParamCounter


class MFUCalculator:

    def __init__(self, replica_config: ReplicaConfig):
        param_counter = ParamCounter(replica_config)
        self._num_params_per_device = param_counter.get_num_parameters_per_device()

        model_config = replica_config.model_config

        self._num_layers_per_device = (
            model_config.num_layers // replica_config.num_pipeline_stages
        )
        self._num_heads_per_device = (
            model_config.num_q_heads // replica_config.tensor_parallel_size
        )
        self._head_dimension = model_config.embedding_dim // model_config.num_q_heads
        self._device_flops = replica_config.device_config.fp16_tflops * 2**40

    def _get_mlp_flops(self, batch_stage: BatchStage) -> float:
        num_tokens = sum(batch_stage.num_tokens)
        return 2 * num_tokens * self._num_params_per_device

    def _get_attention_flops(self, batch_stage: BatchStage) -> float:
        total_flops = 0
        for request, num_tokens in zip(batch_stage.requests, batch_stage.num_tokens):
            total_flops += (
                4  # for number of ops in attention
                * self._num_layers_per_device
                * self._num_heads_per_device
                * self._head_dimension
                * num_tokens  # q length
                * (num_tokens + request.num_processed_tokens)  # kv length
            )

        return total_flops

    def get_mfu(self, batch_stage: BatchStage) -> float:
        mlp_flops = self._get_mlp_flops(batch_stage)
        attention_flops = self._get_attention_flops(batch_stage)
        total_flops = mlp_flops + attention_flops
        total_flops_per_second = total_flops / batch_stage.execution_time
        return total_flops_per_second * 100 / self._device_flops
