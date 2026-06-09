from vidur.config import ReplicaConfig
from vidur.entities import BatchStage
from vidur.entities.request import RequestType
from vidur.logger import init_logger
from vidur.utils.param_counter import ParamCounter

logger = init_logger(__name__)


# MoE模型列表：这些模型需要区分 prefill/decode 的参数量
# MoE model list: These models need separate prefill/decode parameter counts
MOE_MODELS_WITH_PD_SEPARATION = ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']


class MFUCalculator:
    """
    MFU (Model FLOPs Utilization) 计算器
    计算模型计算效率，支持 prefill/decode 分离场景
    
    MFU (Model FLOPs Utilization) Calculator
    Calculates model compute efficiency, supports prefill/decode separation scenarios
    """

    def __init__(self, replica_config: ReplicaConfig):
        self._replica_config = replica_config
        self._model_name = replica_config.model_name
        
        # 判断是否是需要区分 prefill/decode 的 MoE 模型
        # Determine if this is a MoE model requiring prefill/decode separation
        self._is_pd_separated_model = self._model_name in MOE_MODELS_WITH_PD_SEPARATION
        
        param_counter = ParamCounter(replica_config)
        
        # 根据模型类型获取参数量
        # Get parameter counts based on model type
        if self._is_pd_separated_model:
            # MoE模型：返回三元组 (total, prefill, decode)
            # MoE model: returns tuple (total, prefill, decode)
            params_result = param_counter.get_num_parameters_per_device()
            self._num_params_per_device = params_result[0]  # 总参数量 | Total params
            self._prefill_num_params_per_device = params_result[1]  # Prefill参数量 | Prefill params
            self._decode_num_params_per_device = params_result[2]  # Decode参数量 | Decode params
            
            # 打印重要信息便于检查 | Print important info for verification
            # TODO(tianhao909): ParamCounter returns memory bytes (not param count) for new models,
            # causing MFU semantic inconsistency (2*tokens*bytes instead of 2*tokens*params).
            # This is a known limitation; MFU values for MoE models are approximate.
            # TODO(tianhao909): 新模型的 ParamCounter 返回内存字节数而非参数个数，
            # 导致 MFU 语义不一致（2*tokens*bytes 而非 2*tokens*params），MoE 模型的 MFU 为近似值。
            logger.debug(f"[MFUCalculator] MoE model PD separation mode (MoE模型 PD分离模式)")
            logger.debug(f"[MFUCalculator] model_name={self._model_name}")
            logger.debug(f"[MFUCalculator] num_params_per_device (total)={self._num_params_per_device / 1024 / 1024 / 1024:.4f} GB")
            logger.debug(f"[MFUCalculator] prefill_num_params_per_device={self._prefill_num_params_per_device / 1024 / 1024 / 1024:.4f} GB")
            logger.debug(f"[MFUCalculator] decode_num_params_per_device={self._decode_num_params_per_device / 1024 / 1024 / 1024:.4f} GB")
        else:
            # 普通模型：返回单个值
            # Normal model: returns single value
            self._num_params_per_device = param_counter.get_num_parameters_per_device()
            self._prefill_num_params_per_device = self._num_params_per_device
            self._decode_num_params_per_device = self._num_params_per_device
            
            logger.debug(f"[MFUCalculator] Normal model mode (普通模型模式)")
            logger.debug(f"[MFUCalculator] model_name={self._model_name}")
            logger.debug(f"[MFUCalculator] num_params_per_device={self._num_params_per_device}")

        model_config = replica_config.model_config

        self._num_layers_per_device = (
            model_config.num_layers // replica_config.num_pipeline_stages
        )
        self._num_heads_per_device = (
            model_config.num_q_heads // replica_config.tensor_parallel_size
        )
        self._head_dimension = model_config.embedding_dim // model_config.num_q_heads
        self._device_flops = replica_config.device_config.fp16_tflops * 2**40

    def _get_batch_stage_type(self, batch_stage: BatchStage) -> RequestType:
        """
        获取 batch_stage 的类型（prefill 或 decode）
        通过检查第一个 request 的类型来判断
        
        Get batch_stage type (prefill or decode)
        Determined by checking the first request's type
        """
        if not batch_stage.requests:
            return RequestType.MIXED
        # 假设同一个 batch_stage 中所有 request 类型相同
        # Assume all requests in the same batch_stage have the same type
        return batch_stage.requests[0].request_type

    def _get_mlp_flops(self, batch_stage: BatchStage) -> float:
        """
        计算 MLP 层的 FLOPs
        根据 batch_stage 类型选择对应的参数量
        
        Calculate MLP layer FLOPs
        Select corresponding parameter count based on batch_stage type
        """
        num_tokens = sum(batch_stage.num_tokens)
        
        # 对于 MoE 模型，根据 stage 类型选择参数量
        # For MoE models, select parameter count based on stage type
        if self._is_pd_separated_model:
            stage_type = self._get_batch_stage_type(batch_stage)
            if stage_type == RequestType.PREFILL:
                params = self._prefill_num_params_per_device
            elif stage_type == RequestType.DECODE:
                params = self._decode_num_params_per_device
            else:
                # MIXED 类型使用总参数量 | MIXED type uses total params
                params = self._num_params_per_device
        else:
            params = self._num_params_per_device
        
        return 2 * num_tokens * params

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
        
        # 防止除零错误：如果execution_time为0，返回0
        # Prevent division by zero: return 0 if execution_time is 0
        if batch_stage.execution_time == 0:
            logger.warning(f"batch_stage.execution_time is 0, returning MFU as 0")
            return 0.0
        
        total_flops_per_second = total_flops / batch_stage.execution_time
        return total_flops_per_second * 100 / self._device_flops
