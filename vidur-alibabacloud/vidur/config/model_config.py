from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import ActivationType, NormType

logger = init_logger(__name__)


@dataclass
class BaseModelConfig(BaseFixedConfig):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    mlp_hidden_dim: int
    max_position_embeddings: int
    use_gated_mlp: bool
    use_bias: bool
    use_qkv_bias: bool
    activation: ActivationType
    norm: NormType
    post_attn_norm: bool
    vocab_size: int
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False


@dataclass
class DeepseekV3ModelConfig(BaseModelConfig):
    # "num_hidden_layers": 61,
    num_layers: int = 61
    # "num_attention_heads": 128,
    num_q_heads: int = 128
    # "num_key_value_heads": 128,
    num_kv_heads: int = 128
    # "hidden_size": 7168,
    hidden_size: int = 7168
    embedding_dim: int = 7168
    # "intermediate_size": 18432,
    mlp_hidden_dim: int = 18432
    max_position_embeddings: int = 163840
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    #   "quantization_config": {
    #     "activation_scheme": "dynamic",
    #     "fmt": "e4m3",
    #     "quant_method": "fp8",
    #     "weight_block_size": [
    #       128,
    #       128
    #     ]
    #   },
    # "activation_scheme": "dynamic",
    # activation: ActivationType = ActivationType.SILU
    activation: ActivationType = ActivationType.DYNAMIC
    
    #   "rms_norm_eps": 1e-06,
    #   "rope_scaling": {
    #     "beta_fast": 32,
    #     "beta_slow": 1,
    #     "factor": 40,
    #     "mscale": 1.0,
    #     "mscale_all_dim": 1.0,
    #     "original_max_position_embeddings": 4096,
    #     "type": "yarn"
    #   },
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 129280
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    #   "rope_scaling": {
    #     "beta_fast": 32,
    #     "beta_slow": 1,
    #     "factor": 40,
    #     "mscale": 1.0,
    #     "mscale_all_dim": 1.0,
    #     "original_max_position_embeddings": 4096,
    #     "type": "yarn"
    #   },
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False
    # DeepseekV3 specific parameters
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mlp_intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1
    n_group: int = 8
    num_routed_experts: int = 256
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    topk_group: int = 4
    ep_size: int = 1
    first_k_dense_replace: int = 3
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    num_nextn_predict_layers: int = 1

    @staticmethod
    def get_name():
        return "deepseek-671B"
    
    
    

@dataclass
class Qwen3Next80BA3BModelConfig(BaseModelConfig):
    # architectures: list = field(default_factory=lambda: ["Qwen3NextForCausalLM"])
    architectures: str = "Qwen3NextForCausalLM"
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    decoder_sparse_step: int = 1
    eos_token_id: int = 151645
    full_attention_interval: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 5120
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128
    max_position_embeddings: int = 262144
    mlp_only_layers: list = field(default_factory=list)
    model_type: str = "qwen3_next"
    moe_intermediate_size: int = 512
    norm_topk_prob: bool = True
    num_attention_heads: int = 16
    num_experts: int = 512
    num_experts_per_tok: int = 10
    num_hidden_layers: int = 48
    num_key_value_heads: int = 2
    output_router_logits: bool = False
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000000
    router_aux_loss_coef: float = 0.001
    shared_expert_intermediate_size: int = 512
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.57.0.dev0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
    # Fields mapped from base class parameters / 与基类参数对应的字段
    num_layers: int = 48  # maps to num_hidden_layers / 对应 num_hidden_layers
    num_q_heads: int = 16  # maps to num_attention_heads / 对应 num_attention_heads
    num_kv_heads: int = 2  # maps to num_key_value_heads / 对应 num_key_value_heads
    embedding_dim: int = 2048  # maps to hidden_size / 对应 hidden_size
    mlp_hidden_dim: int = 5120  # maps to intermediate_size / 对应 intermediate_size
    use_gated_mlp: bool = True  # per model arch / 根据模型架构设定
    use_bias: bool = False  # per model arch / 根据模型架构设定
    use_qkv_bias: bool = False  # per model arch / 根据模型架构设定
    activation: ActivationType = ActivationType.SILU  # maps to hidden_act / 对应 hidden_act
    norm: NormType = NormType.RMS_NORM  # per model arch / 根据模型架构设定
    post_attn_norm: bool = True  # per model arch / 根据模型架构设定


    @staticmethod
    def get_name():
        return "qwen3-next-80B"
    
    
@dataclass
class Qwen3Moe235BA22BModelConfig(BaseModelConfig):
    # architectures: list = field(default_factory=lambda: ["Qwen3MoeForCausalLM"])
    architectures: str = "Qwen3MoeForCausalLM"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    decoder_sparse_step: int = 1
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 12288
    max_position_embeddings: int = 262144
    max_window_layers: int = 94
    mlp_only_layers: list = field(default_factory=list)
    model_type: str = "qwen3_moe"
    moe_intermediate_size: int = 1536
    norm_topk_prob: bool = True
    num_attention_heads: int = 64
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 94
    num_key_value_heads: int = 4
    output_router_logits: bool = False
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: float = 5000000
    router_aux_loss_coef: float = 0.001
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
    # Fields mapped from base class parameters / 与基类参数对应的字段
    num_layers: int = 94  # maps to num_hidden_layers / 对应 num_hidden_layers
    num_q_heads: int = 64  # maps to num_attention_heads / 对应 num_attention_heads
    num_kv_heads: int = 4  # maps to num_key_value_heads / 对应 num_key_value_heads
    embedding_dim: int = 4096  # maps to hidden_size / 对应 hidden_size
    mlp_hidden_dim: int = 12288  # maps to intermediate_size / 对应 intermediate_size
    use_gated_mlp: bool = True  # per model arch / 根据模型架构设定
    use_bias: bool = False  # per model arch / 根据模型架构设定
    use_qkv_bias: bool = False  # per model arch / 根据模型架构设定
    activation: ActivationType = ActivationType.SILU  # maps to hidden_act / 对应 hidden_act
    norm: NormType = NormType.RMS_NORM  # per model arch / 根据模型架构设定
    post_attn_norm: bool = True  # per model arch / 根据模型架构设定



    @staticmethod
    def get_name():
        # return "qwen3-235B-A22B"
        return "qwen3-moe-235B"


@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = 16384
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 32768
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-Config"


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class MyLlama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = None
    vocab_size: int = 32000

    @staticmethod
    def get_name():
        return "NousResearch/Llama-2-7b-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-70b-hf"


@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-70B"


@dataclass
class InternLMModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 4096
    vocab_size: int = 103168


@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):
    num_layers: int = 60
    num_q_heads: int = 40
    num_kv_heads: int = 40
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 13824

    @staticmethod
    def get_name():
        return "internlm/internlm-20b"


@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 32768
    vocab_size: int = 92544


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 16384
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240
    max_position_embeddings: int = 2048
    use_gated_mlp: bool = False
    use_bias: bool = True
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.GELU
    norm: NormType = NormType.LAYER_NORM
    post_attn_norm: bool = False
    vocab_size: int = 51200
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = 10000
    partial_rotary_factor: float = 0.4
    no_tensor_parallel: bool = True

    @staticmethod
    def get_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = True
    max_position_embeddings: int = 32768
    vocab_size: int = 152064

    @staticmethod
    def get_name():
        return "Qwen/Qwen-Config"


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 64
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 24576
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "Qwen/Qwen-72B"


