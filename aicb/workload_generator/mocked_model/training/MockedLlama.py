"""
Mocked LLaMA model for AICB workload generation.

Models LLaMA 2/3/4 architecture with:
  - Group Query Attention (GQA): fewer KV heads than Q heads
  - SwiGLU FFN: gate_proj, up_proj, down_proj (3 projections)
  - RMSNorm pre-normalization (weight-only, no bias)
  - RoPE (Rotary Position Embedding): cached cos/sin tensors
  - Pre-norm decoder layer:
      x -> RMSNorm -> Attention -> residual(+x) -> RMSNorm -> MLP -> residual(+)

Reuses MegatronColumnLinear / MegatronRowLinear for tensor-parallel sharding
and MegatronEmbedding for the embedding layer.

Supported configs:
  LLaMA-7B:   hidden=4096,  intermediate=11008, num_heads=32,  num_kv_heads=32,  layers=32
  LLaMA-13B:  hidden=5120,  intermediate=13824, num_heads=40,  num_kv_heads=40,  layers=40
  LLaMA-70B:  hidden=8192,  intermediate=28672, num_heads=64,  num_kv_heads=8,   layers=80
  LLaMA-3-8B: hidden=4096,  intermediate=14336, num_heads=32,  num_kv_heads=8,   layers=32
  LLaMA-3-70B: hidden=8192, intermediate=28672, num_heads=64,  num_kv_heads=8,   layers=80
  LLaMA-4-Scout:  hidden=5120,  intermediate=14336, num_heads=40,  num_kv_heads=8,  layers=48, MoE(16)
  LLaMA-4-Maverick: hidden=..., MoE(128)

Based on MockedMegatron.py patterns.
File: MockedLlama.py
License: Apache 2.0
"""

import math
from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, MockedParam
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronColumnLinear,
    MegatronRowLinear,
    MegatronEmbedding,
)
from log_analyzer.log import Workload, LogItem


# ---------------------------------------------------------------------------
# LlamaRMSNorm -- Root Mean Square Layer Normalization
# ---------------------------------------------------------------------------
class LlamaRMSNorm(MockedModel):
    """RMSNorm: weight * x * rsqrt(mean(x^2) + eps)

    Unlike LayerNorm, RMSNorm has:
      - NO bias parameter
      - NO mean-subtraction step
      - ONLY a learnable weight vector of size hidden_size

    In terms of mocked parameters, this means half the parameter count
    of a standard LayerNorm (no bias), and no communication operations.
    """

    def __init__(self, hidden_size, eps=1e-6, name=""):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.name = name
        self.weight = MockedParam((hidden_size,), name=f"{name}_weight" if name else "rmsnorm_weight")

    def parameters(self):
        return [self.weight]

    def activation_memory(self):
        # RMSNorm stores the input activations for backward pass
        # Input: (seq_len, batch_size, hidden_size) per token
        return self.hidden_size


# ---------------------------------------------------------------------------
# LlamaRotaryEmbedding -- Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------
class LlamaRotaryEmbedding(MockedModel):
    """RoPE: applies rotary position embeddings to Q and K tensors.

    RoPE does NOT have trainable parameters. It caches cos and sin
    values computed from the position indices and the rotary base frequency.

    In the mocked model, RoPE contributes:
      - Zero trainable parameters (cos/sin are computed, not learned)
      - Activation memory for the cached cos/sin tensors
      - No communication operations (purely local computation)
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.name = "rotary_emb"

        # Cache for cos and sin values: (max_position_embeddings, dim)
        # Each is float32 (4 bytes) -- stored as activation memory
        self.cos_cached_size = max_position_embeddings * dim
        self.sin_cached_size = max_position_embeddings * dim

    def parameters(self):
        # RoPE has no trainable parameters
        return []

    def activation_memory(self):
        # cos_cached + sin_cached in float32 (4 bytes each)
        return (self.cos_cached_size + self.sin_cached_size) * 4


# ---------------------------------------------------------------------------
# LlamaMLP -- SwiGLU Feed-Forward Network (gate-up-down)
# ---------------------------------------------------------------------------
class LlamaMLP(MockedModel):
    """SwiGLU MLP: SiLU(gate_proj(x)) * up_proj(x) -> down_proj

    Three linear projections instead of Megatron's two (up-down).
    Gate and up operate in parallel; their outputs are multiplied
    element-wise before the down projection.

    Parameter count:
      gate: hidden_size * intermediate_size  (TP-sharded)
      up:   hidden_size * intermediate_size  (TP-sharded)
      down: intermediate_size * hidden_size  (TP-sharded)
      Total: 3 * hidden_size * intermediate_size
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = "mlp_layer_swiglu"
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp = tp
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Gate projection: hidden -> intermediate (TP-sharded output)
        self.gate_proj = MegatronColumnLinear(
            hidden_size,
            intermediate_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mlp_gate",
            sequence_parallel_enabled,
            computation_enable,
            name="mlp_gate_column",
            add_bias_linear=add_bias_linear,
        )

        # Up projection: hidden -> intermediate (TP-sharded output)
        self.up_proj = MegatronColumnLinear(
            hidden_size,
            intermediate_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mlp_up",
            sequence_parallel_enabled,
            computation_enable,
            name="mlp_up_column",
            add_bias_linear=add_bias_linear,
        )

        # Down projection: intermediate -> hidden (TP-sharded input)
        self.down_proj = MegatronRowLinear(
            intermediate_size,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mlp_down",
            sequence_parallel_enabled,
            computation_enable,
            name="mlp_down_row",
            add_bias_linear=add_bias_linear,
        )

    def activation_memory(self):
        # Input activations to the MLP block: (seq_len, batch_size, hidden_size)
        return self.seq_len * self.batch_size * self.hidden_size

    def forward(self):
        workloads = Workload()
        workloads.extend(self.gate_proj.forward())
        workloads.extend(self.up_proj.forward())
        workloads.extend(self.down_proj.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.down_proj.backward())
        workloads.extend(self.up_proj.backward())
        workloads.extend(self.gate_proj.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# LlamaAttention -- Group Query Attention (GQA)
# ---------------------------------------------------------------------------
class LlamaAttention(MockedModel):
    """Group Query Attention: num_kv_heads <= num_attention_heads.

    Q projection uses num_attention_heads; K/V projections use num_kv_heads.
    When num_kv_heads < tp_size, K/V heads are replicated (kv_tp capped).

    TP sharding:
      - Q: num_attention_heads / tp
      - K: num_kv_heads / kv_tp  (kv_tp = min(num_kv_heads, tp))
      - V: num_kv_heads / kv_tp
      - O: num_attention_heads / tp  (reduces back to hidden_size)

    Communication sizes (per TP group):
      - Q/K/V forward:  each all_gather = seq_len * batch_size * (heads/tp) * head_dim
      - O forward:       reduce_scatter = seq_len * batch_size * num_heads * head_dim

    GQA reduces K/V communication by factor (num_kv_heads / num_attention_heads).
    """

    def __init__(
        self,
        num_attention_heads,
        num_kv_heads,
        hidden_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = "attention_layer_gqa"
        self.layer_id = layer_id
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_size = hidden_size
        self.tp = tp
        self.seq_len = seq_len
        self.batch_size = batch_size

        # TP degree for K/V: capped at num_kv_heads (heads cannot be split
        # beyond their count; extra GPUs replicate K/V at no comm cost)
        self.kv_tp = min(num_kv_heads, tp)

        # Q projection: all heads, full TP
        self.q_proj = MegatronColumnLinear(
            hidden_size,
            num_attention_heads * self.head_dim,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_q",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_q_column",
            add_bias_linear=add_bias_linear,
        )

        # K projection: kv_heads only, kv_tp
        self.k_proj = MegatronColumnLinear(
            hidden_size,
            num_kv_heads * self.head_dim,
            self.kv_tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_k",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_k_column",
            add_bias_linear=add_bias_linear,
        )

        # V projection: kv_heads only, kv_tp
        self.v_proj = MegatronColumnLinear(
            hidden_size,
            num_kv_heads * self.head_dim,
            self.kv_tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_v",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_v_column",
            add_bias_linear=add_bias_linear,
        )

        # O projection: merges all Q heads back to hidden
        self.o_proj = MegatronRowLinear(
            num_attention_heads * self.head_dim,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_o",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_o_row",
            add_bias_linear=add_bias_linear,
        )

    def activation_memory(self):
        # Input activations to attention: (seq_len, batch_size, hidden_size)
        return self.seq_len * self.batch_size * self.hidden_size

    def forward(self):
        workloads = Workload()
        workloads.extend(self.q_proj.forward())
        workloads.extend(self.k_proj.forward())
        workloads.extend(self.v_proj.forward())
        workloads.extend(self.o_proj.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.o_proj.backward())
        workloads.extend(self.v_proj.backward())
        workloads.extend(self.k_proj.backward())
        workloads.extend(self.q_proj.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# LlamaDecoderLayer -- Pre-norm Transformer Block
# ---------------------------------------------------------------------------
class LlamaDecoderLayer(MockedModel):
    """LLaMA decoder layer with pre-normalization.

    Structure:
      x -> input_norm(RMSNorm) -> attention -> residual(+x)
        -> post_attn_norm(RMSNorm) -> mlp -> residual(+)

    This is the standard LLaMA pre-norm architecture. Unlike Megatron's
    post-norm design, the norm operations precede their respective sub-layers.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_kv_heads,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        super().__init__()
        self.name = f"llama_layer_{layer_id}"
        self.layer_id = layer_id
        self.hidden_size = hidden_size

        # Pre-attention RMSNorm
        self.input_layernorm = LlamaRMSNorm(
            hidden_size, name=f"input_norm_{layer_id}"
        )

        # Group Query Attention
        self.self_attn = LlamaAttention(
            num_attention_heads,
            num_kv_heads,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            sequence_parallel_enabled,
            computation_enable,
            add_bias_linear,
        )

        # Post-attention RMSNorm
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size, name=f"post_attn_norm_{layer_id}"
        )

        # SwiGLU MLP
        self.mlp = LlamaMLP(
            hidden_size,
            intermediate_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            sequence_parallel_enabled,
            computation_enable,
            add_bias_linear,
        )

    def activation_memory(self):
        return (
            self.input_layernorm.activation_memory()
            + self.self_attn.activation_memory()
            + self.post_attention_layernorm.activation_memory()
            + self.mlp.activation_memory()
        )

    def forward(self):
        workloads = Workload()
        # pre-norm: input_layernorm -> attention, post_attn_norm -> mlp
        # Norm ops have no communication, only computation
        workloads.extend(self.self_attn.forward())
        workloads.extend(self.mlp.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.mlp.backward())
        workloads.extend(self.self_attn.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads


# ---------------------------------------------------------------------------
# LlamaModel -- Complete LLaMA Architecture
# ---------------------------------------------------------------------------
class LlamaModel(MockedModel):
    """LLaMA model: Embedding -> N x DecoderLayer -> FinalNorm -> LM Head.

    The standard LLaMA architecture uses:
      - Pre-norm (RMSNorm before attention and MLP)
      - Group Query Attention (GQA)
      - SwiGLU feed-forward network
      - RoPE position embeddings (no learned positions)
      - Usually untied embedding and lm_head weights

    Config expects these fields (set by get_params / config file):
      hidden_size, ffn_hidden_size (intermediate), num_layers,
      num_attention_heads, num_kv_heads, seq_length, micro_batch,
      tensor_model_parallel_size, padded_vocab_size,
      enable_sequence_parallel, computation_enable, add_bias_linear
    """

    def __init__(self, config):
        super().__init__()
        num_kv_heads = getattr(config, "num_kv_heads", None)
        if num_kv_heads is None:
            num_kv_heads = config.num_attention_heads

        # Embedding layer (reuses Megatron embedding with TP support)
        self.embedding = MegatronEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
        )

        # RoPE: cached cos/sin for max_position_embeddings
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=getattr(
                config, "max_position_embeddings", config.seq_length
            ),
            base=getattr(config, "rope_theta", 10000.0),
        )

        # Decoder layers
        self.layers = [
            LlamaDecoderLayer(
                config.hidden_size,
                config.ffn_hidden_size,
                config.num_attention_heads,
                num_kv_heads,
                config.tensor_model_parallel_size,
                config.seq_length,
                config.micro_batch,
                i,
                config.enable_sequence_parallel,
                config.computation_enable,
                config.add_bias_linear,
            )
            for i in range(config.num_layers)
        ]

        # Final RMSNorm (precedes LM head)
        self.final_norm = LlamaRMSNorm(config.hidden_size, name="final_norm")

        # LM Head: hidden -> vocab (TP-sharded, no bias in LLaMA)
        self.lm_head = MegatronColumnLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
            config.num_layers + 1,
            "lm_head",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=False,
        )

    def activation_memory(self):
        total = 0
        total += self.rotary_emb.activation_memory()
        for layer in self.layers:
            total += layer.activation_memory()
        return total

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        assert all(isinstance(w, LogItem) for w in workloads.workload)
        return workloads
