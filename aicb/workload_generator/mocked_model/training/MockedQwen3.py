"""
Mocked Qwen3 dense model for AICB training workload generation.

Based on MockedMegatron.py.  Qwen3-specific changes versus LLAMA / GPT-3:

  * GQA  – separate num_key_value_heads  (Megatron hardcodes MHA)
  * head_dim from config (128) instead of hidden_size // num_attention_heads
  * QK-Norm – compute-only RMSNorm on query / key  (hardcoded, always on)
  * MLP fix – SwiGLU down-projection uses intermediate_size, not 2*intermediate

Imports MegatronColumnLinear, MegatronRowLinear, MegatronEmbedding, MOEMLP
from MockedMegatron so TP communication primitives are NOT duplicated.
"""

from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, MockedParam, MockedParamsBase
from log_analyzer.log import Workload, LogItem

# ---------------------------------------------------------------------------
# IMPORT: reuse Megatron's TP communication primitives unchanged
# ---------------------------------------------------------------------------
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronColumnLinear,
    MegatronRowLinear,
    MegatronEmbedding,
    MOEMLP,
)


# ===================================================================
# Qwen3-specific modules  (everything below is new)
# ===================================================================

class Qwen3RMSNorm(MockedModel):
    """RMSNorm weight parameter.  Used for layernorm and QK-Norm."""

    def __init__(self, dim, prefix_name="norm", layer_id=0):
        self.layer_id = layer_id
        self.name = prefix_name
        self.weight = MockedParam((dim, 1), name=prefix_name)


# ---------------------------------------------------------------------------
# Attention  (GQA + hardcoded QK-Norm)
# ---------------------------------------------------------------------------

class Qwen3Attention(MockedModel):
    """
    Qwen3 attention with Grouped Query Attention and QK-Norm.

    Projection sizes (GQA-aware):
        q_proj:  hidden_size -> num_attention_heads * head_dim
        k_proj:  hidden_size -> num_key_value_heads  * head_dim
        v_proj:  hidden_size -> num_key_value_heads  * head_dim
        o_proj:  num_attention_heads * head_dim -> hidden_size

    The three input projections are fused into a single MegatronColumnLinear
    so the ALLGATHER on the (unchanged) hidden_size input happens once.

    QK-Norm (hardcoded, always on):
      - two RMSNorm(head_dim) applied per-head after projection, before RoPE
      - compute-only – zero communication impact
      - 256 floats per layer → negligible parameter-count impact
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        self.name = "attention_layer"
        self.layer_id = layer_id

        self.query_projection_size = num_attention_heads * head_dim
        self.kv_projection_size   = num_key_value_heads  * head_dim

        # Fused QKV column – 1 ALLGATHER on hidden_size, same comm vol as Megatron
        self.qkv = MegatronColumnLinear(
            hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            tp, seq_len, batch_size,
            layer_id, "attention",
            sequence_parallel_enabled, computation_enable,
            name="attention_qkv_column",
            add_bias_linear=add_bias_linear,
        )

        # QK-Norm – compute-only, no collectives
        self.q_norm = Qwen3RMSNorm(head_dim, "attention_q_norm", layer_id)
        self.k_norm = Qwen3RMSNorm(head_dim, "attention_k_norm", layer_id)

        # Output projection – REDUCESCATTER on hidden_size, same as Megatron
        self.attention_dense = MegatronRowLinear(
            self.query_projection_size, hidden_size,
            tp, seq_len, batch_size,
            layer_id, "attention",
            sequence_parallel_enabled, computation_enable,
            name="attention_row",
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        w = Workload()
        w.extend(self.qkv.forward())
        # qk_norm.forward() is intentionally NOT called — pure compute, no comms
        w.extend(self.attention_dense.forward())
        return w

    def backward(self):
        w = Workload()
        w.extend(self.qkv.backward())
        w.extend(self.attention_dense.backward())
        return w


# ---------------------------------------------------------------------------
# SwiGLU MLP  (fixes the MegatronMLP sizing issue)
# ---------------------------------------------------------------------------

class Qwen3Mlp(MockedModel):
    """
    Qwen3 SwiGLU MLP with CORRECT projection sizing.

    Gate + Up  (fused column):  hidden_size -> 2 * intermediate_size
    Down       (row):           intermediate_size -> hidden_size

    NOTE: MegatronMlp uses the SAME `ffn_hidden_size` for both column and row.
    For SwiGLU that overcounts down-projection params by 2×.  This class fixes it.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        tp, seq_len, batch_size, layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        self.name = "mlp_layer"
        self.layer_id = layer_id

        # gate_proj + up_proj fused  (column – ALLGATHER on hidden_size)
        self.gate_up = MegatronColumnLinear(
            hidden_size,
            2 * intermediate_size,
            tp, seq_len, batch_size,
            layer_id, "mlp",
            sequence_parallel_enabled, computation_enable,
            name="mlp_gate_up_column",
            add_bias_linear=add_bias_linear,
        )

        # down_proj  (row – REDUCESCATTER on hidden_size)
        # input is intermediate_size, NOT 2*intermediate  (the fix)
        self.down = MegatronRowLinear(
            intermediate_size,
            hidden_size,
            tp, seq_len, batch_size,
            layer_id, "mlp",
            sequence_parallel_enabled, computation_enable,
            name="mlp_down_row",
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        w = Workload()
        w.extend(self.gate_up.forward())
        w.extend(self.down.forward())
        return w

    def backward(self):
        w = Workload()
        w.extend(self.gate_up.backward())
        w.extend(self.down.backward())
        return w


# ---------------------------------------------------------------------------
# Transformer Layer  (pre-norm, same structure as LLaMA)
# ---------------------------------------------------------------------------

class Qwen3TransformerLayer(MockedModel):
    """
    input -> input_layernorm -> attention (+ residual)
          -> post_attention_layernorm -> mlp (+ residual)

    Supports both dense (Qwen3Mlp) and MoE (MOEMLP) FFN via moe_enable.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        tp, seq_len, batch_size, layer_id,
        # MoE params (ignored when moe_enable=False)
        moe_enable=False,
        moe_intermediate_size=0,
        moe_router_topk=0,
        num_experts=0,
        expert_model_parallel_size=1,
        # Standard params
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        self.name = "transformer_layer"
        self.layer_id = layer_id

        self.input_layernorm = Qwen3RMSNorm(hidden_size, "input_layernorm", layer_id)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, "post_attn_norm", layer_id)

        self.attention = Qwen3Attention(
            hidden_size, num_attention_heads, num_key_value_heads, head_dim,
            tp, seq_len, batch_size, layer_id,
            sequence_parallel_enabled, computation_enable, add_bias_linear,
        )

        if moe_enable:
            self.mlp = MOEMLP(
                batch_size, hidden_size, tp, expert_model_parallel_size,
                moe_intermediate_size, seq_len, moe_router_topk,
                num_experts, layer_id,
            )
        else:
            self.mlp = Qwen3Mlp(
                hidden_size, intermediate_size,
                tp, seq_len, batch_size, layer_id,
                sequence_parallel_enabled, computation_enable, add_bias_linear,
            )

    def forward(self):
        w = Workload()
        w.extend(self.attention.forward())
        w.extend(self.mlp.forward())
        return w

    def backward(self):
        w = Workload()
        w.extend(self.attention.backward())
        w.extend(self.mlp.backward())
        return w


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Qwen3Model(MockedModel):
    """
    Complete Qwen3 dense model.

    Config attributes consumed (from HF config.json):
        hidden_size, intermediate_size, num_hidden_layers,
        num_attention_heads, num_key_value_heads, head_dim, vocab_size

    Training attributes (from CLI / args):
        tensor_model_parallel_size, seq_length, micro_batch,
        enable_sequence_parallel, computation_enable, add_bias_linear
    """

    def __init__(self, config):
        h = config.hidden_size
        ff = config.intermediate_size
        n_heads = config.num_attention_heads
        n_kv = config.num_key_value_heads
        h_dim = getattr(config, "head_dim", h // n_heads)
        n_layers = config.num_hidden_layers
        V = config.vocab_size
        tp = config.tensor_model_parallel_size
        S = config.seq_length
        B = config.micro_batch
        sp = getattr(config, "enable_sequence_parallel", True)
        comp = getattr(config, "computation_enable", False)
        bias = getattr(config, "add_bias_linear", False)

        # MoE config (ignored when moe_enable=False)
        moe = getattr(config, "moe_enable", False)
        moe_ff = getattr(config, "moe_intermediate_size", 0)
        moe_topk = getattr(config, "moe_router_topk", 0)
        moe_n_exp = getattr(config, "num_experts", 0)
        moe_ep = getattr(config, "expert_model_parallel_size", 1)

        self.embedding = MegatronEmbedding(V, h, tp, S, B)

        self.layers = [
            Qwen3TransformerLayer(
                h, ff, n_heads, n_kv, h_dim,
                tp, S, B, i,
                moe_enable=moe,
                moe_intermediate_size=moe_ff,
                moe_router_topk=moe_topk,
                num_experts=moe_n_exp,
                expert_model_parallel_size=moe_ep,
                sequence_parallel_enabled=sp,
                computation_enable=comp,
                add_bias_linear=bias,
            )
            for i in range(n_layers)
        ]

        self.final_norm = Qwen3RMSNorm(h, "final_norm", n_layers + 1)

        self.lm_head = MegatronColumnLinear(
            h, V, tp, S, B, n_layers + 2, "lm_head",
            sequence_parallel_enabled=sp,
            computation_enable=comp,
            add_bias_linear=bias,
        )

    def forward(self):
        w = Workload()
        w.extend(self.embedding.forward())
        for layer in self.layers:
            w.extend(layer.forward())
        w.extend(self.lm_head.forward())
        return w

    def backward(self):
        w = Workload()
        w.extend(self.lm_head.backward())
        for layer in reversed(self.layers):
            w.extend(layer.backward())
        w.extend(self.embedding.backward())
        return w


# ===================================================================
# Config / params class
# ===================================================================

class Qwen3Params(MockedParamsBase):
    """
    Qwen3 training configuration.

    Loads HF config.json via load_from_config (inherited).
    Overrides via command-line args via load_from_args (inherited).

    Default values are for Qwen3-8B.
    """

    def __init__(self, config_file=None, args=None):
        # --- HF config.json defaults (Qwen3-8B) ---
        self.hidden_size = 4096
        self.intermediate_size = 12288
        self.num_hidden_layers = 36
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.vocab_size = 151936
        self.rms_norm_eps = 1e-6
        self.rope_theta = 1_000_000
        self.max_position_embeddings = 40960
        self.tie_word_embeddings = False
        self.attention_bias = False

        # --- Training defaults ---
        self.tensor_model_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.pipeline_model_parallel = 1
        self.world_size = 1
        self.seq_length = 4096
        self.micro_batch = 1
        self.computation_enable = False
        self.enable_sequence_parallel = True
        self.add_bias_linear = False
        self.moe_enable = False
        self.moe_intermediate_size = 0
        self.moe_router_topk = 0
        self.num_experts = 0

        super().__init__("Qwen3", "Qwen3", config_file, args)


# ===================================================================
# Smoke test
# ===================================================================

if __name__ == "__main__":
    import sys, os

    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    if config_file and not os.path.exists(config_file):
        print(f"[ERROR] config file not found: {config_file}")
        sys.exit(1)

    cfg = Qwen3Params(config_file)
    model = Qwen3Model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.msg_size() for p in model.parameters())

    print(f"Qwen3  |  {cfg.hidden_size=}  {cfg.intermediate_size=}")
    print(f"       |  {cfg.num_hidden_layers=}  {cfg.num_attention_heads=}  {cfg.num_key_value_heads=}  {cfg.head_dim=}")
    print(f"       |  {cfg.vocab_size=}  tp={cfg.tensor_model_parallel_size}  seq={cfg.seq_length}  mbs={cfg.micro_batch}")
    print(f"       |  total params: {total_params:,}  ({total_bytes/1e9:.2f} GB BF16)")

    fwd = model.forward()
    bwd = model.backward()
    print(f"       |  fwd workloads: {len(fwd.workload)}  bwd workloads: {len(bwd.workload)}")
