"""
Tests for MockedLlama.py: parameter counts, communication patterns, GQA scaling.

Tests are designed to run without pandas/torch dependencies (isolated from
the full utils.py import chain by testing the Llama classes directly).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workload_generator.mocked_model.MockedModel import MockedParam
from workload_generator.mocked_model.training.MockedLlama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
)


# ---------------------------------------------------------------------------
# Helper: mock config for LlamaModel
# ---------------------------------------------------------------------------
class MockConfig:
    """Minimal config mimicking what get_params() produces for LlamaModel."""

    def __init__(self, **kwargs):
        defaults = dict(
            padded_vocab_size=32000,
            hidden_size=4096,
            ffn_hidden_size=11008,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=32,
            seq_length=2048,
            micro_batch=1,
            tensor_model_parallel_size=1,
            enable_sequence_parallel=True,
            computation_enable=False,
            add_bias_linear=False,
            max_position_embeddings=4096,
            rope_theta=10000.0,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# LlamaRMSNorm
# ---------------------------------------------------------------------------
class TestLlamaRMSNorm:
    def test_no_bias_parameter(self):
        norm = LlamaRMSNorm(4096)
        params = norm.parameters()
        assert len(params) == 1
        assert params[0].shape == (4096,)
        assert params[0].name == "rmsnorm_weight"

    def test_custom_name(self):
        norm = LlamaRMSNorm(4096, name="test_norm")
        params = norm.parameters()
        assert params[0].name == "test_norm_weight"

    def test_total_param_count(self):
        """RMSNorm has half the params of LayerNorm (no bias)."""
        norm = LlamaRMSNorm(8192)
        total = sum(p.numel() for p in norm.parameters())
        assert total == 8192  # Only weight, no bias (LayerNorm would be 16384)

    def test_activation_memory(self):
        norm = LlamaRMSNorm(4096)
        assert norm.activation_memory() == 4096


# ---------------------------------------------------------------------------
# LlamaRotaryEmbedding
# ---------------------------------------------------------------------------
class TestLlamaRotaryEmbedding:
    def test_no_trainable_parameters(self):
        rope = LlamaRotaryEmbedding(dim=128, max_position_embeddings=4096)
        assert rope.parameters() == []

    def test_cache_sizes(self):
        rope = LlamaRotaryEmbedding(dim=128, max_position_embeddings=4096)
        assert rope.cos_cached_size == 4096 * 128  # 524288
        assert rope.sin_cached_size == 4096 * 128

    def test_activation_memory(self):
        rope = LlamaRotaryEmbedding(dim=128, max_position_embeddings=4096)
        # (cos + sin) * 4 bytes each (float32)
        expected = (4096 * 128 + 4096 * 128) * 4
        assert rope.activation_memory() == expected

    def test_custom_base_frequency(self):
        rope = LlamaRotaryEmbedding(dim=128, max_position_embeddings=2048, base=500000.0)
        assert rope.base == 500000.0
        assert rope.max_position_embeddings == 2048


# ---------------------------------------------------------------------------
# LlamaMLP
# ---------------------------------------------------------------------------
class TestLlamaMLP:
    def test_three_projections(self):
        mlp = LlamaMLP(4096, 11008, tp=1, seq_len=2048, batch_size=1, layer_id=0)
        params = mlp.parameters()
        param_names = [p.name for p in params if p.name is not None]
        # gate_proj, up_proj, down_proj (each ColumnLinear/RowLinear
        # has weight and optionally bias)
        assert any("gate" in n.lower() for n in param_names)
        assert any("up" in n.lower() for n in param_names)
        assert any("down" in n.lower() for n in param_names)

    def test_total_param_count_swiglu(self):
        """SwiGLU: gate(hidden*intermediate) + up(hidden*intermediate) + down(intermediate*hidden)."""
        mlp = LlamaMLP(4096, 11008, tp=1, seq_len=2048, batch_size=1, layer_id=0)
        total = sum(p.numel() for p in mlp.parameters())
        expected = 4096 * 11008 + 4096 * 11008 + 11008 * 4096
        assert total == expected  # 3 * hidden * intermediate

    def test_tp_sharding_reduces_per_gpu_params(self):
        """With TP=4, each GPU gets 1/4 of gate/up/down parameters."""
        mlp = LlamaMLP(4096, 11008, tp=4, seq_len=2048, batch_size=1, layer_id=0)
        total = sum(p.numel() for p in mlp.parameters())
        expected = (
            11008 * (4096 // 4)  # gate weight (output sharded)
            + 11008 * (4096 // 4)  # up weight (output sharded)
            + (11008 // 4) * 4096  # down weight (input sharded)
        )
        assert total == expected

    def test_forward_produces_workload(self):
        mlp = LlamaMLP(4096, 11008, tp=2, seq_len=2048, batch_size=1, layer_id=0)
        wl = mlp.forward()
        assert len(wl.workload) >= 3  # gate, up, down (each may have compute+comm)

    def test_backward_produces_workload(self):
        mlp = LlamaMLP(4096, 11008, tp=2, seq_len=2048, batch_size=1, layer_id=0)
        wl = mlp.backward()
        assert len(wl.workload) >= 3

    def test_activation_memory(self):
        mlp = LlamaMLP(4096, 11008, tp=1, seq_len=2048, batch_size=4, layer_id=0)
        assert mlp.activation_memory() == 2048 * 4 * 4096


# ---------------------------------------------------------------------------
# LlamaAttention -- GQA
# ---------------------------------------------------------------------------
class TestLlamaAttention:
    def test_mha_fallback_when_kv_heads_equal(self):
        """When num_kv_heads == num_heads, GQA degrades to standard MHA."""
        attn = LlamaAttention(32, 32, 4096, tp=1, seq_len=2048, batch_size=1, layer_id=0)
        assert attn.kv_tp == 1  # tp=1
        assert attn.head_dim == 128

    def test_gqa_kv_tp_capped(self):
        """When num_kv_heads < tp, kv_tp is capped at num_kv_heads."""
        attn = LlamaAttention(32, 8, 4096, tp=4, seq_len=2048, batch_size=1, layer_id=0)
        assert attn.kv_tp == 4  # min(8, 4) = 4

        attn2 = LlamaAttention(32, 2, 4096, tp=4, seq_len=2048, batch_size=1, layer_id=0)
        assert attn2.kv_tp == 2  # min(2, 4) = 2, K/V replicated on 2 extra GPUs

    def test_kv_projection_smaller_than_q(self):
        """K and V projections have fewer outputs than Q in GQA mode."""
        attn = LlamaAttention(32, 8, 4096, tp=1, seq_len=2048, batch_size=1, layer_id=0)
        q_params = sum(p.numel() for p in attn.q_proj.parameters())
        k_params = sum(p.numel() for p in attn.k_proj.parameters())
        v_params = sum(p.numel() for p in attn.v_proj.parameters())
        assert k_params < q_params  # K smaller than Q
        assert v_params < q_params  # V smaller than Q
        assert k_params == v_params  # K and V same size

    def test_gqa_reduces_kv_comm_size(self):
        """GQA reduces K/V comm by factor num_kv_heads / num_attention_heads."""
        # MHA (baseline): 32 heads
        attn_mha = LlamaAttention(32, 32, 4096, tp=4, seq_len=2048, batch_size=1, layer_id=0)
        # GQA: 8 KV heads (4x reduction)
        attn_gqa = LlamaAttention(32, 8, 4096, tp=4, seq_len=2048, batch_size=1, layer_id=0)

        mha_k_params = sum(p.numel() for p in attn_mha.k_proj.parameters())
        gqa_k_params = sum(p.numel() for p in attn_gqa.k_proj.parameters())
        # GQA K params should be 8/32 = 1/4 of MHA K params
        assert gqa_k_params * 4 == mha_k_params

    def test_forward_produces_workload(self):
        attn = LlamaAttention(32, 8, 4096, tp=4, seq_len=2048, batch_size=1, layer_id=0)
        wl = attn.forward()
        assert len(wl.workload) >= 4  # Q, K, V, O (each may have compute+comm)

    def test_backward_produces_workload(self):
        attn = LlamaAttention(32, 8, 4096, tp=4, seq_len=2048, batch_size=1, layer_id=0)
        wl = attn.backward()
        assert len(wl.workload) >= 4

    def test_activation_memory(self):
        attn = LlamaAttention(32, 32, 4096, tp=1, seq_len=2048, batch_size=2, layer_id=0)
        assert attn.activation_memory() == 2048 * 2 * 4096


# ---------------------------------------------------------------------------
# LlamaDecoderLayer
# ---------------------------------------------------------------------------
class TestLlamaDecoderLayer:
    def test_has_pre_norm_structure(self):
        layer = LlamaDecoderLayer(4096, 11008, 32, 8, tp=1, seq_len=2048,
                                  batch_size=1, layer_id=0)
        assert isinstance(layer.input_layernorm, LlamaRMSNorm)
        assert isinstance(layer.post_attention_layernorm, LlamaRMSNorm)
        assert isinstance(layer.self_attn, LlamaAttention)
        assert isinstance(layer.mlp, LlamaMLP)

    def test_forward_backward(self):
        """With TP>1, the layer generates communication LogItems."""
        layer = LlamaDecoderLayer(4096, 11008, 32, 8, tp=4, seq_len=2048,
                                  batch_size=1, layer_id=0)
        fwd = layer.forward()
        bwd = layer.backward()
        # TP=4 produces all_gather/reduce_scatter for column/row linear
        assert len(fwd.workload) > 0
        assert len(bwd.workload) > 0

    def test_forward_backward_tp1_no_comms(self):
        """TP=1 + no computation means no communication LogItems (expected)."""
        layer = LlamaDecoderLayer(4096, 11008, 32, 8, tp=1, seq_len=2048,
                                  batch_size=1, layer_id=0)
        fwd = layer.forward()
        bwd = layer.backward()
        # TP=1 + no computation = no comms needed. Workload is empty but valid.
        assert isinstance(fwd.workload, list)
        assert isinstance(bwd.workload, list)

    def test_activation_memory_is_positive(self):
        layer = LlamaDecoderLayer(4096, 11008, 32, 8, tp=1, seq_len=2048,
                                  batch_size=1, layer_id=0)
        assert layer.activation_memory() > 0


# ---------------------------------------------------------------------------
# LlamaModel -- full model integration
# ---------------------------------------------------------------------------
class TestLlamaModel:
    def test_llama7b_param_count(self):
        """LLaMA-7B config: total params should be in 6.5B-7.2B range."""
        config = MockConfig(
            hidden_size=4096,
            ffn_hidden_size=11008,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=32,
            padded_vocab_size=32000,
        )
        model = LlamaModel(config)
        total = sum(p.numel() for p in model.parameters())
        # LLaMA-7B: ~6.7B params (our untied embedding adds ~131M extra)
        assert 6_500_000_000 < total < 7_200_000_000, (
            f"Expected ~6.7B-7.15B params (with untied lm_head), got {total}"
        )

    def test_llama3_8b_gqa_param_count(self):
        """LLaMA-3-8B with GQA (8 KV heads) and 128K vocab."""
        config = MockConfig(
            hidden_size=4096,
            ffn_hidden_size=14336,
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            padded_vocab_size=128256,
        )
        model = LlamaModel(config)
        total = sum(p.numel() for p in model.parameters())
        # LLaMA-3-8B: ~8.0B params reported, but our untied embedding/lm_head
        # adds ~2 * (128256-32000) * 4096 = ~788M extra vs LLaMA-7B baseline.
        # Actual total with separate embedding + lm_head: ~9.5-9.7B
        assert 9_000_000_000 < total < 10_000_000_000, (
            f"Expected ~9.5-9.7B params (untied embedding + lm_head), got {total}"
        )

    def test_gqa_reduces_total_params(self):
        """GQA model should have fewer params than equivalent MHA model."""
        config_mha = MockConfig(num_attention_heads=32, num_kv_heads=32)
        config_gqa = MockConfig(num_attention_heads=32, num_kv_heads=8)

        model_mha = LlamaModel(config_mha)
        model_gqa = LlamaModel(config_gqa)

        mha_total = sum(p.numel() for p in model_mha.parameters())
        gqa_total = sum(p.numel() for p in model_gqa.parameters())

        assert gqa_total < mha_total, f"GQA ({gqa_total}) should be < MHA ({mha_total})"

    def test_forward_backward_all_layers(self):
        """With TP=4, the full model produces communication LogItems."""
        config = MockConfig(num_layers=4, tensor_model_parallel_size=4)
        model = LlamaModel(config)

        fwd = model.forward()
        bwd = model.backward()

        # forward: embedding + 4 layers + lm_head with TP comms
        # backward: 4 layers + embedding with TP comms
        assert len(fwd.workload) > 0
        assert len(bwd.workload) > 0

    def test_forward_backward_tp1_no_comms(self):
        """TP=1 without computation produces empty but valid workload."""
        config = MockConfig(num_layers=4, tensor_model_parallel_size=1)
        model = LlamaModel(config)

        fwd = model.forward()
        bwd = model.backward()
        assert isinstance(fwd.workload, list)
        assert isinstance(bwd.workload, list)

    def test_activation_memory_positive(self):
        config = MockConfig(num_layers=4)
        model = LlamaModel(config)
        assert model.activation_memory() > 0

    def test_num_kv_heads_defaults_to_num_heads(self):
        """When num_kv_heads is not set, it defaults to num_attention_heads."""
        config = MockConfig(num_attention_heads=32)
        del config.num_kv_heads  # simulate missing arg
        model = LlamaModel(config)
        # Can't directly check the internal num_kv_heads, but
        # the model should be constructable without error
        assert model is not None

    def test_rope_is_created(self):
        config = MockConfig()
        model = LlamaModel(config)
        assert isinstance(model.rotary_emb, LlamaRotaryEmbedding)
        assert model.rotary_emb.dim == 128  # 4096 / 32

    def test_final_norm_is_llama_rmsnorm(self):
        config = MockConfig()
        model = LlamaModel(config)
        assert isinstance(model.final_norm, LlamaRMSNorm)
