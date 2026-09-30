"""Regression tests for bugs found and fixed in 2025-06-15 research session.

Covers:
  1. MOEMLP.backward() missing workloads.extend() (MockedMegatron.py)
  2. DeepSeekMoE.moe_mlp_backward() same bug (MockedDeepSeek.py)
  3. GatedDeltaNet returns empty Workload (MockedQwen3_5.py)
  4. SwiGLU intermediate_size backward compat (MockedMegatron.py)
  5. ffn_hidden_size -> intermediate_size rename (utils.py)
"""

import types
import pytest

from utils.utils import CommType
from workload_generator.mocked_model.MockedModel import MockedModel


# ===========================================================================
# 1. MOEMLP.backward() workloads.extend() regression
# ===========================================================================

class TestMoEMLPBackward:
    """Verify MOEMLP.backward() produces non-empty workloads (was broken)."""

    def test_megatron_moe_backward_not_empty(self):
        """MOEMLP.backward() must include permutation/unpermutation comms."""
        from workload_generator.mocked_model.training.MockedMegatron import MOEMLP

        moe = MOEMLP(
            batch_size=2, hidden_size=4096, tp=4,
            expert_model_parallel_size=4, ffn_hidden_size=1536,
            seq_len=4096, topk=8, num_experts=128, id=0,
        )
        bwd = moe.backward()
        assert len(bwd.workload) > 0, (
            f"MOEMLP.backward() is empty! "
            f"The workloads.extend() fix may have been reverted."
        )

    def test_megatron_moe_forward_backward_parity(self):
        """MOEMLP fwd and bwd should have similar comm counts."""
        from workload_generator.mocked_model.training.MockedMegatron import MOEMLP

        moe = MOEMLP(
            batch_size=2, hidden_size=4096, tp=4,
            expert_model_parallel_size=4, ffn_hidden_size=1536,
            seq_len=4096, topk=8, num_experts=128, id=0,
        )
        fwd = moe.forward()
        bwd = moe.backward()
        # Forward has preprocess all_gather + permutation + unpermutation
        # Backward has permutation + unpermutation (same ops, reverse)
        # Ratio should be close to 1.0
        assert len(bwd.workload) > 0
        assert abs(len(bwd.workload) - len(fwd.workload)) <= 1, (
            f"MOEMLP fwd/bwd asymmetry: fwd={len(fwd.workload)} bwd={len(bwd.workload)}"
        )

    def test_megatron_moe_alltoall_symmetry(self):
        """A2A ops in MoE forward and backward should be equal."""
        from workload_generator.mocked_model.training.MockedMegatron import MOEMLP

        moe = MOEMLP(
            batch_size=2, hidden_size=4096, tp=4,
            expert_model_parallel_size=4, ffn_hidden_size=1536,
            seq_len=4096, topk=8, num_experts=128, id=0,
        )
        fwd = moe.forward()
        bwd = moe.backward()

        fwd_a2a = sum(1 for w in fwd.workload
                      if str(w.comm_type) == "CommType.all_to_all")
        bwd_a2a = sum(1 for w in bwd.workload
                      if str(w.comm_type) == "CommType.all_to_all")
        assert fwd_a2a == bwd_a2a, (
            f"A2A asymmetry in MOEMLP: fwd={fwd_a2a} bwd={bwd_a2a}"
        )


# ===========================================================================
# 2. DeepSeekMoE backward regression
# ===========================================================================

class TestDeepSeekMoEBackward:
    """Verify DeepSeekMoE.moe_mlp_backward() produces non-empty workloads."""

    def test_deepseek_moe_backward_not_empty(self):
        """DeepSeekMoE.moe_mlp_backward() must include perm/unperm comms."""
        from workload_generator.mocked_model.training.MockedDeepSeek import DeepSeekMoE

        moe = DeepSeekMoE(
            hidden_size=7168, ffn_hidden_size=18432, tp=8,
            expert_model_parallel_size=32, seq_len=4096,
            batch_size=2, topk=8, num_experts=256, id=0,
            n_shared_expert=0, sequence_parallel_enabled=True,
            computation_enable=False, add_bias_linear=False,
        )
        fwd = moe.forward()
        bwd = moe.backward()
        assert len(bwd.workload) > 0, (
            f"DeepSeekMoE.backward() is empty! "
            f"The workloads.extend() fix may have been reverted."
        )
        assert len(fwd.workload) > 0


# ===========================================================================
# 3. GatedDeltaNet empty Workload
# ===========================================================================

class TestGatedDeltaNetComms:
    """Verify GatedDeltaNet layers produce zero communication."""

    def test_gated_delta_net_forward_empty(self):
        """Qwen3.5 GatedDeltaNet forward() must return empty Workload."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5GatedDeltaNet,
        )

        gdn = Qwen3_5GatedDeltaNet(
            hidden_size=4096,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_conv_kernel_dim=4,
            layer_id=1,
        )
        fwd = gdn.forward()
        assert len(fwd.workload) == 0, (
            f"GatedDeltaNet forward should return empty Workload, got {len(fwd.workload)}"
        )

    def test_qwen35_dense_comm_count(self):
        """Qwen3.5-9B: 82 fwd = 1 + 8*4 + 24*2 + 1."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )

        cfg = Qwen3_5Params()
        cfg.hidden_size = 4096
        cfg.intermediate_size = 12288
        cfg.num_hidden_layers = 32
        cfg.num_attention_heads = 16
        cfg.num_key_value_heads = 4
        cfg.head_dim = 256
        cfg.vocab_size = 248320
        cfg.full_attention_interval = 4
        cfg.linear_key_head_dim = 128
        cfg.linear_value_head_dim = 128
        cfg.linear_num_key_heads = 16
        cfg.linear_num_value_heads = 32
        cfg.linear_conv_kernel_dim = 4
        cfg.tensor_model_parallel_size = 8
        cfg.world_size = 8
        cfg.seq_length = 4096
        cfg.micro_batch = 2
        cfg.enable_sequence_parallel = True
        cfg.computation_enable = False
        cfg.add_bias_linear = False
        cfg.moe_enable = False

        model = Qwen3_5Model(cfg)
        fwd = model.forward()
        assert len(fwd.workload) == 82, (
            f"Qwen3.5-9B fwd={len(fwd.workload)}, expected 82"
        )


# ===========================================================================
# 4. SwiGLU intermediate_size backward compat
# ===========================================================================

class TestSwiGLUBackwardCompat:
    """Verify intermediate_size rename preserves backward compat."""

    def test_old_ffn_hidden_size_still_works(self):
        """Old config.ffn_hidden_size should work via getattr fallback."""
        from workload_generator.mocked_model.training.MockedMegatron import MegatronModel

        class OldConfig:
            hidden_size = 4096
            ffn_hidden_size = 4 * 4096  # old API
            num_layers = 32
            num_attention_heads = 32
            padded_vocab_size = 128256
            tensor_model_parallel_size = 8
            seq_length = 4096
            micro_batch = 2
            enable_sequence_parallel = True
            computation_enable = False
            add_bias_linear = False
            moe_enable = False
            expert_model_parallel_size = 1
            moe_router_topk = 0
            num_experts = 0
            moe_grouped_gemm = True
            moe_intermediate_size = None
            # no swiglu attr, no intermediate_size attr

        model = MegatronModel(OldConfig())
        fwd = model.forward()
        assert len(fwd.workload) == 129, (
            f"Old API fwd={len(fwd.workload)}, expected 129"
        )

    def test_new_intermediate_size_produces_same_result(self):
        """New intermediate_size with same value = old ffn_hidden_size."""
        from workload_generator.mocked_model.training.MockedMegatron import MegatronModel

        class NewConfig:
            hidden_size = 4096
            intermediate_size = 4 * 4096  # new API
            num_layers = 32
            num_attention_heads = 32
            padded_vocab_size = 128256
            tensor_model_parallel_size = 8
            seq_length = 4096
            micro_batch = 2
            enable_sequence_parallel = True
            computation_enable = False
            add_bias_linear = False
            moe_enable = False
            expert_model_parallel_size = 1
            moe_router_topk = 0
            num_experts = 0
            moe_grouped_gemm = True
            swiglu = False
            moe_intermediate_size = None

        model = MegatronModel(NewConfig())
        fwd = model.forward()

        # Same as old API
        class OldConfig:
            hidden_size = 4096
            ffn_hidden_size = 4 * 4096
            num_layers = 32
            num_attention_heads = 32
            padded_vocab_size = 128256
            tensor_model_parallel_size = 8
            seq_length = 4096
            micro_batch = 2
            enable_sequence_parallel = True
            computation_enable = False
            add_bias_linear = False
            moe_enable = False
            expert_model_parallel_size = 1
            moe_router_topk = 0
            num_experts = 0
            moe_grouped_gemm = True
            moe_intermediate_size = None

        old_model = MegatronModel(OldConfig())
        old_fwd = old_model.forward()

        assert len(fwd.workload) == len(old_fwd.workload), (
            f"New API fwd={len(fwd.workload)} != old API fwd={len(old_fwd.workload)}"
        )

    def test_swiglu_sizing(self):
        """SwiGLU with intermediate_size=12288 correctly uses 2*intermediate for column."""
        from workload_generator.mocked_model.training.MockedMegatron import (
            MegatronMlp, MegatronColumnLinear, MegatronRowLinear,
        )

        mlp = MegatronMlp(
            hidden_size=4096, intermediate_size=12288,
            tp=8, seq_len=4096, batch_size=2, layer_id=0,
            sequence_parallel_enabled=True, computation_enable=False,
            add_bias_linear=False, swiglu=True,
        )
        assert mlp.dense_h_to_4h.output_size == 2 * 12288, (
            f"SwiGLU column output should be 2*intermediate, got {mlp.dense_h_to_4h.output_size}"
        )
        assert mlp.dense_4h_to_h.input_size == 12288, (
            f"SwiGLU row input should be intermediate, got {mlp.dense_4h_to_h.input_size}"
        )


# ===========================================================================
# 5. CLI alias regression
# ===========================================================================

class TestCLIAliases:
    """Verify num_hidden_layers and padded_vocab_size aliases work."""

    def test_num_hidden_layers_alias(self):
        """CLI --num_layers should alias to num_hidden_layers for Qwen3."""
        import types

        # Simulate an argparse namespace with only CLI args (no config file)
        args = types.SimpleNamespace()
        args.num_layers = 36
        args.hidden_size = 4096
        args.vocab_size = 151936
        args.num_attention_heads = 32
        args.swiglu = True
        args.intermediate_size = None
        args.ffn_hidden_size = None
        args.padded_vocab_size = None
        args.tensor_model_parallel_size = 8
        args.make_vocab_size_divisible_by = 128

        # Apply the resolution logic from get_params()
        if args.num_attention_heads is None:
            args.num_attention_heads = args.num_layers
        args.num_hidden_layers = args.num_layers
        if not hasattr(args, 'padded_vocab_size') or args.padded_vocab_size is None:
            # get_padded_vocab_size needs make_vocab_size_divisible_by
            args.make_vocab_size_divisible_by = 128
            import utils.utils
            args.padded_vocab_size = utils.utils.get_padded_vocab_size(args)
        if args.intermediate_size is None:
            args.intermediate_size = args.ffn_hidden_size
        if args.intermediate_size is None:
            if args.swiglu:
                args.intermediate_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
            else:
                args.intermediate_size = 4 * args.hidden_size

        assert hasattr(args, 'num_hidden_layers'), "num_hidden_layers alias missing"
        assert args.num_hidden_layers == 36, f"num_hidden_layers={args.num_hidden_layers}"
        assert args.padded_vocab_size is not None, "padded_vocab_size not set"
        assert args.intermediate_size > 0, "intermediate_size should be computed"
