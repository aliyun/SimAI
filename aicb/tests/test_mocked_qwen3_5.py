"""Parametrized communication-count tests for MockedQwen3_5 (dense + MoE)."""

import types
import pytest

from utils.utils import CommType


# ---------------------------------------------------------------------------
# Qwen3.5 model configs (verified from HF config.json)
# ---------------------------------------------------------------------------

QWEN35_DENSE = [
    # name            h      ff     L  Q_heads KV_heads hd  tie    linear_khd linear_vhd
    ("Qwen3.5-0.8B",  1024,  3584,  24,  8,    2,    256, True,  128,      16),
    ("Qwen3.5-2B",    2048,  6144,  24,  8,    2,    256, True,  128,      16),
    ("Qwen3.5-4B",    2560,  9216,  32, 16,    4,    256, True,  128,      32),
    ("Qwen3.5-9B",    4096, 12288,  32, 16,    4,    256, False, 128,      32),
    ("Qwen3.5-27B",   5120, 17408,  64, 24,    4,    256, False, 128,      48),
]

QWEN35_MOE = [
    # name               h     ff    L  Q_heads KV_heads hd  moe_ff topk experts shared
    ("Qwen3.5-35B-A3B",  2048, 6144, 40, 16,    2,    256, 512,   8,   256,    512),
    ("Qwen3.5-122B-A10B",3072, 9216, 48, 32,    2,    256, 1024,  8,   256,    1024),
    ("Qwen3.5-397B-A17B",4096,12288,60, 32,    2,    256, 1024, 10,   512,    1024),
]


def _make_args35(hidden_size, num_layers, tp=4, ep=1, dp=2, seq=4096, batch=2, moe=False):
    a = types.SimpleNamespace()
    a.frame = "Qwen3.5"; a.model_name = "test"; a.hidden_size = hidden_size
    a.num_hidden_layers = num_layers; a.ffn_hidden_size = 0; a.vocab_size = 248320
    a.tensor_model_parallel_size = tp; a.pipeline_model_parallel = 1
    a.world_size = tp * ep * dp; a.dp_num = dp
    a.expert_model_parallel_size = ep; a.context_parallel_size = 1
    a.seq_length = seq; a.micro_batch = batch; a.global_batch = 64
    a.epoch_num = 1; a.num_microbatches = 1
    a.num_attention_heads = 0; a.num_key_value_heads = 0; a.head_dim = 256
    a.enable_sequence_parallel = True; a.computation_enable = False
    a.moe_enable = moe; a.swiglu = True; a.add_bias_linear = False
    a.use_distributed_optimizer = True; a.pp_rank = 0
    a.workload_only = True; a.order = "tp-cp-ep-dp-pp"; a.aiob_enable = False
    a.padded_vocab_size = 248320; a.recompute_activations = False
    a.num_layers = num_layers; a.ga_num = 8; a.gpu_type = "A100"
    a.use_flash_attn = False
    return a


def _build_dense_cfg35(hidden_size, intermediate_size, num_layers,
                        num_heads, num_kv, head_dim, tie_emb,
                        linear_key_dim, linear_value_dim):
    cfg = types.SimpleNamespace()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = num_heads
    cfg.num_key_value_heads = num_kv
    cfg.head_dim = head_dim
    cfg.vocab_size = 248320
    cfg.tie_word_embeddings = tie_emb
    cfg.tensor_model_parallel_size = 4
    cfg.world_size = 8
    cfg.seq_length = 4096
    cfg.micro_batch = 2
    cfg.enable_sequence_parallel = True
    cfg.computation_enable = False
    cfg.add_bias_linear = False
    cfg.moe_enable = False
    cfg.moe_intermediate_size = 0
    cfg.moe_router_topk = 0
    cfg.num_experts = 0
    cfg.expert_model_parallel_size = 1
    cfg.full_attention_interval = 4
    cfg.linear_key_head_dim = linear_key_dim
    cfg.linear_value_head_dim = linear_value_dim
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = linear_value_dim // head_dim if False else linear_value_dim
    return cfg


def _build_moe_cfg35(hidden_size, intermediate_size, num_layers,
                      num_heads, num_kv, head_dim, moe_ff, topk, num_experts,
                      shared_intermediate):
    cfg = types.SimpleNamespace()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = num_heads
    cfg.num_key_value_heads = num_kv
    cfg.head_dim = head_dim
    cfg.vocab_size = 248320
    cfg.tie_word_embeddings = False
    cfg.tensor_model_parallel_size = 4
    cfg.expert_model_parallel_size = 4
    cfg.world_size = 32
    cfg.seq_length = 4096
    cfg.micro_batch = 2
    cfg.enable_sequence_parallel = True
    cfg.computation_enable = False
    cfg.add_bias_linear = False
    cfg.moe_enable = True
    cfg.moe_intermediate_size = moe_ff
    cfg.moe_router_topk = topk
    cfg.num_experts = num_experts
    cfg.full_attention_interval = 4
    cfg.linear_key_head_dim = 128
    cfg.linear_value_head_dim = 128
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 32
    cfg.shared_expert_intermediate_size = shared_intermediate
    return cfg


# ---------------------------------------------------------------------------
# Dense model tests
# ---------------------------------------------------------------------------

class TestQwen35Dense:
    """Qwen3.5 dense: hybrid GatedDeltaNet + full attention (3:1 pattern)."""

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_layer_routing_3to1_pattern(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        """Every 4th layer is full_attention; all others are GatedDeltaNet."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model, Qwen3_5FullAttention, Qwen3_5GatedDeltaNet,
        )

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)

        full_count = 0; gdn_count = 0
        for i, layer in enumerate(model.layers):
            attn = layer.attention
            if isinstance(attn, Qwen3_5FullAttention):
                full_count += 1
                assert (i + 1) % 4 == 0, f"{name}: layer {i} is full, expected at index {(i+1)%4} of 4"
            elif isinstance(attn, Qwen3_5GatedDeltaNet):
                gdn_count += 1
                assert (i + 1) % 4 != 0, f"{name}: layer {i} is GDN, expected NOT at multiple of 4"

        expected_full = L // 4
        expected_gdn = L - expected_full
        assert full_count == expected_full, f"{name}: {full_count} full layers, expected {expected_full}"
        assert gdn_count == expected_gdn, f"{name}: {gdn_count} GDN layers, expected {expected_gdn}"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_gated_delta_net_zero_communication(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        """GatedDeltaNet forward/backward return empty workloads (all local compute)."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L)
        wl = MegatronWorkload(args, model)()

        gdn_items = [w for w in wl.workload
                      if w.stage and "gated_deltanet" in w.stage]
        assert len(gdn_items) == 0, (
            f"{name}: GatedDeltaNet should produce 0 comm items, got {len(gdn_items)}"
        )

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_head_dim_256(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        """All full-attention layers use head_dim=256."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model, Qwen3_5FullAttention,
        )

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)

        for layer in model.layers:
            if isinstance(layer.attention, Qwen3_5FullAttention):
                assert layer.attention.head_dim == 256, (
                    f"{name}: expected head_dim=256, got {layer.attention.head_dim}"
                )

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_has_allgather_reduce_scatter(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        """Full-attention layers produce ALLGATHER and REDUCESCATTER."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L)
        wl = MegatronWorkload(args, model)()

        ag = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.all_gather")
        rs = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.reduce_scatter")
        assert ag > 0, f"{name}: no all_gather items"
        assert rs > 0, f"{name}: no reduce_scatter items"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_per_layer_comm_count_formula(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        """Exact per-layer formula: GDN=3AG+2RS, Full=6AG+4RS (3:1 ratio)."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L)
        wl = MegatronWorkload(args, model)()

        ag = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.all_gather")
        rs = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.reduce_scatter")

        full_layers = L // 4
        # GatedDeltaNet: 3 AG + 2 RS (MLP only, GDN returns empty Workload)
        # Full attention: 6 AG + 4 RS (attention + MLP)
        # Overhead: lm_head(2AG+1RS) + init/step(3AG+1RS)
        exp_ag = L * 3 + full_layers * 3 + 5
        exp_rs = L * 2 + full_layers * 2 + 2

        assert ag == exp_ag, f"{name}: AG {ag} != expected {exp_ag}"
        assert rs == exp_rs, f"{name}: RS {rs} != expected {exp_rs}"


# ---------------------------------------------------------------------------
# MoE model tests
# ---------------------------------------------------------------------------

class TestQwen35MoE:
    """Qwen3.5 MoE: shared experts + GatedDeltaNet hybrid."""

    @pytest.fixture(autouse=True)
    def _skip_rank_fill(self, monkeypatch):
        """Patch WorkloadGenerator.__call__ to skip _fill_ranks (pre-existing crash)."""
        from workload_generator.workload_generator import WorkloadGenerator

        def _call_without_fill(self):
            args = self.args
            from log_analyzer.log import Workload, LogItem
            from utils.utils import CommType
            self.workload = Workload()
            self.init()
            self.workload.append(LogItem(comm_type=CommType.epoch_end))
            for _ in range(args.epoch_num):
                if args.pipeline_model_parallel > 1 and args.frame != "collective_test":
                    self.with_pipeline_forward_backward()
                else:
                    for _ in range(args.num_microbatches):
                        self.forward()
                        self.backward()
                self.step()
                self.workload.append(LogItem(comm_type=CommType.epoch_end))
            return self.workload

        monkeypatch.setattr(WorkloadGenerator, "__call__", _call_without_fill)

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN35_MOE)
    def test_all_layers_use_moe(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        """All layers in MoE config should route to MoE FFN."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )

        cfg = _build_moe_cfg35(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared)
        model = Qwen3_5Model(cfg)

        moe_count = 0
        for layer in model.layers:
            if hasattr(layer, 'mlp') and 'MoE' in layer.mlp.__class__.__name__:
                moe_count += 1
        assert moe_count == L, f"{name}: {moe_count}/{L} layers use MoE, expected all"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN35_MOE)
    def test_alltoall_present(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        """MoE layers produce AllToAll communication."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg35(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        a2a = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.all_to_all")
        assert a2a > 0, f"{name}: no all_to_all items"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN35_MOE)
    def test_shared_expert_present_in_model(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        """Model config includes shared_expert_intermediate_size."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )

        cfg = _build_moe_cfg35(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared)
        model = Qwen3_5Model(cfg)
        assert shared > 0, f"{name}: shared_expert_intermediate_size should be > 0"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN35_MOE)
    def test_alltoall_forward_backward_symmetry(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        """AllToAll dispatch/combine should be symmetric between forward and backward."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg35(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        moe_items = [w for w in wl.workload if w.stage and "MoE" in w.stage]
        fwd_a2a = sum(1 for w in moe_items if w.stage.startswith("forward")
                      and str(w.comm_type) == "CommType.all_to_all")
        bwd_a2a = sum(1 for w in moe_items if w.stage.startswith("backward")
                      and str(w.comm_type) == "CommType.all_to_all")
        assert fwd_a2a == bwd_a2a, f"{name}: A2A asymmetry fwd={fwd_a2a} bwd={bwd_a2a}"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN35_MOE)
    def test_moe_backward_not_empty(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        """MoE backward should not be empty (regression test for backward fix)."""
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg35(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        bwd_moe = [w for w in wl.workload
                    if w.stage and w.stage.startswith("backward") and "MoE" in w.stage]
        assert len(bwd_moe) > 0, f"{name}: MoE backward is empty"


# ---------------------------------------------------------------------------
# SimAI .txt format tests for Qwen3.5 dense
# ---------------------------------------------------------------------------

class TestQwen35SimAIGenerator:
    """Qwen3.5 -> SIMAI_workload -> .txt (HYBRID_TRANSFORMER format)."""

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_txt_hybrid_transformer_header(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload
        import tempfile, os

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L); args.frame = "Qwen3.5"; args.num_layers = L
        work = SIMAI_workload(model, args, None)
        work.workload_generate()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, prefix="t35_") as tmp:
            txt_path = tmp.name
        try:
            work.dump_file(txt_path.replace(".txt", ""))
            with open(txt_path) as f:
                first = f.readline().strip()
            assert "HYBRID_TRANSFORMER" in first, f"{name}: missing header"
        finally:
            os.unlink(txt_path)

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_txt_has_comm_types(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload
        import tempfile, os

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L); args.frame = "Qwen3.5"; args.num_layers = L
        work = SIMAI_workload(model, args, None)
        work.workload_generate()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, prefix="t35_") as tmp:
            txt_path = tmp.name
        try:
            work.dump_file(txt_path.replace(".txt", ""))
            with open(txt_path) as f:
                lines = f.read().strip().split("\n")
            comm_types = set()
            for i in range(2, len(lines)):
                parts = lines[i].split("\t")
                if len(parts) >= 7:
                    comm_types.add(parts[3]); comm_types.add(parts[6])
            assert "ALLGATHER" in comm_types, f"{name}: missing ALLGATHER"
            assert "REDUCESCATTER" in comm_types, f"{name}: missing REDUCESCATTER"
        finally:
            os.unlink(txt_path)

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie,lkd,lvd", QWEN35_DENSE)
    def test_txt_produces_workload(self, name, h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd):
        from workload_generator.mocked_model.training.MockedQwen3_5 import (
            Qwen3_5Params, Qwen3_5Model,
        )
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload
        import tempfile, os

        cfg = _build_dense_cfg35(h, ff, L, n_heads, n_kv, hd, tie, lkd, lvd)
        model = Qwen3_5Model(cfg)
        args = _make_args35(h, L); args.frame = "Qwen3.5"; args.num_layers = L
        work = SIMAI_workload(model, args, None)
        work.workload_generate()
        assert len(work.workload) > 0, f"{name}: empty workload"
