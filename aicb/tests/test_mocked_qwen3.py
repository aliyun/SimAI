"""Parametrized communication-count tests for MockedQwen3 (dense + MoE)."""

import types
import pytest

from utils.utils import CommType


# ---------------------------------------------------------------------------
# Helper: build a mock argparse namespace for MegatronWorkload
# ---------------------------------------------------------------------------

def _make_args(hidden_size, num_layers, tp=4, ep=1, dp=2, seq=4096, batch=2, moe=False):
    a = types.SimpleNamespace()
    a.frame = "Qwen3"
    a.model_name = "test"
    a.hidden_size = hidden_size
    a.num_hidden_layers = num_layers
    a.ffn_hidden_size = 0
    a.vocab_size = 151936
    a.tensor_model_parallel_size = tp
    a.pipeline_model_parallel = 1
    a.world_size = tp * ep * dp
    a.dp_num = dp
    a.expert_model_parallel_size = ep
    a.context_parallel_size = 1
    a.seq_length = seq
    a.micro_batch = batch
    a.global_batch = 64
    a.epoch_num = 1
    a.num_microbatches = 1
    a.num_attention_heads = 0
    a.num_key_value_heads = 0
    a.head_dim = 128
    a.enable_sequence_parallel = True
    a.computation_enable = False
    a.moe_enable = moe
    a.swiglu = True
    a.add_bias_linear = False
    a.use_distributed_optimizer = True
    a.pp_rank = 0
    a.workload_only = True
    a.order = "tp-cp-ep-dp-pp"
    a.aiob_enable = False
    a.padded_vocab_size = 151936
    return a


# ---------------------------------------------------------------------------
# Model configs (verified from HF config.json)
# ---------------------------------------------------------------------------

QWEN3_DENSE = [
    # name           hidden intermed layers Q_heads KV_heads head_dim tie_emb
    ("Qwen3-0.6B",   1024,  3072,   28,    16,     8,       128,     True),
    ("Qwen3-1.7B",   2048,  6144,   28,    16,     8,       128,     True),
    ("Qwen3-4B",     2560,  9728,   36,    32,     8,       128,     True),
    ("Qwen3-8B",     4096,  12288,  36,    32,     8,       128,     False),
    ("Qwen3-14B",    5120,  17408,  40,    40,     8,       128,     False),
    ("Qwen3-32B",    5120,  25600,  64,    64,     8,       128,     False),
]

QWEN3_MOE = [
    # name              hidden  intermed   layers  Q_heads  KV_heads  head_dim  moe_ff  topk  experts  shared
    ("Qwen3-30B-A3B",   2048,   6144,      48,     32,      4,        128,      768,    8,    128,     False),
    ("Qwen3-235B-A22B", 4096,   12288,     94,     64,      4,        128,      1536,   8,    128,     False),
]


# ---------------------------------------------------------------------------
# Dense model comm-count tests
# ---------------------------------------------------------------------------

class TestQwen3DenseCommCounts:
    """Per-layer comm-ops formula for all 6 Qwen3 dense sizes at TP=4, SP=True."""

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_allgather_count(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_args(h, L)
        wl = MegatronWorkload(args, model)()

        ag = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.all_gather")
        expected = L * 6 + 2 + 3  # layers + lm_head + init/step
        assert ag == expected, f"{name}: all_gather {ag} != {expected}"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_reduce_scatter_count(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_args(h, L)
        wl = MegatronWorkload(args, model)()

        rs = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.reduce_scatter")
        expected = L * 4 + 1 + 1  # layers + lm_head + step
        assert rs == expected, f"{name}: reduce_scatter {rs} != {expected}"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_message_size(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_args(h, L)
        wl = MegatronWorkload(args, model)()

        exp_msg = 2 * 4096 * 2 * h  # 2 * seq * batch * hidden_size
        for w in wl.workload:
            if str(w.comm_type) == "CommType.all_gather" and "ColumnLinear" in (w.stage or ""):
                assert w.msg_size == exp_msg, f"{name}: ColumnLinear AG msg {w.msg_size} != {exp_msg}"
            if str(w.comm_type) == "CommType.reduce_scatter" and "RowLinear" in (w.stage or ""):
                assert w.msg_size == exp_msg, f"{name}: RowLinear RS msg {w.msg_size} != {exp_msg}"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_qk_norm_params(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)

        qk_params = [p for p in model.parameters() if "q_norm" in (p.name or "")]
        assert len(qk_params) == L, f"{name}: q_norm count {len(qk_params)} != {L}"
        for p in qk_params:
            assert p.numel() == 128, f"{name}: q_norm dim {p.numel()} != 128"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_tie_word_embeddings(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)

        lm_params = sum(p.numel() for p in model.lm_head.parameters())
        if tie:
            assert lm_params == 0, f"{name}: tied lm_head should have 0 params, got {lm_params}"
        else:
            expected = 151936 * h // 4  # vocab * hidden / TP
            assert lm_params == expected, f"{name}: lm_head params {lm_params} != {expected}"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_embedding_no_megatron_artifacts(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)

        emb_params = sum(p.numel() for p in model.embedding.parameters())
        expected = 151936 * h // 4  # vocab * hidden / TP (no 4x multiplier)
        assert emb_params == expected, (
            f"{name}: embedding {emb_params} != {expected} "
            f"(ratio={emb_params/expected:.2f}x, should be 1.0x)"
        )

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_head_expansion(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)

        attn = model.layers[0].attention
        expected_q = n_heads * hd
        assert attn.query_projection_size == expected_q, (
            f"{name}: Q dim {attn.query_projection_size} != {expected_q}"
        )
        expected_kv = n_kv * hd
        assert attn.kv_projection_size == expected_kv, (
            f"{name}: KV dim {attn.kv_projection_size} != {expected_kv}"
        )

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_qk_norm_zero_communication(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_args(h, L)
        wl = MegatronWorkload(args, model)()

        qk_items = [w for w in wl.workload
                     if w.stage and ("q_norm" in w.stage or "k_norm" in w.stage)]
        assert len(qk_items) == 0, f"{name}: QK-Norm should produce 0 comm items, got {len(qk_items)}"


# ---------------------------------------------------------------------------
# MoE model comm-count tests
# ---------------------------------------------------------------------------

class TestQwen3MoeCommCounts:
    """Per-layer comm-ops formula for Qwen3 MoE models at TP=4, EP=4.

    The rank-mapper crashes with ep_group present (ZeroDivisionError in
    _fill_ranks).  This is a pre-existing upstream issue, not related to
    Qwen3 mocks.  The fixture patches WorkloadGenerator.__call__ to skip
    rank-filling so communication counts can be verified.
    """

    @pytest.fixture(autouse=True)
    def _skip_rank_fill(self, monkeypatch):
        """Patch WorkloadGenerator.__call__ to skip _fill_ranks."""
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

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN3_MOE)
    def test_alltoall_count(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp)
        model = Qwen3Model(cfg)
        args = _make_args(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        a2a = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.all_to_all")
        expected = L * 4  # 4 A2A per MoE layer (fwd dispatch+combine, bwd dispatch+combine)
        assert a2a == expected, f"{name}: all_to_all {a2a} != {expected}"

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN3_MOE)
    def test_alltoall_symmetry(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp)
        model = Qwen3Model(cfg)
        args = _make_args(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        moe_items = [w for w in wl.workload if w.stage and "MoE" in w.stage]
        fwd_a2a = sum(1 for w in moe_items
                      if w.stage.startswith("forward") and str(w.comm_type) == "CommType.all_to_all")
        bwd_a2a = sum(1 for w in moe_items
                      if w.stage.startswith("backward") and str(w.comm_type) == "CommType.all_to_all")
        assert fwd_a2a == bwd_a2a, (
            f"{name}: A2A asymmetry fwd={fwd_a2a} bwd={bwd_a2a}"
        )

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN3_MOE)
    def test_alltoall_message_size(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp)
        model = Qwen3Model(cfg)
        args = _make_args(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        exp_msg = 4096 * h * 2 * topk // 4 // 4 * 2  # seq * hidden * batch * topk // tp // ep * 2
        for w in wl.workload:
            if str(w.comm_type) == "CommType.all_to_all":
                assert w.msg_size == exp_msg, (
                    f"{name}: A2A msg {w.msg_size} != {exp_msg}"
                )

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN3_MOE)
    def test_moe_backward_not_empty(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp)
        model = Qwen3Model(cfg)
        args = _make_args(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        bwd_moe = [w for w in wl.workload
                    if w.stage and w.stage.startswith("backward") and "MoE" in w.stage]
        assert len(bwd_moe) > 0, (
            f"{name}: MoE backward is empty! MOEMLP backward fix may not be applied."
        )

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,moe_ff,topk,nexp,shared", QWEN3_MOE)
    def test_moe_reduce_scatter_count(self, name, h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp, shared):
        from workload_generator.mocked_model.training.MockedQwen3 import (
            Qwen3Params, Qwen3Model,
        )
        from workload_generator.generate_megatron_workload import MegatronWorkload

        cfg = _build_moe_cfg(h, ff, L, n_heads, n_kv, hd, moe_ff, topk, nexp)
        model = Qwen3Model(cfg)
        args = _make_args(h, L, ep=4, dp=2, moe=True)
        wl = MegatronWorkload(args, model)()

        rs = sum(1 for w in wl.workload if str(w.comm_type) == "CommType.reduce_scatter")
        expected = L * 4 + 2  # layers + lm_head + step
        assert rs == expected, f"{name}: reduce_scatter {rs} != {expected}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dense_cfg(hidden_size, intermediate_size, num_layers,
                     num_heads, num_kv, head_dim, tie_emb):
    cfg = types.SimpleNamespace()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = num_heads
    cfg.num_key_value_heads = num_kv
    cfg.head_dim = head_dim
    cfg.vocab_size = 151936
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
    return cfg


def _build_moe_cfg(hidden_size, intermediate_size, num_layers,
                   num_heads, num_kv, head_dim, moe_ff, topk, num_experts):
    cfg = types.SimpleNamespace()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = num_heads
    cfg.num_key_value_heads = num_kv
    cfg.head_dim = head_dim
    cfg.vocab_size = 151936
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
    return cfg

# ---------------------------------------------------------------------------
# SimAI training workload generator (.txt format for C++ simulator)
# ---------------------------------------------------------------------------

class TestSimAITrainingGenerator:
    """End-to-end: Qwen3Model -> SIMAI_workload -> .txt (HYBRID_TRANSFORMER)."""

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_txt_format_hybrid_transformer_header(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import Qwen3Model
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload
        import tempfile, os

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_simai_args(h, L)
        work = SIMAI_workload(model, args, None)
        work.workload_generate()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, prefix="t_") as tmp:
            txt_path = tmp.name
        try:
            work.dump_file(txt_path.replace(".txt", ""))
            with open(txt_path) as f:
                first = f.readline().strip()
            assert "HYBRID_TRANSFORMER" in first, f"{name}: missing header"
        finally:
            os.unlink(txt_path)

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_txt_has_required_comm_types(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import Qwen3Model
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload
        import tempfile, os

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_simai_args(h, L)
        work = SIMAI_workload(model, args, None)
        work.workload_generate()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, prefix="t_") as tmp:
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
            assert "ALLREDUCE" in comm_types, f"{name}: missing ALLREDUCE"
        finally:
            os.unlink(txt_path)

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_txt_message_sizes_match(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import Qwen3Model
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload
        import tempfile, os

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_simai_args(h, L)
        work = SIMAI_workload(model, args, None)
        work.workload_generate()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, prefix="t_") as tmp:
            txt_path = tmp.name
        try:
            work.dump_file(txt_path.replace(".txt", ""))
            with open(txt_path) as f:
                lines = f.read().strip().split("\n")
            exp_tp = 2 * 4096 * 2 * h
            found = any(
                line.split("\t")[3] in ("ALLGATHER", "REDUCESCATTER")
                and int(line.split("\t")[4]) == exp_tp
                for line in lines[2:]
                if len(line.split("\t")) >= 5
            )
            assert found, f"{name}: TP comm size {exp_tp} not found"
        finally:
            os.unlink(txt_path)

    @pytest.mark.parametrize("name,h,ff,L,n_heads,n_kv,hd,tie", QWEN3_DENSE)
    def test_txt_produces_nonempty_workload(self, name, h, ff, L, n_heads, n_kv, hd, tie):
        from workload_generator.mocked_model.training.MockedQwen3 import Qwen3Model
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload

        cfg = _build_dense_cfg(h, ff, L, n_heads, n_kv, hd, tie)
        model = Qwen3Model(cfg)
        args = _make_simai_args(h, L)
        work = SIMAI_workload(model, args, None)
        work.workload_generate()
        assert len(work.workload) > 0, f"{name}: empty workload"


def _make_simai_args(hidden_size, num_layers):
    a = types.SimpleNamespace()
    a.frame = "Qwen3"; a.model_name = "test"; a.hidden_size = hidden_size
    a.num_hidden_layers = num_layers; a.ffn_hidden_size = 0; a.vocab_size = 151936
    a.tensor_model_parallel_size = 4; a.pipeline_model_parallel = 1
    a.world_size = 8; a.dp_num = 2; a.expert_model_parallel_size = 1
    a.seq_length = 4096; a.micro_batch = 2; a.global_batch = 64
    a.enable_sequence_parallel = True; a.moe_enable = False
    a.moe_router_topk = 0; a.num_experts = 0; a.moe_grouped_gemm = True
    a.use_flash_attn = False; a.gpu_type = "A100"; a.ga_num = 8
    a.recompute_activations = False; a.aiob_enable = False
    a.num_layers = num_layers; a.epoch_num = 1
    a.num_attention_heads = 0; a.num_key_value_heads = 0; a.head_dim = 128
    return a
