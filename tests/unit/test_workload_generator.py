"""Unit tests for server.simulation.workload_generator."""

import pytest

from server.simulation.workload_generator import (
    MODEL_CONFIGS,
    COMPUTE_TIMES,
    generate_megatron_workload,
    generate_custom_workload,
    generate_workload_content,
    parse_workload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_lines(content: str):
    """Split content into header, count, and layer lines."""
    lines = content.split("\n")
    return lines[0], int(lines[1]), lines[2:]


def _layer_names(layers):
    return [l["name"] for l in layers]


# ---------------------------------------------------------------------------
# Layer count correctness (2N + 12)
# ---------------------------------------------------------------------------


class TestLayerCounts:
    """Each model size must produce exactly 2*num_layers + 12 items."""

    @pytest.mark.parametrize(
        "model_size,expected_layers",
        [
            ("7B", 76),   # 2*32 + 12
            ("13B", 92),  # 2*40 + 12
            ("70B", 172), # 2*80 + 12
            ("175B", 204),# 2*96 + 12
        ],
    )
    def test_model_produces_correct_layer_count(self, model_size, expected_layers):
        content, layers = generate_megatron_workload(model_size=model_size, tp_size=8)
        _, count, data_lines = _parse_lines(content)
        assert len(layers) == expected_layers
        assert count == expected_layers
        assert len(data_lines) == expected_layers

    def test_layer_count_formula_2n_plus_12(self):
        """Verify the 2N+12 formula for all models."""
        for model_size, config in MODEL_CONFIGS.items():
            _, layers = generate_megatron_workload(model_size=model_size)
            expected = 2 * config["num_layers"] + 12
            assert len(layers) == expected, f"{model_size}: got {len(layers)}, expected {expected}"


# ---------------------------------------------------------------------------
# Header correctness
# ---------------------------------------------------------------------------


class TestHeader:
    """Verify the workload header contains correct parallelism parameters."""

    def test_header_contains_tp_size(self):
        content, _ = generate_megatron_workload(tp_size=4, model_size="7B")
        header = content.split("\n")[0]
        assert "model_parallel_NPU_group: 4" in header

    def test_header_contains_all_gpus(self):
        content, _ = generate_megatron_workload(all_gpus=64, model_size="13B")
        header = content.split("\n")[0]
        assert "all_gpus: 64" in header

    def test_header_contains_pp_size(self):
        content, _ = generate_megatron_workload(pp_size=2, model_size="7B")
        header = content.split("\n")[0]
        assert "pp: 2" in header

    def test_header_contains_ep_size(self):
        content, _ = generate_megatron_workload(ep_size=4, model_size="7B")
        header = content.split("\n")[0]
        assert "ep: 4" in header

    def test_line2_matches_actual_layer_count(self):
        content, layers = generate_megatron_workload(model_size="70B")
        _, count, data_lines = _parse_lines(content)
        assert count == len(layers)
        assert count == len(data_lines)


# ---------------------------------------------------------------------------
# Layer structure
# ---------------------------------------------------------------------------


class TestLayerStructure:
    """Verify the correct layer types appear in the right order."""

    def test_prefix_layers_present(self):
        _, layers = generate_megatron_workload(model_size="7B")
        names = _layer_names(layers[:4])
        assert names == ["norm", "grad_norm", "layernorm", "embedding_layer"]

    def test_suffix_layers_present(self):
        _, layers = generate_megatron_workload(model_size="7B")
        names = _layer_names(layers[-8:])
        assert names == [
            "embedding_norm",
            "cross_entropy1", "cross_entropy2", "cross_entropy3",
            "optimizer1", "optimizer2", "optimizer3", "optimizer4",
        ]

    def test_transformer_layers_alternate_attention_mlp(self):
        _, layers = generate_megatron_workload(model_size="13B")
        transformer = layers[4:-8]  # skip prefix and suffix
        assert len(transformer) == 80  # 2*40
        for i in range(0, len(transformer), 2):
            assert transformer[i]["name"] == "attention_layer"
            assert transformer[i + 1]["name"] == "mlp_layer"

    def test_norm_and_grad_norm_prefix_layers(self):
        _, layers = generate_megatron_workload(model_size="7B")
        assert layers[0]["fwd_comm_type"] == "BROADCAST"
        assert layers[1]["fwd_comm_type"] == "ALLGATHER"
        assert layers[1]["ig_comm_type"] == "REDUCESCATTER"
        assert layers[2]["wg_comm_type"] == "ALLREDUCE"


# ---------------------------------------------------------------------------
# Compute times
# ---------------------------------------------------------------------------


class TestComputeTimes:
    """Verify layers have model-appropriate compute times."""

    @pytest.mark.parametrize("model_size", ["7B", "13B", "70B", "175B"])
    def test_attention_compute_matches_reference(self, model_size):
        _, layers = generate_megatron_workload(model_size=model_size)
        attn_layer = layers[4]  # first attention
        assert attn_layer["name"] == "attention_layer"
        assert attn_layer["fwd_compute"] == COMPUTE_TIMES[model_size]["attn_fwd"]

    @pytest.mark.parametrize("model_size", ["7B", "13B", "70B", "175B"])
    def test_mlp_compute_matches_reference(self, model_size):
        _, layers = generate_megatron_workload(model_size=model_size)
        mlp_layer = layers[5]  # first mlp
        assert mlp_layer["name"] == "mlp_layer"
        assert mlp_layer["fwd_compute"] == COMPUTE_TIMES[model_size]["mlp_fwd"]


# ---------------------------------------------------------------------------
# Communication sizes
# ---------------------------------------------------------------------------


class TestCommSizes:
    """Verify communication sizes are computed correctly."""

    def test_tp_comm_size_formula(self):
        """tp_comm = 2 * micro_batch * seq_length * hidden_size."""
        # 13B: seq=2048, hidden=5120 → 2*1*2048*5120 = 20971520
        _, layers = generate_megatron_workload(model_size="13B", tp_size=8)
        attn = layers[4]
        assert attn["fwd_comm_size"] == 2 * 1 * 2048 * 5120

    def test_embedding_norm_comm_size(self):
        """embedding_norm = hidden * vocab * 2."""
        # 13B: hidden=5120, vocab=32000 → 5120*32000*2 = 327680000
        _, layers = generate_megatron_workload(model_size="13B")
        emb_norm = [l for l in layers if l["name"] == "embedding_norm"][0]
        assert emb_norm["fwd_comm_size"] == 5120 * 32000 * 2

    def test_comm_sizes_are_positive(self):
        """All transformer layer comm sizes must be positive."""
        for model_size in MODEL_CONFIGS:
            _, layers = generate_megatron_workload(model_size=model_size)
            transformer = layers[4:-8]
            for layer in transformer:
                assert layer["fwd_comm_size"] > 0, f"{model_size}/{layer['name']}"

    def test_optimizer_comm_size_is_trivial(self):
        _, layers = generate_megatron_workload(model_size="7B")
        optimizers = [l for l in layers if l["name"].startswith("optimizer")]
        for opt in optimizers:
            assert opt["fwd_comm_size"] == 4


# ---------------------------------------------------------------------------
# Parseability
# ---------------------------------------------------------------------------


class TestParseability:
    """Generated output must be parseable by parse_workload()."""

    def test_output_parseable_by_parse_workload(self, tmp_path):
        content, layers = generate_megatron_workload(model_size="13B", tp_size=8)
        path = tmp_path / "workload.txt"
        path.write_text(content)
        parsed = parse_workload(str(path))
        assert parsed["num_layers"] == len(layers)
        assert parsed["model_parallel_NPU_group"] == 8
        assert len(parsed["layers"]) == len(layers)


# ---------------------------------------------------------------------------
# Custom workload
# ---------------------------------------------------------------------------


class TestCustomWorkload:
    """Test custom workload generation preserves user layers."""

    def test_custom_workload_preserves_user_layers(self):
        custom_layers = [
            {"name": "custom_op1", "fwd_compute": 1000, "fwd_comm_type": "ALLREDUCE",
             "fwd_comm_size": 4096},
            {"name": "custom_op2", "fwd_compute": 2000, "fwd_comm_type": "NONE",
             "fwd_comm_size": 0},
        ]
        content = generate_custom_workload(
            tp_size=4, dp_size=2, pp_size=1, ep_size=1, all_gpus=8,
            layers_config=custom_layers,
        )
        lines = content.split("\n")
        assert int(lines[1]) == 2
        assert "custom_op1" in lines[2]
        assert "custom_op2" in lines[3]

    def test_custom_workload_header_format(self):
        content = generate_custom_workload(
            tp_size=4, dp_size=2, pp_size=1, ep_size=1, all_gpus=16,
            layers_config=[{"name": "test"}],
        )
        header = content.split("\n")[0]
        assert "model_parallel_NPU_group: 4" in header
        assert "all_gpus: 16" in header
