"""
Workload Generator — delegates to AICB's SIMAI_workload for full fidelity.
"""

import math
import os
import sys
import tempfile
from typing import Dict, List, Tuple

# Add aicb to path so we can import SIMAI_workload directly
_AICB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "aicb")
if _AICB_DIR not in sys.path:
    sys.path.insert(0, _AICB_DIR)

from workload_generator.SimAI_training_workload_generator import SIMAI_workload
import workload_generator.SimAI_training_workload_generator as _aicb_module
from workload_generator.mocked_model.training.MockedMegatron import MegatronModel
from workload_generator.mocked_model.training.MockedDeepSeek import DeepSeekV3Model


MODEL_CONFIGS = {
    # GPT family (AICB benchmark suite v1.1)
    "GPT-7B": {
        "family": "GPT",
        "hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32,
        "ffn_hidden_size": 16384, "vocab_size": 50257, "seq_length": 2048,
        "recommended": {"tp": 1, "pp": 1, "ep": 1, "gpus": 8},
    },
    "GPT-13B": {
        "family": "GPT",
        "hidden_size": 5120, "num_layers": 40, "num_attention_heads": 40,
        "ffn_hidden_size": 20480, "vocab_size": 50257, "seq_length": 2048,
        "recommended": {"tp": 2, "pp": 1, "ep": 1, "gpus": 16},
    },
    "GPT-22B": {
        "family": "GPT",
        "hidden_size": 6144, "num_layers": 48, "num_attention_heads": 64,
        "ffn_hidden_size": 24576, "vocab_size": 50257, "seq_length": 2048,
        "recommended": {"tp": 4, "pp": 1, "ep": 1, "gpus": 32},
    },
    "GPT-175B": {
        "family": "GPT",
        "hidden_size": 12288, "num_layers": 96, "num_attention_heads": 96,
        "ffn_hidden_size": 49152, "vocab_size": 50257, "seq_length": 2048,
        "recommended": {"tp": 8, "pp": 8, "ep": 1, "gpus": 128},
    },
    # LLaMA family (AICB benchmark suite v1.1)
    "LLaMA-7B": {
        "family": "LLaMA",
        "hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32,
        "ffn_hidden_size": 11008, "vocab_size": 32000, "seq_length": 2048,
        "recommended": {"tp": 1, "pp": 1, "ep": 1, "gpus": 8},
    },
    "LLaMA-13B": {
        "family": "LLaMA",
        "hidden_size": 5120, "num_layers": 40, "num_attention_heads": 40,
        "ffn_hidden_size": 13824, "vocab_size": 32000, "seq_length": 2048,
        "recommended": {"tp": 2, "pp": 1, "ep": 1, "gpus": 16},
    },
    "LLaMA-65B": {
        "family": "LLaMA",
        "hidden_size": 8192, "num_layers": 80, "num_attention_heads": 64,
        "ffn_hidden_size": 28672, "vocab_size": 32000, "seq_length": 4096,
        "recommended": {"tp": 8, "pp": 2, "ep": 1, "gpus": 64},
    },
    "Llama3-405B": {
        "family": "LLaMA",
        "hidden_size": 16384, "num_layers": 128, "num_attention_heads": 128,
        "ffn_hidden_size": 53248, "vocab_size": 32000, "seq_length": 8192,
        "recommended": {"tp": 8, "pp": 16, "ep": 1, "gpus": 512},
    },
    # Mixtral 8x7B MoE (AICB: TP=2, EP=8)
    "Mixtral-8x7B": {
        "family": "Mixtral",
        "hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32,
        "ffn_hidden_size": 14336, "vocab_size": 32000, "seq_length": 4096,
        "num_experts": 8, "n_shared_expert": 0, "n_dense_layers": 0,
        "moe_router_topk": 2,
        "recommended": {"tp": 2, "pp": 1, "ep": 8, "gpus": 32},
    },
    # DeepSeek (MoE)
    "DeepSeek-16B": {
        "family": "DeepSeek",
        "hidden_size": 2048, "num_layers": 27, "num_attention_heads": 16,
        "ffn_hidden_size": 1408, "vocab_size": 129280, "seq_length": 4096,
        "num_experts": 64, "n_shared_expert": 2, "n_dense_layers": 1,
        "moe_router_topk": 6,
        "q_lora_rank": 0, "kv_lora_rank": 512,
        "qk_nope_dim": 128, "qk_rope_dim": 64, "v_head_dim": 128,
        "recommended": {"tp": 4, "pp": 1, "ep": 4, "gpus": 32},
    },
    "DeepSeek-236B": {
        "family": "DeepSeek",
        "hidden_size": 7168, "num_layers": 61, "num_attention_heads": 128,
        "ffn_hidden_size": 2048, "vocab_size": 129280, "seq_length": 4096,
        "num_experts": 256, "n_shared_expert": 1, "n_dense_layers": 3,
        "moe_router_topk": 8,
        "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_dim": 128, "qk_rope_dim": 64, "v_head_dim": 128,
        "recommended": {"tp": 8, "pp": 4, "ep": 8, "gpus": 512},
    },
    "DeepSeek-671B": {
        "family": "DeepSeek",
        "hidden_size": 18432, "num_layers": 61, "num_attention_heads": 128,
        "ffn_hidden_size": 2048, "vocab_size": 129280, "seq_length": 4096,
        "num_experts": 256, "n_shared_expert": 1, "n_dense_layers": 3,
        "moe_router_topk": 8,
        "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_dim": 128, "qk_rope_dim": 64, "v_head_dim": 128,
        "recommended": {"tp": 8, "pp": 8, "ep": 16, "gpus": 2048},
    },
}

_LEGACY_MODEL_MAP = {"7B": "LLaMA-7B", "13B": "LLaMA-13B", "70B": "LLaMA-65B", "175B": "GPT-175B", "405B": "Llama3-405B"}


def _make_divisible(n: int, d: int) -> int:
    return math.ceil(n / d) * d


class _Args:
    """Minimal args object matching what SIMAI_workload expects."""
    def __init__(
        self,
        cfg: dict,
        tp_size: int,
        dp_size: int,
        pp_size: int,
        ep_size: int,
        all_gpus: int,
        micro_batch: int = 1,
        ga: int = 0,
        vpp: int = 1,
        aiob_enable: bool = False,
        comp_filepath: str = "",
    ):
        family = cfg.get("family", "GPT")
        hidden = cfg["hidden_size"]
        num_layers = cfg["num_layers"]
        vocab = cfg["vocab_size"]
        seq = cfg["seq_length"]
        ffn = cfg.get("ffn_hidden_size", hidden * 4)
        num_experts = cfg.get("num_experts", 0)
        is_moe = num_experts > 0

        self.tensor_model_parallel_size = tp_size
        self.pipeline_model_parallel = pp_size
        self.expert_model_parallel_size = ep_size if is_moe else 1
        self.dp_num = max(1, all_gpus // (tp_size * pp_size * max(ep_size, 1)))
        self.world_size = all_gpus

        self.hidden_size = hidden
        self.num_layers = num_layers
        self.num_attention_heads = cfg["num_attention_heads"]
        self.ffn_hidden_size = ffn
        self.vocab_size = vocab
        self.padded_vocab_size = _make_divisible(vocab, tp_size)
        self.seq_length = seq
        self.micro_batch = micro_batch
        self.vpp = vpp
        self.global_batch = micro_batch * self.dp_num * ga if ga > 0 else micro_batch * self.dp_num

        self.enable_sequence_parallel = True
        self.add_bias_linear = False
        self.computation_enable = aiob_enable
        self.aiob_enable = aiob_enable
        self.comp_filepath = comp_filepath
        self.recompute_activations = False

        self.moe_enable = is_moe
        self.num_experts = num_experts
        self.moe_router_topk = cfg.get("moe_router_topk", 1)
        self.n_shared_expert = cfg.get("n_shared_expert", 0)
        self.n_dense_layers = cfg.get("n_dense_layers", 0)
        self.moe_grouped_gemm = True

        # DeepSeek-specific MLA params
        self.q_lora_rank = cfg.get("q_lora_rank", 0)
        self.kv_lora_rank = cfg.get("kv_lora_rank", 512)
        self.qk_nope_dim = cfg.get("qk_nope_dim", 128)
        self.qk_rope_dim = cfg.get("qk_rope_dim", 64)
        self.v_head_dim = cfg.get("v_head_dim", 128)

        self.frame = family if family == "DeepSeek" else "Megatron"


def _build_model(cfg: dict, args: _Args):
    family = cfg.get("family", "GPT")
    if family == "DeepSeek":
        return DeepSeekV3Model(args)
    return MegatronModel(args)


def generate_workload_content(
    model_type: str,
    tp_size: int,
    dp_size: int,
    pp_size: int,
    ep_size: int,
    vpp: int,
    ga: int,
    all_gpus: int,
    num_layers: int,
    hidden_size: int,
    seq_length: int,
    micro_batch_size: int,
    vocab_size: int,
    num_attention_heads: int,
    ffn_hidden_size: int = None,
    model_size: str = "LLaMA-7B",
    num_experts: int = 0,
    n_dense_layers: int = 0,
    moe_router_topk: int = 1,
    aiob_enable: bool = False,
    comp_filepath: str = "",
) -> Tuple[str, List[str]]:
    """Generate AICB-compatible workload by delegating to SIMAI_workload."""
    resolved = _LEGACY_MODEL_MAP.get(model_size, model_size)
    cfg = dict(MODEL_CONFIGS.get(resolved, MODEL_CONFIGS["LLaMA-7B"]))

    # Allow caller overrides
    cfg["num_layers"] = num_layers
    cfg["hidden_size"] = hidden_size
    cfg["seq_length"] = seq_length
    cfg["vocab_size"] = vocab_size
    cfg["num_attention_heads"] = num_attention_heads
    if ffn_hidden_size is not None:
        cfg["ffn_hidden_size"] = ffn_hidden_size
    if num_experts > 0:
        cfg["num_experts"] = num_experts
    if n_dense_layers > 0:
        cfg["n_dense_layers"] = n_dense_layers
    if moe_router_topk > 1:
        cfg["moe_router_topk"] = moe_router_topk

    args = _Args(cfg, tp_size, dp_size, pp_size, ep_size, all_gpus, micro_batch_size, ga,
                 aiob_enable=aiob_enable, comp_filepath=comp_filepath, vpp=vpp)

    # Load AIOB compute cache when enabled
    compute_cache = None
    if aiob_enable and comp_filepath:
        from utils.utils import extract_averages
        if os.path.exists(comp_filepath):
            compute_cache = extract_averages(comp_filepath, args)
        else:
            print(f"[warn] AIOB comp_filepath not found: {comp_filepath}")

    # Inject args into AICB module namespace (workload_generate uses bare `args`)
    _aicb_module.args = args

    model = _build_model(cfg, args)
    wl = SIMAI_workload(model, args, compute_cache=compute_cache)
    if compute_cache is not None:
        wl.workload_generate_aiob()
    else:
        wl.workload_generate()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wl.dump_file(tmp_path[:-4])  # dump_file appends .txt
        with open(tmp_path) as f:
            content = f.read().rstrip("\n")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    lines = content.split("\n")
    layer_lines = lines[2:] if len(lines) > 2 else []
    return content, layer_lines


def generate_megatron_workload(
    tp_size: int = 8,
    dp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
    all_gpus: int = 8,
    model_size: str = "LLaMA-7B",
    ga: int = 8,
    vpp: int = 1,
    aiob_enable: bool = False,
    comp_filepath: str = "",
) -> Tuple[str, List[str]]:
    resolved = _LEGACY_MODEL_MAP.get(model_size, model_size)
    cfg = MODEL_CONFIGS.get(resolved, MODEL_CONFIGS["LLaMA-7B"])
    return generate_workload_content(
        model_type="megatron",
        model_size=resolved,
        tp_size=tp_size,
        dp_size=dp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        all_gpus=all_gpus,
        vpp=vpp,
        ga=ga,
        aiob_enable=aiob_enable,
        comp_filepath=comp_filepath,
        num_layers=cfg["num_layers"],
        hidden_size=cfg["hidden_size"],
        seq_length=cfg["seq_length"],
        micro_batch_size=1,
        vocab_size=cfg["vocab_size"],
        num_attention_heads=cfg["num_attention_heads"],
        ffn_hidden_size=cfg.get("ffn_hidden_size"),
        num_experts=cfg.get("num_experts", 0),
        n_dense_layers=cfg.get("n_dense_layers", 0),
        moe_router_topk=cfg.get("moe_router_topk", 1),
    )


def generate_custom_workload(
    tp_size: int,
    dp_size: int,
    pp_size: int,
    ep_size: int,
    all_gpus: int,
    layers_config: List[Dict],
) -> str:
    ga = max(1, all_gpus // (tp_size * pp_size * max(ep_size, 1) * dp_size))
    header = (
        f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {tp_size} "
        f"ep: {ep_size} pp: {pp_size} vpp: 1 ga: {ga} all_gpus: {all_gpus} "
        f"checkpoints: 0 checkpoint_initiates: 0 pp_comm: 0"
    )
    lines = [header, str(len(layers_config))]
    for layer in layers_config:
        name = layer.get("name", "layer")
        fc = layer.get("fwd_compute", 0)
        fct = layer.get("fwd_comm_type", "NONE")
        fcs = layer.get("fwd_comm_size", 0)
        wc = layer.get("wg_compute", 0)
        wct = layer.get("wg_comm_type", "NONE")
        wcs = layer.get("wg_comm_size", 0)
        ic = layer.get("ig_compute", 0)
        ict = layer.get("ig_comm_type", "NONE")
        ics = layer.get("ig_comm_size", 0)
        rpt = layer.get("repeat", 100)
        lines.append(f"{name}\t-1\t{fc}\t{fct}\t{fcs}\t{wc}\t{wct}\t{wcs}\t{ic}\t{ict}\t{ics}\t{rpt}")
    return "\n".join(lines)


def parse_workload(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return {}

    header = lines[0].strip()
    config: Dict = {}
    parts = header.split()
    for i, part in enumerate(parts):
        if part.endswith(":") and i + 1 < len(parts):
            key = part[:-1]
            value = parts[i + 1]
            try:
                config[key] = int(value)
            except ValueError:
                config[key] = value

    num_layers = int(lines[1].strip()) if len(lines) > 1 else 0
    parsed_layers = []
    for i in range(2, min(2 + num_layers, len(lines))):
        layer_parts = lines[i].split()
        if layer_parts:
            parsed_layers.append({"name": layer_parts[0], "parts": layer_parts[1:]})

    config["layers"] = parsed_layers
    config["num_layers"] = num_layers
    return config
