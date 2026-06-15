"""
Mocked Qwen3 MoE model for AICB inference workload generation.

Qwen3-235B-A22B architecture:
  - 94 decoder layers with GQA + MoE (128 experts, top-8, no shared experts)
  - RMSNorm pre-normalization
  - SwiGLU FFN for expert computation
  - Expert Parallelism (EP) for MoE routing (all_to_all dispatch/combine)

Reuses MegatronColumnLinear/MegatronRowLinear for TP and MOEMLP for EP.

File: MockedQwen3Moe.py
License: Apache 2.0
"""

from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, MockedParam, MockedParamsBase
from workload_generator.mocked_model.training.MockedMegatron import (
    MegatronColumnLinear,
    MegatronRowLinear,
)
from log_analyzer.log import Workload, LogItem


# ---------------------------------------------------------------------------
# Qwen3MoeRMSNorm -- Root Mean Square Normalization
# ---------------------------------------------------------------------------
class Qwen3MoeRMSNorm(MockedModel):
    def __init__(self, hidden_size, layerid, prefix_name):
        super().__init__()
        self.name = prefix_name + "norm"
        self.layer_id = layerid
        self.weight = MockedParam((hidden_size,), name=f"{self.name}_weight")
        self.hidden_size = hidden_size

    def activation_memory(self):
        return self.hidden_size


# ---------------------------------------------------------------------------
# Qwen3MoeAttention -- Group Query Attention for Qwen3
# ---------------------------------------------------------------------------
class Qwen3MoeAttention(MockedModel):
    def __init__(
        self,
        num_attention_heads,
        num_kv_heads,
        hidden_size,
        tp,
        seq_len,
        batch_size,
        layerid,
        sequence_parallel_enabled=True,
        computation_enable=False,
    ):
        super().__init__()
        self.name = "attention_layer"
        self.layer_id = layerid
        self.head_dim = hidden_size // num_attention_heads
        self.kv_tp = min(num_kv_heads, tp)

        self.q_proj = MegatronColumnLinear(
            hidden_size, num_attention_heads * self.head_dim, tp,
            seq_len, batch_size, layerid, "attention_q",
            sequence_parallel_enabled, computation_enable,
            name="qwen3_q_column",
        )
        self.k_proj = MegatronColumnLinear(
            hidden_size, num_kv_heads * self.head_dim, self.kv_tp,
            seq_len, batch_size, layerid, "attention_k",
            sequence_parallel_enabled, computation_enable,
            name="qwen3_k_column",
        )
        self.v_proj = MegatronColumnLinear(
            hidden_size, num_kv_heads * self.head_dim, self.kv_tp,
            seq_len, batch_size, layerid, "attention_v",
            sequence_parallel_enabled, computation_enable,
            name="qwen3_v_column",
        )
        self.o_proj = MegatronRowLinear(
            num_attention_heads * self.head_dim, hidden_size, tp,
            seq_len, batch_size, layerid, "attention_o",
            sequence_parallel_enabled, computation_enable,
            name="qwen3_o_row",
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.q_proj.forward())
        workloads.extend(self.k_proj.forward())
        workloads.extend(self.v_proj.forward())
        workloads.extend(self.o_proj.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.o_proj.backward())
        workloads.extend(self.v_proj.backward())
        workloads.extend(self.k_proj.backward())
        workloads.extend(self.q_proj.backward())
        return workloads


# ---------------------------------------------------------------------------
# Qwen3MoeRoute -- MoE Router with EP dispatch
# ---------------------------------------------------------------------------
class Qwen3MoeRoute(MockedModel):
    def __init__(self, hidden_size, num_experts, topk, ep_size,
                 seq_len, batch_size, tp, layerid):
        super().__init__()
        self.name = "moe_route"
        self.layer_id = layerid
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self.ep_size = ep_size
        self.tp = tp
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.router_weight = MockedParam(
            (hidden_size, num_experts), name=f"moe_router_w_{layerid}"
        )

    def forward(self):
        workloads = Workload()
        msg_per_token = self.hidden_size * self.topk
        if self.ep_size > 1:
            workloads.append(LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                msg_size=self.seq_len * msg_per_token * self.batch_size * 2,
                stage=f"forward.MoE.route.dispatch.L{self.layer_id}",
            ))
        return workloads

    def backward(self):
        workloads = Workload()
        msg_per_token = self.hidden_size * self.topk
        if self.ep_size > 1:
            workloads.append(LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                msg_size=self.seq_len * msg_per_token * self.batch_size * 2,
                stage=f"backward.MoE.route.combine.L{self.layer_id}",
            ))
        return workloads


# ---------------------------------------------------------------------------
# Qwen3MoeExpert -- SwiGLU FFN per Expert
# ---------------------------------------------------------------------------
class Qwen3MoeExpert(MockedModel):
    def __init__(self, hidden_size, intermediate_size, num_local_experts,
                 tp, seq_len, batch_size, layerid):
        super().__init__()
        self.name = "moe_expert"
        self.layer_id = layerid
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.tp = tp
        self.seq_len = seq_len
        self.batch_size = batch_size

        fc1_out = intermediate_size * num_local_experts
        fc1_per_tp = divide(fc1_out, tp)
        fc2_in = intermediate_size * num_local_experts
        fc2_per_tp = divide(fc2_in, tp)

        # gate + up projections (combined into one weight for simplicity)
        self.gate_up_weight = MockedParam(
            (hidden_size, fc1_per_tp), name=f"expert_gate_up_w_{layerid}"
        )
        # down projection
        self.down_weight = MockedParam(
            (fc2_per_tp, hidden_size), name=f"expert_down_w_{layerid}"
        )

    def forward(self):
        workloads = Workload()
        fc1_per_tp = divide(self.intermediate_size * self.num_local_experts, self.tp)
        # Gate+up combined matmul
        workloads.append(LogItem(
            comm_type=CommType.computation,
            msg_size=(
                (self.seq_len, self.batch_size, self.hidden_size),
                (self.hidden_size, fc1_per_tp),
            ),
            stage=f"forward.MoE.expert.gate_up.L{self.layer_id}",
        ))
        # Down projection matmul
        workloads.append(LogItem(
            comm_type=CommType.computation,
            msg_size=(
                (self.seq_len, self.batch_size, fc1_per_tp),
                (fc1_per_tp, self.hidden_size),
            ),
            stage=f"forward.MoE.expert.down.L{self.layer_id}",
        ))
        return workloads

    def backward(self):
        workloads = Workload()
        fc2_per_tp = divide(self.intermediate_size * self.num_local_experts, self.tp)
        workloads.append(LogItem(
            comm_type=CommType.computation,
            msg_size=(
                (fc2_per_tp, self.seq_len * self.batch_size),
                (self.seq_len * self.batch_size, self.hidden_size),
            ),
            stage=f"backward.MoE.expert.down.L{self.layer_id}",
        ))
        workloads.append(LogItem(
            comm_type=CommType.computation,
            msg_size=(
                (self.hidden_size, self.seq_len * self.batch_size),
                (self.seq_len * self.batch_size, fc2_per_tp),
            ),
            stage=f"backward.MoE.expert.gate_up.L{self.layer_id}",
        ))
        return workloads


# ---------------------------------------------------------------------------
# Qwen3MoeBlock -- Route + Expert Gemm with EP communication
# ---------------------------------------------------------------------------
class Qwen3MoeBlock(MockedModel):
    def __init__(self, hidden_size, intermediate_size, num_experts, topk,
                 ep_size, tp, seq_len, batch_size, layerid):
        super().__init__()
        self.name = "moe_block"
        self.layer_id = layerid
        self.hidden_size = hidden_size
        self.topk = topk
        self.ep_size = ep_size
        self.tp = tp
        self.seq_len = seq_len
        self.batch_size = batch_size

        num_local_experts = num_experts // ep_size

        self.route = Qwen3MoeRoute(
            hidden_size, num_experts, topk, ep_size,
            seq_len, batch_size, tp, layerid,
        )
        self.moeGemm = Qwen3MoeExpert(
            hidden_size, intermediate_size, num_local_experts,
            tp, seq_len, batch_size, layerid,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.route.forward())
        workloads.extend(self.moeGemm.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.moeGemm.backward())
        workloads.extend(self.route.backward())
        return workloads


# ---------------------------------------------------------------------------
# Qwen3MoeTransformerLayer -- Pre-norm GQA + MoE Block
# ---------------------------------------------------------------------------
class Qwen3MoeTransformerLayer(MockedModel):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads,
                 num_kv_heads, num_experts, topk, ep_size, tp,
                 seq_len, batch_size, layerid,
                 sequence_parallel_enabled=True, computation_enable=False):
        super().__init__()
        self.pre_norm = Qwen3MoeRMSNorm(hidden_size, layerid, prefix_name="attention_")
        self.attention = Qwen3MoeAttention(
            num_attention_heads, num_kv_heads, hidden_size, tp,
            seq_len, batch_size, layerid,
            sequence_parallel_enabled, computation_enable,
        )
        self.post_norm = Qwen3MoeRMSNorm(hidden_size, layerid, prefix_name="moe_")
        self.MoE = Qwen3MoeBlock(
            hidden_size, intermediate_size, num_experts, topk,
            ep_size, tp, seq_len, batch_size, layerid,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.attention.forward())
        workloads.extend(self.MoE.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.MoE.backward())
        workloads.extend(self.attention.backward())
        return workloads


# ---------------------------------------------------------------------------
# Qwen3MoeModel -- Full Qwen3 MoE Architecture
# ---------------------------------------------------------------------------
class Qwen3MoeModel(MockedModel):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_kv_heads = getattr(config, "num_kv_heads", config.num_attention_heads)
        self.topk = config.num_experts_per_tok
        self.num_hidden_layers = config.num_hidden_layers
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.ep_size = getattr(config, "expert_model_parallel_size", 1)
        self.tp = getattr(config, "tensor_model_parallel_size", 1)
        self.seq_len = getattr(config, "seq_length", 2048)
        self.batch_size = getattr(config, "micro_batch", 1)
        self.sp_enabled = getattr(config, "enable_sequence_parallel", True)
        self.comp_enable = getattr(config, "computation_enable", False)

        self.layers = [
            Qwen3MoeTransformerLayer(
                self.hidden_size, self.moe_intermediate_size,
                config.num_attention_heads, self.num_kv_heads,
                self.num_experts, self.topk, self.ep_size, self.tp,
                self.seq_len, self.batch_size, i,
                self.sp_enabled, self.comp_enable,
            )
            for i in range(self.num_hidden_layers)
        ]

    def forward(self):
        workloads = Workload()
        for layer in self.layers:
            workloads.extend(layer.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        return workloads


# ---------------------------------------------------------------------------
# Qwen3MoeParams -- Model configuration
# ---------------------------------------------------------------------------
class Qwen3MoeParams(MockedParamsBase):
    def __init__(self, config_file=None, args=None):
        super().__init__("Qwen3-Moe-235B", "Qwen3-Moe", config_file, args)
        if hasattr(self, 'router_expert') or hasattr(self, 'duped_expert'):
            self.num_experts = self.router_expert + self.duped_expert


if __name__ == "__main__":
    import sys
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    args = Qwen3MoeParams(config_file)
    model = Qwen3MoeModel(args)
    workloads = model.forward()
    filename = f"qwen3_moe_workload_{args.seq_length}s_{args.micro_batch}bs.csv"
    workloads.dump(filename)
    print(f"Generated {filename} with {len(workloads.workload)} LogItems")
