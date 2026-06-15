"""
Mocked Qwen3 Next model for AICB inference workload generation.

Qwen3-Next-80B architecture:
  - Hybrid decoder: alternating full-attention and GatedDeltaNet (linear SSM) layers
  - GatedDeltaNet: linear attention variant with gated delta mechanism
  - MoE layers with EP routing (all_to_all dispatch/combine)
  - RMSNorm pre-normalization
  - SwiGLU FFN for expert computation

Reuses MockedQwen3Moe components for attention/norm/MoE.

GatedDeltaNet is approximated as:
  - Similar computation volume to standard attention (projection matmuls)
  - P2P sequential dependency along the sequence dimension (modeled as isend/irecv)
  - NOTE: This is an approximation; GatedDeltaNet's actual cost depends on kernel fusion.

File: MockedQwen3Next.py
License: Apache 2.0
"""

from utils.utils import CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, MockedParam, MockedParamsBase
from workload_generator.mocked_model.inference.MockedQwen3Moe import (
    Qwen3MoeRMSNorm,
    Qwen3MoeAttention,
    Qwen3MoeBlock,
)
from log_analyzer.log import Workload, LogItem


# ---------------------------------------------------------------------------
# Qwen3NextRMSNorm -- Reuses Qwen3MoeRMSNorm
# ---------------------------------------------------------------------------
Qwen3NextRMSNorm = Qwen3MoeRMSNorm


# ---------------------------------------------------------------------------
# Qwen3NextGatedDeltaNet -- Linear Attention with Gated Delta
# ---------------------------------------------------------------------------
class Qwen3NextGatedDeltaNet(MockedModel):
    """GatedDeltaNet: a linear attention variant with:
      - Input projection (hidden -> 2 * hidden)
      - Forget gate + output gate
      - Delta rule update
    Approximated as attention-level computation with P2P sequence dependency.
    """

    def __init__(self, hidden_size, tp, seq_len, batch_size, layerid,
                 computation_enable=False):
        super().__init__()
        self.name = "attention_gdn"
        self.layer_id = layerid
        self.hidden_size = hidden_size
        self.tp = tp
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.comp_enable = computation_enable

        # Input projection: hidden -> 2*hidden (value + gate branches)
        self.in_proj_weight = MockedParam(
            (hidden_size, 2 * hidden_size // tp),
            name=f"gdn_in_proj_w_{layerid}",
        )
        # Output projection: hidden -> hidden
        self.out_proj_weight = MockedParam(
            (hidden_size // tp, hidden_size),
            name=f"gdn_out_proj_w_{layerid}",
        )
        # Delta projection
        self.delta_weight = MockedParam(
            (hidden_size, hidden_size // tp),
            name=f"gdn_delta_w_{layerid}",
        )

    def forward(self):
        workloads = Workload()
        # Input projection computation
        if self.comp_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=(
                    (self.seq_len, self.batch_size, self.hidden_size),
                    (self.hidden_size, 2 * self.hidden_size // self.tp),
                ),
                stage=f"forward.GatedDeltaNet.in_proj.L{self.layer_id}",
            ))
        # Sequential scan along sequence (P2P dependency)
        workloads.append(LogItem(
            comm_type=CommType.isend,
            comm_group=CommGroup.all,
            msg_size=self.hidden_size * self.batch_size,
            stage=f"forward.GatedDeltaNet.scan_send.L{self.layer_id}",
        ))
        # Delta + output projection
        if self.comp_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=(
                    (self.seq_len, self.batch_size, self.hidden_size // self.tp),
                    (self.hidden_size // self.tp, self.hidden_size),
                ),
                stage=f"forward.GatedDeltaNet.out_proj.L{self.layer_id}",
            ))
        return workloads

    def backward(self):
        workloads = Workload()
        if self.comp_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=(
                    (self.hidden_size, self.seq_len * self.batch_size),
                    (self.seq_len * self.batch_size, self.hidden_size // self.tp),
                ),
                stage=f"backward.GatedDeltaNet.out_proj.L{self.layer_id}",
            ))
        workloads.append(LogItem(
            comm_type=CommType.irecv,
            comm_group=CommGroup.all,
            msg_size=self.hidden_size * self.batch_size,
            stage=f"backward.GatedDeltaNet.scan_recv.L{self.layer_id}",
        ))
        if self.comp_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=(
                    (2 * self.hidden_size // self.tp, self.seq_len * self.batch_size),
                    (self.seq_len * self.batch_size, self.hidden_size),
                ),
                stage=f"backward.GatedDeltaNet.in_proj.L{self.layer_id}",
            ))
        return workloads


# ---------------------------------------------------------------------------
# Qwen3NextAttention -- Standard GQA for full-attention layers
# ---------------------------------------------------------------------------
Qwen3NextAttention = Qwen3MoeAttention


# ---------------------------------------------------------------------------
# Qwen3NextRoute -- MoE Router for Qwen3 Next
# ---------------------------------------------------------------------------
Qwen3NextRoute = __import__(
    'workload_generator.mocked_model.inference.MockedQwen3Moe',
    fromlist=['Qwen3MoeRoute']
).Qwen3MoeRoute


# ---------------------------------------------------------------------------
# Qwen3NextExpert -- SwiGLU FFN for Qwen3 Next
# ---------------------------------------------------------------------------
Qwen3NextExpert = __import__(
    'workload_generator.mocked_model.inference.MockedQwen3Moe',
    fromlist=['Qwen3MoeExpert']
).Qwen3MoeExpert


# ---------------------------------------------------------------------------
# Qwen3NextBlock -- Route + Expert Gemm
# ---------------------------------------------------------------------------
Qwen3NextBlock = __import__(
    'workload_generator.mocked_model.inference.MockedQwen3Moe',
    fromlist=['Qwen3MoeBlock']
).Qwen3MoeBlock


# ---------------------------------------------------------------------------
# Qwen3NextTransformerLayer -- Alternating Attention / GatedDeltaNet
# ---------------------------------------------------------------------------
class Qwen3NextTransformerLayer(MockedModel):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads,
                 num_kv_heads, num_experts, topk, ep_size, tp,
                 seq_len, batch_size, layerid, full_attention_flag,
                 sequence_parallel_enabled=True, computation_enable=False):
        super().__init__()
        self.pre_norm = Qwen3NextRMSNorm(hidden_size, layerid, prefix_name="attention_")

        if full_attention_flag == 0:
            self.attention = Qwen3NextAttention(
                num_attention_heads, num_kv_heads, hidden_size, tp,
                seq_len, batch_size, layerid,
                sequence_parallel_enabled, computation_enable,
            )
        else:
            self.attention = Qwen3NextGatedDeltaNet(
                hidden_size, tp, seq_len, batch_size, layerid,
                computation_enable,
            )

        self.post_norm = Qwen3NextRMSNorm(hidden_size, layerid, prefix_name="moe_")
        self.MoE = Qwen3NextBlock(
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
# Qwen3NextModel -- Full Qwen3 Next Architecture
# ---------------------------------------------------------------------------
class Qwen3NextModel(MockedModel):
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
        self.full_attn_interval = getattr(config, "full_attention_interval", 4)

        self.layers = [
            Qwen3NextTransformerLayer(
                self.hidden_size, self.moe_intermediate_size,
                config.num_attention_heads, self.num_kv_heads,
                self.num_experts, self.topk, self.ep_size, self.tp,
                self.seq_len, self.batch_size, i,
                (i + 1) % self.full_attn_interval,
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
# Qwen3NextParams -- Model configuration
# ---------------------------------------------------------------------------
class Qwen3NextParams(MockedParamsBase):
    def __init__(self, config_file=None, args=None):
        super().__init__("Qwen3-Next-80B", "Qwen3-Next", config_file, args)
        if hasattr(self, 'router_expert') or hasattr(self, 'duped_expert'):
            self.num_experts = self.router_expert + self.duped_expert


if __name__ == "__main__":
    import sys
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    args = Qwen3NextParams(config_file)
    model = Qwen3NextModel(args)
    workloads = model.forward()
    filename = f"qwen3_next_workload_{args.seq_length}s_{args.micro_batch}bs.csv"
    workloads.dump(filename)
    print(f"Generated {filename} with {len(workloads.workload)} LogItems")
