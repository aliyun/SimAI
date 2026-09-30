"""
Copyright (c) 2024, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Mocked model for Qwen3.5 dense + MoE (training workloads).

Architecture:
  - 3:1 hybrid attention: 75% GatedDeltaNet (linear O(L) + causal conv1d),
    25% full GQA attention (head_dim=256, partial_rotary=0.25, MRoPE, attn_output_gate).
  - MoE: 256-512 experts, top-8 or top-10 routing, WITH shared experts
    (routed experts + 1 always-active shared expert).
  - Pre-LN RMSNorm, SwiGLU MLP. Full-attention layers use QK-Norm (per-head
    RMSNorm on Q/K). GatedDeltaNet layers use RMSNormGated. head_dim=256,
    rope_theta=10M. QK-Norm = pure local compute, no communication impact.
  - Config keys: hidden_size, intermediate_size, full_attention_interval,
    layer_types, linear_key_head_dim, linear_num_key_heads, etc.

Key simulation notes:
  - GatedDeltaNet layers: ZERO communication on attention (all local
    state-matrix recurrence + causal conv1d). Only MLP/MoE adds comm.
  - Full-attention layers: QKV projection + output projection, both TP-sharded
    (same communication pattern as Qwen3/Megatron, larger dims).
  - Shared expert: additional dense MLP (MegatronMlp pattern), always active.
"""

from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam, MockedParamsBase
from log_analyzer.log import Workload, LogItem


# ---------------------------------------------------------------------------
# Tensor-parallel building blocks (identically-ported from MockedQwen3.py)
# ---------------------------------------------------------------------------

class Qwen3_5RowLinear(MockedModel):
    """Row-parallel linear (output dimension sharded)."""

    def __init__(
        self, input_size, output_size, tp, seq_len, batch_size, layer_id,
        prefix_name, sequence_parallel_enabled=True, computation_enable=False,
        name=None, add_bias_linear=False,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_row"
        self.input_size, self.output_size = input_size, output_size
        self.input_size_per_partition = divide(input_size, tp)
        self.weight = MockedParam((output_size, self.input_size_per_partition), name=name)
        if add_bias_linear:
            self.bias = MockedParam((output_size, 1), name=self.name + "_bias")
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size
        self.comm_size = 2 * seq_len * batch_size * output_size

    def activation_memory(self):
        return self.seq_len * self.input_size

    def forward(self):
        workloads = Workload()
        if self.computation_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=((self.seq_len, self.batch_size, self.input_size_per_partition),
                          (self.input_size_per_partition, self.output_size)),
                stage="forward.Qwen3_5RowLinear." + self.name))
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(LogItem(
                    comm_type=CommType.reduce_scatter,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="forward.Qwen3_5RowLinear"))
            else:
                workloads.append(LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="forward.Qwen3_5RowLinear"))
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="backward.Qwen3_5RowLinear"))
        if self.computation_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=((self.seq_len, self.batch_size, self.output_size), self.weight.shape),
                stage="backward.Qwen3_5RowLinear." + self.name))
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=((self.output_size, self.seq_len * self.batch_size),
                          (self.seq_len * self.batch_size, self.input_size_per_partition)),
                stage="backward.Qwen3_5RowLinear." + self.name))
        return workloads


class Qwen3_5ColumnLinear(MockedModel):
    """Column-parallel linear (input dimension sharded)."""

    def __init__(
        self, input_size, output_size, tp, seq_len, batch_size, layer_id,
        prefix_name, sequence_parallel_enabled=True, computation_enable=False,
        name=None, add_bias_linear=False,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_column"
        self.input_size, self.output_size = input_size, output_size
        self.output_size_per_partition = divide(output_size, tp)
        self.weight = MockedParam((input_size, self.output_size_per_partition), name=name)
        if add_bias_linear:
            self.bias = MockedParam((self.output_size_per_partition, 1), name=self.name + "_bias")
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size
        self.comm_size = 2 * seq_len * batch_size * input_size
        if self.tensor_model_parallel_size > 1 and self.sequence_parallel_enabled:
            self.seq_len *= self.tensor_model_parallel_size

    def activation_memory(self):
        return self.seq_len * self.input_size

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="forward.Qwen3_5ColumnLinear"))
        if self.computation_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=((self.seq_len, self.batch_size, self.input_size),
                          (self.input_size, self.output_size_per_partition)),
                stage="forward.Qwen3_5ColumnLinear." + self.name))
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="backward.Qwen3_5ColumnLinear"))
        if self.computation_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=((self.seq_len, self.batch_size, self.output_size_per_partition),
                          (self.output_size_per_partition, self.input_size)),
                stage="backward.Qwen3_5ColumnLinear." + self.name))
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(LogItem(
                    comm_type=CommType.reduce_scatter,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="backward.Qwen3_5ColumnLinear"))
        if self.computation_enable:
            workloads.append(LogItem(
                comm_type=CommType.computation,
                msg_size=((self.output_size_per_partition, self.seq_len * self.batch_size),
                          (self.seq_len * self.batch_size, self.input_size)),
                stage="backward.Qwen3_5ColumnLinear." + self.name))
        if self.tensor_model_parallel_size > 1:
            if not self.sequence_parallel_enabled:
                workloads.append(LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="backward.Qwen3_5ColumnLinear"))
        return workloads


# ---------------------------------------------------------------------------
# Qwen3.5-specific modules
# ---------------------------------------------------------------------------

class Qwen3_5RMSNorm(MockedModel):
    """RMSNorm with weight parameter."""

    def __init__(self, dim, prefix_name, layer_id=0):
        self.layer_id = layer_id
        self.name = prefix_name
        self.weight = MockedParam((dim, 1), name=prefix_name)


class Qwen3_5FullAttention(MockedModel):
    """
    Standard GQA full attention: every 4th layer.

    head_dim=256, partial_rotary=0.25, attn_output_gate=True, MRoPE.
    Communication pattern: same as MegatronAttention -- TP all-gather on QKV
    column input, TP reduce-scatter on attention output.
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
        self.name = "full_attention_layer"
        self.layer_id = layer_id
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.query_projection_size = num_attention_heads * head_dim
        self.kv_projection_size = num_key_value_heads * head_dim

        # QKV projection: hidden_size -> Q + K + V (combined)
        total_qkv_dim = self.query_projection_size + 2 * self.kv_projection_size
        self.qkv = Qwen3_5ColumnLinear(
            hidden_size, total_qkv_dim, tp, seq_len, batch_size, layer_id,
            "full_attention",
            sequence_parallel_enabled, computation_enable,
            name="qkv_column", add_bias_linear=add_bias_linear,
        )

        # Output projection: heads * head_dim -> hidden_size
        self.attention_dense = Qwen3_5RowLinear(
            self.query_projection_size, hidden_size, tp, seq_len, batch_size,
            layer_id, "full_attention",
            sequence_parallel_enabled, computation_enable,
            name="attention_row", add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.qkv.forward())
        # MRoPE, partial_rotary, attn_output_gate happen here: all local compute
        workloads.extend(self.attention_dense.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.qkv.backward())
        workloads.extend(self.attention_dense.backward())
        return workloads


class Qwen3_5GatedDeltaNet(MockedModel):
    """
    Linear attention (O(L) complexity): 75% of layers.

    State recurrence: S_t = S_{t-1} * alpha * (I - beta * k * k^T) + beta * v * k^T
    with causal_conv1d(kernel_dim=4) for local positional info.

    Communication model (comm-accurate):
      - QKVZ + BA input projection: replicated across TP ranks, local compute
      - Causal conv1d: local, no communication
      - Gated delta rule recurrence: local, no communication
      - Output projection: replicated across TP ranks, local compute

    All operations in GatedDeltaNet are local compute -- the projections are
    replicated (not TP-sharded) because their parameter count is small relative
    to the MLP/MoE that follows, and the recurrent state makes TP complex.
    forward()/backward() return empty Workload.

    Raw MockedParam objects track full-size parameters for accurate DP gradient
    sync sizing in the step() method.
    """

    def __init__(
        self,
        hidden_size,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_conv_kernel_dim,
        layer_id,
    ):
        self.name = "gated_deltanet_layer"
        self.layer_id = layer_id

        # QKVZ combined projection:
        #   Q = linear_num_key_heads * linear_key_head_dim
        #   K = linear_num_key_heads * linear_key_head_dim
        #   V = linear_num_value_heads * linear_value_head_dim
        #   Z = linear_num_value_heads * linear_value_head_dim  (output gate)
        q_dim = linear_num_key_heads * linear_key_head_dim
        v_dim = linear_num_value_heads * linear_value_head_dim
        qkvz_dim = q_dim * 2 + v_dim * 2

        # BA projection: beta gate + decay log-delta per value head
        ba_dim = linear_num_value_heads * 2

        # Combined input projection weight: full-size, replicated (no TP)
        self.in_proj_weight = MockedParam(
            (hidden_size, qkvz_dim + ba_dim), name="gdn_in_proj"
        )

        # Causal conv1d weight: small parameter, local
        self.conv_weight = MockedParam(
            (linear_num_key_heads, 1, linear_conv_kernel_dim), name="gdn_conv1d"
        )

        # Output projection weight: full-size, replicated (no TP)
        self.out_proj_weight = MockedParam(
            (v_dim, hidden_size), name="gdn_out_proj"
        )

    def forward(self):
        """All local compute -- no TP collectives generated."""
        return Workload()

    def backward(self):
        """All local compute -- no TP collectives generated."""
        return Workload()


class Qwen3_5Mlp(MockedModel):
    """
    Qwen3.5 SwiGLU MLP for dense models.

    Uses intermediate_size from config.
    gate_proj + up_proj (combined ColumnLinear) -> down_proj (RowLinear).
    """

    def __init__(
        self, hidden_size, intermediate_size, tp, seq_len, batch_size, layer_id,
        sequence_parallel_enabled=True, computation_enable=False, add_bias_linear=False,
    ):
        self.name = "mlp_layer"
        self.layer_id = layer_id
        # SwiGLU: gate_proj and up_proj combined -> 2 * intermediate_size
        self.gate_up_proj = Qwen3_5ColumnLinear(
            hidden_size, 2 * intermediate_size, tp, seq_len, batch_size, layer_id,
            "mlp", sequence_parallel_enabled, computation_enable,
            name="mlp_gate_up_column", add_bias_linear=add_bias_linear,
        )
        self.down_proj = Qwen3_5RowLinear(
            intermediate_size, hidden_size, tp, seq_len, batch_size, layer_id,
            "mlp", sequence_parallel_enabled, computation_enable,
            name="mlp_down_row", add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.gate_up_proj.forward())
        workloads.extend(self.down_proj.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.gate_up_proj.backward())
        workloads.extend(self.down_proj.backward())
        return workloads


class Qwen3_5MoEMLP(MockedModel):
    """
    Qwen3.5 MoE layer with shared experts.

    Standard Megatron MoE pattern:
      - All-to-all dispatch/combine across EP for routed experts.
      - TP all-gather / reduce-scatter within MoE computation.
      - Shared expert: independent dense SwiGLU MLP, always active, no routing.

    Qwen3.5 MoE specifics:
      - 256 experts (122B-A10B, 35B-A3B) or 512 experts (397B-A17B)
      - top-8 or top-10 routing
      - moe_intermediate_size per expert
      - 1 shared expert with shared_expert_intermediate_size
    """

    def __init__(
        self,
        batch_size,
        hidden_size,
        tp,
        expert_model_parallel_size,
        moe_intermediate_size,
        seq_len,
        topk,
        num_experts,
        shared_expert_intermediate_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        self.name = "moe_layer"
        self.layer_id = layer_id
        self.tp_size = tp
        self.ep_size = expert_model_parallel_size
        self.topk = topk
        self.seq_len = seq_len
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # Routed experts: weights concatenated + TP-sharded
        num_local_experts = num_experts // expert_model_parallel_size
        fc1_output_size = moe_intermediate_size * num_local_experts
        fc1_output_size_per_partition = divide(fc1_output_size, tp)
        fc2_input_size = moe_intermediate_size * num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp)
        self.weight1 = MockedParam((hidden_size, fc1_output_size_per_partition), name="moe_gate_up")
        self.weight2 = MockedParam((fc2_input_size_per_partition, hidden_size), name="moe_down")

        # Shared expert: dense SwiGLU MLP, always active
        if shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3_5Mlp(
                hidden_size, shared_expert_intermediate_size, tp, seq_len,
                batch_size, layer_id,
                sequence_parallel_enabled, computation_enable, add_bias_linear,
            )
        else:
            self.shared_expert = None

    def _dispatch_combine(self, stage):
        """All-to-all token dispatch and combine across EP group."""
        workloads = Workload()
        if self.ep_size > 1:
            # Dispatch: send tokens to expert-owning EP ranks
            # Dividing by ep_size: each EP rank receives ~1/ep of total tokens
            workloads.append(LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                msg_size=(self.seq_len * self.hidden_size * self.batch_size * self.topk // self.tp_size // self.ep_size) * 2,
                stage=f"{stage}.MoE.dispatch",
            ))
        if self.tp_size > 1:
            # Within TP: gather full token batch for grouped GEMM
            # Dividing by ep_size: after dispatch each EP rank holds ~1/ep of tokens
            workloads.append(LogItem(
                comm_type=CommType.all_gather,
                comm_group=CommGroup.tp_group,
                msg_size=2 * self.hidden_size * self.topk * self.batch_size * self.seq_len // self.ep_size,
                stage=f"{stage}.MoE.permutation",
            ))
        return workloads

    def _combine_undispatch(self, stage):
        """Reverse of dispatch: reduce-scatter TP then all-to-all EP."""
        workloads = Workload()
        if self.tp_size > 1:
            # Dividing by ep_size: after dispatch each EP rank holds ~1/ep of tokens
            workloads.append(LogItem(
                comm_type=CommType.reduce_scatter,
                comm_group=CommGroup.tp_group,
                msg_size=2 * self.hidden_size * self.batch_size * self.topk * self.seq_len // self.ep_size,
                stage=f"{stage}.MoE.unpermutation",
            ))
        if self.ep_size > 1:
            # Dividing by ep_size: each EP rank sends ~1/ep of total tokens
            workloads.append(LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                msg_size=(self.seq_len * self.hidden_size * self.batch_size * self.topk // self.tp_size // self.ep_size) * 2,
                stage=f"{stage}.MoE.combine",
            ))
        return workloads

    def forward(self):
        workloads = Workload()
        if self.shared_expert is not None:
            workloads.extend(self.shared_expert.forward())
        if self.tp_size > 1 or self.ep_size > 1:
            workloads.append(LogItem(
                comm_type=CommType.all_gather,
                comm_group=CommGroup.ep_group,
                msg_size=2 * self.ep_size * self.num_experts * self.tp_size,
                stage="forward.MoE.preprocess",
            ))
        workloads.extend(self._dispatch_combine(stage="forward"))
        workloads.extend(self._combine_undispatch(stage="forward"))
        return workloads

    def backward(self):
        workloads = Workload()
        if self.shared_expert is not None:
            workloads.extend(self.shared_expert.backward())
        workloads.extend(self._dispatch_combine(stage="backward"))
        workloads.extend(self._combine_undispatch(stage="backward"))
        return workloads


class Qwen3_5TransformerLayer(MockedModel):
    """
    Hybrid Qwen3.5 transformer layer.

    75% GatedDeltaNet (linear attention) + 25% full GQA attention.
    Layer type determined by full_attention_interval: every Nth layer
    uses full attention, all others use GatedDeltaNet.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        full_attention_interval,
        # GatedDeltaNet params
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_conv_kernel_dim,
        # MoE params (ignored when moe_enable=False)
        moe_enable,
        moe_intermediate_size,
        moe_router_topk,
        num_experts,
        expert_model_parallel_size,
        shared_expert_intermediate_size,
        # Standard training params
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
    ):
        self.name = "transformer_layer"
        self.layer_id = layer_id

        self.input_layernorm = Qwen3_5RMSNorm(
            hidden_size, prefix_name="input_layernorm", layer_id=layer_id
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            hidden_size, prefix_name="post_attention_layernorm", layer_id=layer_id
        )

        # Determine attention type: every full_attention_interval-th layer is full
        is_full_attention = (layer_id + 1) % full_attention_interval == 0

        if is_full_attention:
            self.attention = Qwen3_5FullAttention(
                hidden_size, num_attention_heads, num_key_value_heads, head_dim,
                tp, seq_len, batch_size, layer_id,
                sequence_parallel_enabled, computation_enable, add_bias_linear,
            )
        else:
            self.attention = Qwen3_5GatedDeltaNet(
                hidden_size, linear_key_head_dim, linear_value_head_dim,
                linear_num_key_heads, linear_num_value_heads,
                linear_conv_kernel_dim,
                layer_id,
            )

        # MLP or MoE
        if moe_enable:
            self.mlp = Qwen3_5MoEMLP(
                batch_size, hidden_size, tp, expert_model_parallel_size,
                moe_intermediate_size, seq_len, moe_router_topk,
                num_experts, shared_expert_intermediate_size, layer_id,
                sequence_parallel_enabled, computation_enable, add_bias_linear,
            )
        else:
            self.mlp = Qwen3_5Mlp(
                hidden_size, intermediate_size, tp, seq_len, batch_size, layer_id,
                sequence_parallel_enabled, computation_enable, add_bias_linear,
            )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.attention.forward())
        workloads.extend(self.mlp.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.attention.backward())
        workloads.extend(self.mlp.backward())
        return workloads


class Qwen3_5Embedding(MockedModel):
    """Qwen3.5 embedding layer with TP vocabulary sharding."""

    def __init__(self, vocab_size, hidden_size, tp, seq_len, batch_size):
        self.name = "embedding_layer"
        self.layer_id = 0
        num_embedding_per_partition = divide(vocab_size, tp)
        self.word_embedding = MockedParam(
            (num_embedding_per_partition, hidden_size), name="word_embedding"
        )
        self.tensor_model_parallel_size = tp
        self.comm_size = 2 * batch_size * seq_len * hidden_size

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.tp_group,
                comm_group_size=self.tensor_model_parallel_size,
                msg_size=self.comm_size,
                stage="forward.Qwen3_5Embedding"))
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.tp_group,
                comm_group_size=self.tensor_model_parallel_size,
                msg_size=self.comm_size,
                stage="backward.Qwen3_5Embedding"))
        return workloads


class Qwen3_5Model(MockedModel):
    """
    Complete Qwen3.5 model: embedding -> N hybrid transformer layers -> lm_head.

    Supports both dense and MoE variants. MoE params are ignored when
    moe_enable=False.
    """

    def __init__(self, config):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", 256)
        num_layers = config.num_hidden_layers
        vocab_size = config.vocab_size
        tp = config.tensor_model_parallel_size
        seq_len = config.seq_length
        batch_size = config.micro_batch

        # Qwen3.5-specific
        full_attention_interval = getattr(config, "full_attention_interval", 4)
        linear_key_head_dim = getattr(config, "linear_key_head_dim", 128)
        linear_value_head_dim = getattr(config, "linear_value_head_dim", 128)
        linear_num_key_heads = getattr(config, "linear_num_key_heads", 16)
        linear_num_value_heads = getattr(config, "linear_num_value_heads", 16)
        linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

        moe_enable = config.moe_enable
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        moe_router_topk = config.moe_router_topk
        num_experts = config.num_experts
        expert_ep = config.expert_model_parallel_size
        shared_inter = getattr(config, "shared_expert_intermediate_size", 0)

        self.embedding = Qwen3_5Embedding(vocab_size, hidden_size, tp, seq_len, batch_size)

        self.layers = [
            Qwen3_5TransformerLayer(
                hidden_size, intermediate_size,
                num_attention_heads, num_key_value_heads, head_dim,
                full_attention_interval,
                linear_key_head_dim, linear_value_head_dim,
                linear_num_key_heads, linear_num_value_heads,
                linear_conv_kernel_dim,
                moe_enable, moe_intermediate_size, moe_router_topk,
                num_experts, expert_ep, shared_inter,
                tp, seq_len, batch_size, i,
                sequence_parallel_enabled=config.enable_sequence_parallel,
                computation_enable=config.computation_enable,
                add_bias_linear=config.add_bias_linear,
            )
            for i in range(num_layers)
        ]

        self.lm_head = Qwen3_5ColumnLinear(
            hidden_size, vocab_size, tp, seq_len, batch_size,
            num_layers + 1, "lm_head",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=config.add_bias_linear,
        )
        self.final_norm = Qwen3_5RMSNorm(
            hidden_size, prefix_name="final_norm", layer_id=num_layers + 2
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        workloads.extend(self.lm_head.forward())
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.lm_head.backward())
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        return workloads


# ---------------------------------------------------------------------------
# Config / params class
# ---------------------------------------------------------------------------

class Qwen3_5Params(MockedParamsBase):
    """
    Qwen3.5 training configuration.

    Accepts a HuggingFace config.json file and/or command-line args.
    Handles the nested 'text_config' structure from Qwen3.5 HF configs.

    Defaults are for Qwen3.5-9B dense.
    """

    def __init__(self, config_file=None, args=None):
        # Default to Qwen3.5-9B dense
        self.hidden_size = 4096
        self.intermediate_size = 12288
        self.num_hidden_layers = 32
        self.num_attention_heads = 16
        self.num_key_value_heads = 4
        self.head_dim = 256
        self.vocab_size = 248320
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10_000_000
        self.max_position_embeddings = 262144

        # Qwen3.5-specific defaults
        self.full_attention_interval = 4
        self.linear_key_head_dim = 128
        self.linear_value_head_dim = 128
        self.linear_num_key_heads = 16
        self.linear_num_value_heads = 16
        self.linear_conv_kernel_dim = 4

        # Training-specific defaults
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
        self.moe_router_topk = 0
        self.num_experts = 0
        self.moe_intermediate_size = 0
        self.shared_expert_intermediate_size = 0
        self.moe_grouped_gemm = True

        super().__init__("Qwen3.5", "Qwen3.5", config_file, args)

        # Unpack nested text_config if present (HF Qwen3.5 multimodal format)
        if hasattr(self, 'text_config'):
            for key, value in self.text_config.items():
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    args = Qwen3_5Params(config_file)
    model = Qwen3_5Model(args)
    print(f"Model: {args.model_name}")
    print(f"  hidden_size={args.hidden_size}")
    print(f"  intermediate_size={args.intermediate_size}")
    print(f"  num_hidden_layers={args.num_hidden_layers}")
    print(f"  num_attention_heads={args.num_attention_heads}")
    print(f"  num_key_value_heads={args.num_key_value_heads}")
    print(f"  head_dim={args.head_dim}")
    print(f"  vocab_size={args.vocab_size}")
    print(f"  full_attention_interval={args.full_attention_interval}")
    if args.moe_enable:
        print(f"  MoE: experts={args.num_experts} topk={args.moe_router_topk}")
        print(f"  moe_intermediate_size={args.moe_intermediate_size}")
        print(f"  shared_expert_intermediate_size={args.shared_expert_intermediate_size}")
    print(f"  tp={args.tensor_model_parallel_size}")
    print(f"  seq_len={args.seq_length}")
    print(f"  batch_size={args.micro_batch}")
    print(f"  total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  total layers: {len(model.layers)}")

    # Count layer types
    full_layers = sum(1 for i in range(args.num_hidden_layers)
                      if (i + 1) % args.full_attention_interval == 0)
    linear_layers = args.num_hidden_layers - full_layers
    print(f"  layer breakdown: {full_layers} full-attention + {linear_layers} GatedDeltaNet")

    fwd = model.forward()
    print(f"  forward workloads: {len(fwd.workload)}")
    bwd = model.backward()
    print(f"  backward workloads: {len(bwd.workload)}")
