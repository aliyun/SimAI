"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam
from log_analyzer.log import Workload, LogItem


# mocked version of Megatron RowParallelLinear
class MegatronRowLinear(MockedModel):
    def __init__(
        self,
        input_size,
        output_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        prefix_name,
        sequence_parallel_enabled=True,
        computation_enable=False,
        name=None,
        add_bias_linear=False,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_row"
        self.input_size, self.output_size = input_size, output_size
        self.input_size_per_partition = divide(input_size, tp)
        self.weight = MockedParam(
            (output_size, self.input_size_per_partition), name=name
        )
        if add_bias_linear:
            self.bias = MockedParam((output_size, 1), name=self.name + "_bias")
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size
        self.comm_size = 2 * seq_len * batch_size * output_size
    
    def activation_memory(self):
        # ctx.save_for_backward(input, weight)
        return self.seq_len * self.input_size

    def forward(self):
        workloads = Workload()
        # output_ = torch.matmul(total_input, weight.t()): (s, b, h)
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.input_size_per_partition),
                        (self.input_size_per_partition, self.output_size),
                    ),
                    stage="forward.MegatronRowLinear." + self.name,
                )
            )
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                # output_ = reduce_scatter_to_sequence_parallel_region(output_parallel): (s/tp, b, h)
                workloads.append(
                    LogItem(
                        comm_type=CommType.reduce_scatter,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="forward.MegatronRowLinear",
                    )
                )
            else:
                # output_ = reduce_from_tensor_model_parallel_region(output_parallel)
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="forward.MegatronRowLinear",
                    )
                )
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                # output_ = reduce_scatter_to_sequence_parallel_region(output_parallel): (s/tp, b, h)
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_gather,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronRowLinear",
                    )
                )
        # grad_input = grad_output.matmul(weight): (s, b, h)*(h, h'/N)
        # grad_weight = grad_output.t().matmul(total_input): (h, s*b)*(s*b, h'/N)
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.output_size),
                        self.weight.shape,
                    ),
                    stage="backward.MegatronRowLinear" + self.name,
                )
            )
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.output_size, self.seq_len * self.batch_size),
                        (self.seq_len * self.batch_size, self.input_size_per_partition),
                    ),
                    stage="backward.MegatronRowLinear" + self.name,
                )
            )
        return workloads


class MegatronColumnLinear(MockedModel):
    def __init__(
        self,
        input_size,
        output_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        prefix_name,
        sequence_parallel_enabled=True,
        computation_enable=False,
        name=None,
        add_bias_linear=False,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_column"
        self.input_size, self.output_size = input_size, output_size
        self.output_size_per_partition = divide(output_size, tp)
        self.weight = MockedParam(
            (input_size , self.output_size_per_partition), name=name
        )
        if add_bias_linear:
            self.bias = MockedParam(
                (self.output_size_per_partition, 1), name=self.name + "_bias"
            )
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size
        self.comm_size = 2 * seq_len * batch_size * input_size
        if self.tensor_model_parallel_size > 1 and self.sequence_parallel_enabled:
            self.seq_len *= self.tensor_model_parallel_size
    
    def activation_memory(self):
        # ctx.save_for_backward(input, weight)
        return self.seq_len * self.input_size

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_gather,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="forward.MegatronColumnLinear",
                    )
                )
        # output = torch.matmul(total_input, weight.t())
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.input_size),
                        (self.input_size, self.output_size_per_partition),
                    ),
                    stage="forward.MegatronColumnLinear." + self.name,
                )
            )
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_gather,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronColumnLinear",
                    )
                )
        # grad_input = grad_output.matmul(weight): (s, b, h'/N)*(h'/N, h)
        # grad_weight = grad_output.t().matmul(total_input): (h, s*b)*(s*b, h'/N)
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.output_size_per_partition),
                        (self.output_size_per_partition, self.input_size),
                    ),
                    stage="backward.MegatronColumnLinear." + self.name,
                )
            )
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.reduce_scatter,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronColumnLinear",
                    )
                )
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (
                            self.output_size_per_partition,
                            self.seq_len * self.batch_size,
                        ),
                        (self.seq_len * self.batch_size, self.input_size),
                    ),
                    stage="backward.MegatronColumnLinear." + self.name,
                )
            )
        if self.tensor_model_parallel_size > 1:
            if not self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronColumnLinear",
                    )
                )
        return workloads


class FusedLayernorm(MockedModel):
    def __init__(self, hidden_size):
        self.layer_id = 0
        self.name = "fused"
        self.weight = MockedParam((hidden_size, 1))
        self.bias = MockedParam((hidden_size, 1))


class MegatronAttention(MockedModel):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled,
        computation_enable,
        add_bias_linear,
    ):
        self.name = "attention_layer"
        self.layer_id = layer_id
        self.kv_channels = hidden_size // num_attention_heads
        self.kv_projection_size = self.kv_channels * num_attention_heads
        self.query_projection_size = self.kv_channels * num_attention_heads
        self.qkv = MegatronColumnLinear(
            hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_column",
            add_bias_linear=add_bias_linear,
        )
        self.attention_dense = MegatronRowLinear(
            self.query_projection_size,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_row",
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.qkv.forward())
        workloads.extend(self.attention_dense.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.qkv.backward())
        workloads.extend(self.attention_dense.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class MegatronMlp(MockedModel):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled,
        computation_enable,
        add_bias_linear,
    ):
        self.name = "mlp_layer"
        self.layer_id = layer_id
        self.dense_h_to_4h = MegatronColumnLinear(
            hidden_size,
            ffn_hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mlp",
            sequence_parallel_enabled,
            computation_enable,
            name="mlp_column",
            add_bias_linear=add_bias_linear,
        )
        self.dense_4h_to_h = MegatronRowLinear(
            ffn_hidden_size,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mlp",
            sequence_parallel_enabled,
            computation_enable,
            name="mlp_row",
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.dense_h_to_4h.forward())
        workloads.extend(self.dense_4h_to_h.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.dense_h_to_4h.backward())
        workloads.extend(self.dense_4h_to_h.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class MOEMLP(MockedModel):
    def __init__(
        self,
        batch_size,
        hidden_size,
        tp,
        expert_model_parallel_size,
        ffn_hidden_size,
        seq_len,
        topk,
        num_experts,
        id,
    ):
        self.name = "mlp_moelayer"
        self.layer_id = id
        num_local_experts = num_experts // expert_model_parallel_size
        fc1_output_size = ffn_hidden_size * num_local_experts
        fc1_output_size_per_parttition = divide(fc1_output_size, tp)
        fc2_input_size = ffn_hidden_size * num_local_experts
        fc2_input_size_per_parttition = divide(fc2_input_size, tp)
        self.weight1 = MockedParam((hidden_size, fc1_output_size_per_parttition))
        self.weight2 = MockedParam((fc2_input_size_per_parttition, hidden_size))
        self.tp_size = tp
        self.ep_size = expert_model_parallel_size
        self.topk = topk
        self.seq_len = seq_len
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def permutation(self, stage):
        workloads = Workload()
        if self.ep_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_to_all,
                    comm_group=CommGroup.ep_group,
                    msg_size=(
                        self.seq_len * self.hidden_size * self.batch_size * self.topk // self.tp_size
                    )
                    * 2,
                    stage=f"{stage}.MoE.dispatch",
                )
            )
        if self.tp_size > 1:
            # TODO:we assume tokens consistent split to all experts, but actually its not
            workloads.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    msg_size=2
                    * self.hidden_size
                    * self.topk * self.batch_size
                    * self.seq_len,
                    stage=f"{stage}.MoE.permutation",
                )
            )

        return workloads

    def unpermutation(self, stage):
        workloads = Workload()
        if self.tp_size > 1:
            # TODO:we assume tokens consistent split to all experts, but actually its not
            workloads.append(
                LogItem(
                    comm_type=CommType.reduce_scatter,
                    comm_group=CommGroup.tp_group,
                    msg_size=2
                    * self.hidden_size * self.batch_size
                    * self.topk
                    * self.seq_len,
                    stage=f"{stage}.MoE.unpermutation",
                )
            )

        if self.ep_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_to_all,
                    comm_group=CommGroup.ep_group,
                    msg_size=(
                        self.seq_len * self.hidden_size * self.batch_size * self.topk // self.tp_size
                    )
                    * 2,
                    stage=f"{stage}.MoE.combine",
                )
            )

        return workloads

    def forward(self):
        workloads = Workload()
        if self.tp_size > 1 or self.ep_size > 1:
            workloads.append(LogItem(
                        comm_type=CommType.all_gather,
                        comm_group=CommGroup.ep_group,
                        msg_size=2 * self.ep_size * self.num_experts * self.tp_size,
                        stage=f"forward.MoE.preprocess",
                    ))
        workloads.extend(self.permutation(stage="forward"))
        workloads.extend(self.unpermutation(stage="forward"))
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        self.permutation(stage="backward")
        self.unpermutation(stage="backward")
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class SequentialMLP(MockedModel):
    def __init__(self):
        print("Not implement yet!")
        pass


class MegatronTransformorLayer(MockedModel):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        tp,
        seq_len,
        batch_size,
        num_attention_heads,
        layer_id,
        expert_model_parallel_size,
        moe_router_topk,
        num_experts,
        moe_grouped_gemm=True,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
        moe_enable=False,
    ):
        self.attention = MegatronAttention(
            num_attention_heads,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            sequence_parallel_enabled,
            computation_enable,
            add_bias_linear,
        )
        self.pre_mlp_layernorm = FusedLayernorm(hidden_size)
        self.post_attention_layernorm_bias = MockedParam((hidden_size, 1))
        if moe_enable:
            self.mlp = MOEMLP(
                batch_size,
                hidden_size,
                tp,
                expert_model_parallel_size,
                ffn_hidden_size,
                seq_len,
                moe_router_topk,
                num_experts,
                layer_id,
            )
        else:
            self.mlp = MegatronMlp(
                hidden_size,
                ffn_hidden_size,
                tp,
                seq_len,
                batch_size,
                layer_id,
                sequence_parallel_enabled,
                computation_enable,
                add_bias_linear,
            )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.attention.forward())
        workloads.extend(self.mlp.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.attention.backward())
        workloads.extend(self.mlp.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class MegatronEmbedding(MockedModel):
    def __init__(self, padded_vocab_size, hidden_size, tp, seq_len, batch_size):
        self.name = "embedding_layer"
        self.layer_id = 0
        num_embedding_per_partition = divide(padded_vocab_size, tp)
        self.word_embedding = MockedParam(
            (4 * num_embedding_per_partition, hidden_size), name=self.name
        )
        self.tensor_model_parallel_size = tp
        # TODO : position embedding shape is max_sequence_length not sequence_length
        self.position_embedding = MockedParam((seq_len, hidden_size))
        self.comm_size = 2 * batch_size * seq_len * hidden_size

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="forward.MegatronEmbedding",
                )
            )
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="backward.MegatronEmbedding",
                )
            )
        return workloads


class MegatronModel(MockedModel):
    def __init__(self, config):
        self.embedding = MegatronEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
        )
        self.layers = [
            MegatronTransformorLayer(
                config.hidden_size,
                config.ffn_hidden_size,
                config.tensor_model_parallel_size,
                config.seq_length,
                config.micro_batch,
                config.num_attention_heads,
                i,
                config.expert_model_parallel_size,
                config.moe_router_topk,
                config.num_experts,
                config.moe_grouped_gemm,
                config.enable_sequence_parallel,
                config.computation_enable,
                config.add_bias_linear,
                config.moe_enable,
            )
            for i in range(config.num_layers)
        ]
        self.final_norm = MegatronColumnLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
            1,
            "final",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=config.add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads
