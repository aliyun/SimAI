"""
Provides DeepSeek implementation for MockedModel

Based on https://github.com/deepseek-ai/DeepSeek-V3/tree/f6e34dd26772dd4a216be94a8899276c5dca9e43

@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}


File: MockedDeepSeek.py
License: Apache 2.0
"""

from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.mocked_model.training.MockedMegatron import *
from utils.utils import CommGroup, CommType

# multiplier to convert BF16 to (FP8 + FP32 scale)
# from https://github.com/deepseek-ai/DeepEP/blob/ef70b83e3b35a84aadc5385b02c95c5d1bcf299c/tests/test_internode.py#L194
FP8_FACTOR = (1 + 4 / 128) / 2

class DeepSeekLinear(MockedModel):
    """
    non-sharded Linear for DeepSeek

    Attributes:
        in_feature (int): input dimention
        out_feature (int): output dimention
        computation_enable (bool): if True, add compute LogItem
        name (str): layer name
        bias (bool): if ture, add bias term. Defaults to False
        // maybe don't need these
        seq_len (int):
        batch_size (int):
    """

    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        computation_enable: bool,
        seq_len: int,
        batch_size: int,
        layer_id: int,
        name: str = "",
        bias: bool = False,
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.name = name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.layer_id = layer_id
        self.w = MockedParam(
            (self.in_feature, self.out_feature), name=self.name + "_linear"
        )
        if bias:
            self.bias = MockedParam((out_feature, 1), name=self.name + "_bias")
        self.computation_enable = computation_enable

    def forward(self):
        workloads = Workload()
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.in_feature),
                        (self.in_feature, self.out_feature),
                    ),
                    stage=f"forward.Linear.{self.name}",
                )
            )
        return workloads

    def backward(self):
        workloads = Workload()
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.out_feature, self.seq_len * self.batch_size),
                        (self.seq_len * self.batch_size, self.in_feature),
                    ),
                    stage=f"backward.Linear.{self.name}",
                )
            )
        return workloads


class DeepSeekMLA(MockedModel):
    """Multi Latent Attentnion layer for DeepSeek"""

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
        qk_rope_dim,
        qk_nope_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
    ):
        self.name = "attention_layer_mla"
        self.layer_id = layer_id
        self.qk_dim = qk_nope_dim + qk_rope_dim
        self.v_head_dim = v_head_dim

        self.kv_channels = hidden_size // num_attention_heads
        self.kv_projection_size = self.kv_channels * num_attention_heads
        self.query_projection_size = self.kv_channels * num_attention_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        # Q down projection
        self.wq_a = DeepSeekLinear(
            in_feature=hidden_size,
            out_feature=self.q_lora_rank,
            computation_enable=computation_enable,
            name="attention_linear_q_lora",
            bias=add_bias_linear,
            layer_id=layer_id,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        self.q_norm = FusedLayernorm(self.q_lora_rank)
        self.wq_b = MegatronColumnLinear(
            self.q_lora_rank,
            (num_attention_heads) * self.qk_dim,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_q",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_q_column",
            add_bias_linear=add_bias_linear,
        )

        # KV down projection
        self.wkv_a = DeepSeekLinear(
            in_feature=hidden_size,
            out_feature=self.kv_lora_rank + qk_rope_dim,
            computation_enable=computation_enable,
            name="attention_linear_kv_lora",
            bias=add_bias_linear,
            layer_id=layer_id,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        self.kv_norm = FusedLayernorm(self.kv_lora_rank)
        self.wkv_b = MegatronColumnLinear(
            self.kv_lora_rank,
            num_attention_heads * (qk_nope_dim + self.v_head_dim),
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_kv",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_kv_column",
            add_bias_linear=add_bias_linear,
        )
        self.wo = MegatronRowLinear(
            num_attention_heads * self.v_head_dim,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention_o",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_o_row",
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        # extend for Linear parts

        # for down projected Q
        # from deepseek.py: https://github.com/deepseek-ai/DeepSeek-V3/blob/f6e34dd26772dd4a216be94a8899276c5dca9e43/inference/model.py#L461C1-L461C53
        # q = self.wq_b(self.q_norm(self.wq_a(x)))
        workloads.extend(self.wq_a.forward())
        # since we don't have a forward() impl for norm, thus skip norm
        # workloads.extend(self.q_norm.forward())

        workloads.extend(self.wq_b.forward())
        # add RoPE (ommited here)

        # for down projected KV
        workloads.extend(self.wkv_a.forward())
        # similarly for kv_norm, we skip norm because no forward() impl for norm
        # workloads.extend(self.kv_norm.forward())
        workloads.extend(self.wkv_b.forward())

        # for O
        workloads.extend(self.wo.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        # similar to fwd but .backward()
        workloads.extend(self.wq_a.backward())
        workloads.extend(self.wq_b.backward())
        workloads.extend(self.wkv_a.backward())
        workloads.extend(self.wkv_b.backward())
        workloads.extend(self.wo.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class DeepSeekMoE(MockedModel):
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
        n_shared_expert,
        sequence_parallel_enabled,
        computation_enable,
        add_bias_linear,
    ):
        self.name = "mlp_moelayer"
        self.layer_id = id
        self.name = "mlp_moelayer"
        self.layer_id = id
        self.expert_model_parallel_size = expert_model_parallel_size
        num_local_experts = num_experts // expert_model_parallel_size
        fc1_output_size = ffn_hidden_size * num_local_experts
        fc1_output_size_per_parttition = divide(fc1_output_size, tp)
        fc2_input_size = ffn_hidden_size * num_local_experts
        fc2_input_size_per_parttition = divide(fc2_input_size, tp)
        self.weight1 = MockedParam((hidden_size, fc1_output_size_per_parttition))
        self.weight2 = MockedParam((fc2_input_size_per_parttition, hidden_size))
        self.weight3 = MockedParam((hidden_size, fc2_input_size_per_parttition))
        self.tp_size = tp
        self.topk = topk
        self.seq_len = seq_len
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.shared_experts = None
        if n_shared_expert > 0:
            self.shared_experts = MegatronMlp(
                hidden_size,
                ffn_hidden_size,
                tp,
                seq_len,
                batch_size,
                id,
                sequence_parallel_enabled,
                computation_enable,
                add_bias_linear,
            )

    def permutation(self, stage):
        workloads = Workload()
        if self.expert_model_parallel_size > 1:
            # only for FWD
            # FP8 dispatch, include input/128 matrix for scale
            # based on DeepEP https://github.com/parthpower/DeepEP/commit/50aee15f592bc22142eb04b7d718296b19613ae9
            if stage == "forward":
                scaled = FP8_FACTOR
            else:
                scaled = 1
            workloads.append(
                LogItem(
                    comm_type=CommType.all_to_all,
                    comm_group=CommGroup.ep_group,
                    msg_size=(
                        self.seq_len * self.hidden_size * self.batch_size * self.topk // self.tp_size
                    )
                    * 2
                    * scaled,
                    stage=f"{stage}.MoE.dispatch",
                )
            )
        if self.tp_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    msg_size=(
                        self.hidden_size * self.topk * self.batch_size * self.seq_len
                    )
                    * 2,
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
                    msg_size=self.hidden_size
                    * self.batch_size
                    * self.topk
                    * self.seq_len
                    * 2,
                    stage=f"{stage}.MoE.unpermutation",
                )
            )

        if self.expert_model_parallel_size > 1:
            # bf16 all-to-all combine
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

    def moe_mlp_forward(self):
        workloads = Workload()
        workloads.extend(self.permutation(stage="forward"))
        workloads.extend(self.unpermutation(stage="forward"))
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def moe_mlp_backward(self):
        workloads = Workload()
        self.permutation(stage="backward")
        self.unpermutation(stage="backward")
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def forward(self):
        wl = Workload()
        wl.extend(self.moe_mlp_forward())
        if self.shared_experts != None:
            wl.extend(self.shared_experts.forward())
        return wl

    def backward(self):
        wl = Workload()
        if self.shared_experts != None:
            wl.extend(self.shared_experts.backward())
        wl.extend(self.moe_mlp_backward())
        return wl


class DeepSeekTransformer(MockedModel):
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
        qk_rope_dim,
        qk_nope_dim,
        v_head_dim,
        expert_model_parallel_size,
        ffn_hidden_size,
        moe_router_topk,
        num_experts,
        n_shared_expert,
        q_lora_rank,
        kv_lora_rank,
        use_dense,
    ):
        self.attention = DeepSeekMLA(
            num_attention_heads,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            sequence_parallel_enabled,
            computation_enable,
            add_bias_linear,
            qk_rope_dim,
            qk_nope_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
        )
        self.pre_mlp_layernorm = FusedLayernorm(hidden_size)
        self.post_attention_layernorm_bias = MockedParam((hidden_size, 1))
        if use_dense:
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
        else:
            self.mlp = DeepSeekMoE(
                batch_size,
                hidden_size,
                tp,
                expert_model_parallel_size,
                ffn_hidden_size,
                seq_len,
                moe_router_topk,
                num_experts,
                layer_id,
                n_shared_expert,
                sequence_parallel_enabled,
                computation_enable,
                add_bias_linear,
            )

    def forward(self):
        worklods = Workload()
        worklods.extend(self.attention.forward())
        worklods.extend(self.mlp.forward())
        return worklods

    def backward(self):
        workloads = Workload()
        workloads.extend(self.attention.backward())
        workloads.extend(self.mlp.backward())
        return workloads


class DeepSeekV3Model(MockedModel):
    def __init__(self, config):
        self.embedding = MegatronEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
        )
        self.layers = [
            DeepSeekTransformer(
                num_attention_heads=config.num_attention_heads,
                hidden_size=config.hidden_size,
                add_bias_linear=config.add_bias_linear,
                batch_size=config.micro_batch,
                computation_enable=config.computation_enable,
                expert_model_parallel_size=config.expert_model_parallel_size,
                ffn_hidden_size=config.ffn_hidden_size,
                layer_id=id,
                moe_router_topk=config.moe_router_topk,
                n_shared_expert=config.n_shared_expert,
                num_experts=config.num_experts,
                qk_nope_dim=config.qk_nope_dim,  # new
                qk_rope_dim=config.qk_rope_dim,  # new
                seq_len=config.seq_length,
                sequence_parallel_enabled=config.enable_sequence_parallel,
                tp=config.tensor_model_parallel_size,
                q_lora_rank=config.q_lora_rank,  # new
                kv_lora_rank=config.kv_lora_rank,  # new
                v_head_dim=config.v_head_dim,  # new
                use_dense=(id < config.n_dense_layers),
            )
            for id in range(config.num_layers)
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
            fwd = layer.forward()
            if not isinstance(fwd, Workload):
                continue
            workloads.extend(fwd)

        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            bwd = layer.backward()
            if not isinstance(bwd, Workload):
                continue
            workloads.extend(bwd)
        workloads.extend(self.embedding.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads
