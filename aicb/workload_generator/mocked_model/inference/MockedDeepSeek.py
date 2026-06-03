from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam, MockedParamsBase
from log_analyzer.log import Workload, LogItem

# multiplier to convert BF16 to (FP8 + FP32 scale)
# from https://github.com/deepseek-ai/DeepEP/blob/ef70b83e3b35a84aadc5385b02c95c5d1bcf299c/tests/test_internode.py#L194
FP8_FACTOR = (1 + 4 / 128) / 2

class DeepSeekEmbedding(MockedModel):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            tp,
            seq_len,
            batch_size
        ):
        self.name = "embedding_layer"
        self.layer_id = 0
        num_embedding_per_partition = divide(vocab_size, tp) 
        self.weight = MockedParam((num_embedding_per_partition, hidden_size), elem_size= 1)
        self.tensor_model_parallel_size = tp
        self.comm_size = 2 * batch_size * seq_len * hidden_size  #allreduce

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="forward.DeepSeekEmbedding",
                )
            )
        return workloads

    
class DeepSeekRowLinear(MockedModel):
    def __init__(
        self,
        input_size,
        output_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        prefix_name,
        # sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
        elem_size = 1,
        name=None,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_row"
        self.input_size, self.output_size = input_size, output_size
        self.input_size_per_partition = divide(input_size, tp)
        self.weight = MockedParam(
            (output_size, self.input_size_per_partition),elem_size, name=name
            )
        if add_bias_linear:
            self.bias = MockedParam((output_size, 1), elem_size, name=self.name + "_bias")
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_length, self.micro_batch = tp, seq_len, batch_size
        self.comm_size = 2 * seq_len * batch_size * output_size  #allreduce

    def forward(self):
        workloads = Workload()
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_length, self.micro_batch, self.input_size_per_partition),
                        (self.input_size_per_partition, self.output_size),
                    ),
                    stage="forward.DeepSeekRowLinear." + self.name,
                )
            )
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                        LogItem(
                            comm_type=CommType.all_reduce,
                            comm_group=CommGroup.tp_group,
                            comm_group_size=self.tensor_model_parallel_size,
                            msg_size=self.comm_size,
                            stage="forward.DeepSeekRowLinear." + self.name,
                        )
                    )
        return workloads

class DeepSeekColumnLinear(MockedModel):
    def __init__(
        self,
        input_size,
        output_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        prefix_name="",
        # sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
        elem_size = 1,
        name=None,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_column"
        self.input_size, self.output_size = input_size, output_size
        self.output_size_per_partition = divide(output_size, tp)
        self.weight = MockedParam(
            # fp8, elem_size = 1
            (input_size , self.output_size_per_partition), elem_size, name=name
        )
        if add_bias_linear:
            self.bias = MockedParam(
                (self.output_size_per_partition, 1), elem_size, name=self.name + "_bias"
            )
        # self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_length, self.micro_batch = tp, seq_len, batch_size
        # self.comm_size = 1 * seq_len * batch_size * input_size #fp8

    def forward(self):
        workloads = Workload()
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_length, self.micro_batch, self.input_size),
                        (self.input_size, self.output_size_per_partition)
                    ),
                    stage="forward.DeepSeekColumnLinear." + self.name,
                )
            )
        return workloads


class DeepSeekAttention(MockedModel):
    def __init__(
        self,
        hidden_size, 
        tp,
        head_num,
        d_q_c,   #1536
        d_kv_c,  #512
        d_r,    #64
        d_q,
        d_kv,
        seq_len,
        batch_size,
        layer_id,
        computation_enable=False,
        add_bias_linear=False,
        elem_size = 1

    ):
        self.layer_id = layer_id
        self.name = "attention_layer"
        self.hidden_size = hidden_size
        self.tp = tp
        self.n_heads = head_num//tp
        self.q_lora = d_q_c
        self.kv_lora = d_kv_c
        self.rope_dim = d_r,
        self.q_head_dim = d_q,
        self.kv_head_dim = d_kv,
        self.w_qkv_c = DeepSeekColumnLinear(
            hidden_size,
            (d_q_c + d_kv_c + d_r),
            1,  #不分tp
            seq_len,
            batch_size,
            layer_id,
            "attention",
            computation_enable,
            add_bias_linear=False,
            elem_size = 1,
            name="attention_colum"
        )
        #TODO: 每个矩阵的权重没有表征
        # self.weight_a = MockedParam()
        self.w_q_cr = DeepSeekColumnLinear(
            d_q_c,
            (head_num*(d_q + d_r)),
            tp,     #对head分TP
            seq_len,
            batch_size,
            layer_id,
            "attention",
            computation_enable,
            add_bias_linear=False,
            elem_size = 1,
            name="attention_colum"
        )
        #TODO：此处对于softmax(QK) V 的计算感觉不能简单用 ColumnLinear来表示，
        # 但是又占了计算时间，先放着不写，看Mocked—Computation里是否会添加

        self.q_k_v = DeepSeekColumnLinear(
            d_kv_c,
            (head_num * 2 * d_kv),
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention",
            computation_enable,
            add_bias_linear=False,
            elem_size = 1,
            name="attention_colum"
        )

        self.wo = DeepSeekRowLinear(
            (head_num * d_kv),
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention",
            # sequence_parallel_enabled=True,
            computation_enable,
            add_bias_linear=False,
            elem_size = 1,
            name="attention_row",
        )
    def forward(self):
        workloads = Workload()
        workloads.extend(self.w_qkv_c.forward())
        workloads.extend(self.w_q_cr.forward())
        workloads.extend(self.q_k_v.forward())
        workloads.extend(self.wo.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

class DeepSeekMLP(MockedModel):
    def __init__(
          self,
          hidden_size,
          expert_dim,   #ffn_hidden_dim
          tp,
          seq_len,
          batch_size,
          layer_id,
        # sequence_parallel_enabled,
          computation_enable,
          add_bias_linear = False,
          elem_size = 1,
          name=""
    ):
        self.layer_id = layer_id
        self.name = name
        self.w1 = DeepSeekColumnLinear(
                    hidden_size,expert_dim*2,tp,seq_len,
                    batch_size, layer_id, name,computation_enable,add_bias_linear, elem_size,name="mlp_column"
                    )
        self.w2 = DeepSeekRowLinear(
                    expert_dim,hidden_size,tp,seq_len,
                    batch_size, layer_id, name,computation_enable,add_bias_linear, elem_size,name="mlp_row")
    def forward(self):
        workloads = Workload()
        workloads.extend(self.w1.forward())
        workloads.extend(self.w2.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

class DeepSeekMOE(MockedModel):
    def __init__(
            self,
            hidden_size,
            total_experts,
            expert_model_parallel_size,
            experts_topk,
            expert_dim,
            tp,
            seq_len,
            batch_size,
            id,
            shared_experts_cnt,
            computation_enable,
            add_bias_linear,
            elem_size
        ):
        self.tp = tp
        self.name = "sparse_moelayer"
        self.layer_id = id
        num_local_experts = total_experts // expert_model_parallel_size
        # fc1_output_size = expert_dim * num_local_experts
        # fc1_output_size_per_parttition = divide(fc1_output_size, tp)
        # fc2_input_size = expert_dim * num_local_experts
        # fc2_input_size_per_parttition = divide(fc2_input_size, tp)
        # self.weight1 = MockedParam((hidden_size, fc1_output_size_per_parttition))
        # self.weight2 = MockedParam((fc2_input_size_per_parttition, hidden_size))
        # self.tensor_model_parallel_size = tp
        self.expert_model_parallel_size = expert_model_parallel_size
        self.topk = experts_topk
        self.seq_length = seq_len
        self.num_experts = total_experts
        self.micro_batch = batch_size
        self.hidden_size = hidden_size
        self.shared_experts = []
        
        self.w_gate = MockedParam((self.num_experts, hidden_size),elem_size,name = "moe_gate")
        
        self.expert = DeepSeekMLP(
            hidden_size,
            num_local_experts * expert_dim,
            1,  #不分tp
            seq_len,
            batch_size,
            id,
            # sequence_parallel_enabled,
            computation_enable,
            add_bias_linear = False,
            elem_size = 1,
            name = "moe_expert"
        )
        
        for i in range(shared_experts_cnt):
            self.shared_experts.append(
                DeepSeekMLP(
                    hidden_size,
                    expert_dim,   #ffn_hidden_dim
                    1,        #也没分TP
                    seq_len,
                    batch_size,
                    id,
                # sequence_parallel_enabled,
                    computation_enable,
                    add_bias_linear = False,
                    name = "shared_experts"
                )
            )
    def dispatch(self):
        workloads = Workload()
        workloads.append(
            LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                comm_group_size= self.expert_model_parallel_size,
                msg_size=self.seq_length
                * self.hidden_size
                * self.micro_batch
                * (self.topk-1)
                * 2 * FP8_FACTOR,
                stage = "moe.dispatch"
            )
        )
        return workloads
    def combine(self):
        workloads = Workload()
        workloads.append(
            LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                comm_group_size= self.expert_model_parallel_size,
                msg_size=self.seq_length
                * self.hidden_size
                * self.micro_batch
                * (self.topk-1)
                * 2,
                stage = "moe.combine"
            )
        )
        return workloads
    def forward(self):
        workloads = Workload()
        for shared_expert in self.shared_experts:
            workloads.extend(shared_expert.forward())
        workloads.extend(self.dispatch())
        workloads.extend(self.expert.forward())
        workloads.extend(self.combine())
        return workloads
        
        
class DeepSeekRMSNorm(MockedModel):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.name = "RMSNorm"
        self.weight = MockedParam((hidden_dim,1),1, name = self.name)

        
class DeepSeekTransformerLayer(MockedModel):
    def __init__(
            self,
            dense_layer, 
            layerid,
            hidden_size,
            total_experts, 
            expert_model_parallel_size,
            experts_topk,
            expert_dim, 
            tp,
            seq_len,
            batch_size,
            head_num,
            d_q_c,   #1536
            d_kv_c,  #512
            d_r,    #128
            d_q,
            d_kv,
            shared_experts_cnt,
            # sequence_parallel_enabled=True,
            computation_enable=False,
            add_bias_linear=False,
            elem_size = 1
            ):
        # self.atten_norm = DeepSeekRMSNorm(hidden_size)
        self.attention = DeepSeekAttention(
            hidden_size, 
            tp,
            head_num,
            d_q_c,   #1536
            d_kv_c,  #512
            d_r,    #128
            d_q,
            d_kv,
            seq_len,
            batch_size,
            layerid,
            computation_enable=True,
            add_bias_linear=False,
            elem_size = 1,
        )
        # self.ffn_norm = DeepSeekRMSNorm(hidden_size)
        self.id = layerid
        self.dense_layer = dense_layer
        #不知道MockedParam这个类dpsk中是否有用
        if layerid < dense_layer:
            self.mlp = DeepSeekMLP(
                hidden_size,
                expert_dim,   #ffn_hidden_dim
                tp,
                seq_len,
                batch_size,
                layerid,
                # sequence_parallel_enabled,
                computation_enable,
                #  add_bias_linear
                name = "dense_mlp"
            )
        else:   
            self.mlp = DeepSeekMOE(
                hidden_size,
                total_experts,
                expert_model_parallel_size,
                experts_topk,
                expert_dim,
                tp,
                seq_len,
                batch_size,
                layerid,
                shared_experts_cnt,
                # sequence_parallel_enabled,
                computation_enable,
                add_bias_linear,
                elem_size=1
            )
    def forward(self):
        workloads = Workload()
        workloads.extend(self.attention.forward())
        workloads.extend(self.mlp.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class DeepSeekModel(MockedModel):
    def __init__(self, config):
        #embedding层：
        # self.embedding = DeepSeekEmbedding(
        #     config.vocab_size,
        #     config.hidden_size,
        #     config.tensor_model_parallel_size,
        #     config.seq_length,
        #     config.micro_batch
        # )

        #Transformer层：
        self.layers = [
            DeepSeekTransformerLayer(
                config.dense_layer,
                i,
                config.hidden_size,
                config.num_experts, 
                config.expert_model_parallel_size,
                config.moe_router_topk,
                config.expert_dim, 
                config.tensor_model_parallel_size,
                config.seq_length,
                config.micro_batch,
                config.num_attention_heads,
                config.d_q_c,   #1536
                config.d_kv_c,  #512
                config.d_r,    #128
                config.d_q,
                config.d_kv,
                config.shared_experts,
                computation_enable=config.computation_enable,
                add_bias_linear=config.add_bias_linear,
                elem_size = 1
            )
            for i in range(config.num_layers)
        ]

        #Norm层：
        # self.norm = DeepSeekRMSNorm(config.hidden_size)

        #Final_layer
        self.final = DeepSeekColumnLinear(
            config.hidden_size,
            config.vocab_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
            1,
            "final",
            # sequence_parallel_enabled=True,
            computation_enable=config.computation_enable,
            add_bias_linear=config.add_bias_linear,
            elem_size = 1,
            name=None,
        )
    
    def forward(self, config):
        workloads = Workload()
        # workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        # workloads.extend(self.norm.forward())
        workloads.extend(self.final.forward())
        if config.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=config.tensor_model_parallel_size,
                    msg_size=1 * config.micro_batch * config.vocab_size, # fp8, config.vocab_size应该是总的vocab_size(即分到每个的*tp)
                    stage="forward.final",
                )
            )
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class DeepSeekParams(MockedParamsBase):
    def __init__(self, config_file=None, args=None):
        # Initialize base class with default values
        super().__init__("DeepSeek-671B", "DeepSeek", config_file, args)

        # Enable computation_enable
        self.computation_enable = True
        self.add_bias_linear = False

        # Recalculate total_experts if needed
        if hasattr(self, 'router_expert') or hasattr(self, 'duped_expert'):
            self.num_experts = self.router_expert + self.duped_expert


if __name__ == "__main__":
    import sys
    
    # Check if a config file is provided as a command line argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    args = DeepSeekParams(config_file)
    model = DeepSeekModel(args)
    workloads = model.forward(args)
    filename = "testdpsk-1.csv"
    workloads.dump(filename)
    print("Finish Model initialization")