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

"""example of running zero1/2 on llama-13B
python -m workload_generator.deepspeed_stage1_workload \
  --stage=1 --world_size=624 --global_batch=624 \
  --num_layers=40 --epoch_num=2 --hidden_size=5120 --ffn_hidden_size=13696 \
  --reduce_bucket_size=2000000000 --allgather_bucket_size=2000000000

python -m workload_generator.deepspeed_stage1_2_workload \
  --stage=2 --world_size=256 --global_batch=1024 --num_attention_heads=40 \
  --num_layers=40 --epoch_num=100 --hidden_size=5120 --ffn_hidden_size=13824 \
  --reduce_bucket_size=26214400 --allgather_bucket_size=500000000 --contiguous_gradients
"""

import math
from workload_generator.mocked_model.training.MockedDeepspeed import DeepspeedForCausalLM
from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.workload_generator import WorkloadGenerator
from utils.utils import CommGroup, CommType, get_params, WorkloadWriter
from log_analyzer.log import LogItem


class DeepSpeedStage1(WorkloadGenerator):
    """workload generator for deepspeed engine setup
    mock comm behavior of DeepSpeedEngine.__init__
    """

    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.name = "deepspeed_stage1"
        self.compute_enable = args.computation_enable
        self.batch_size = args.micro_batch
        self.seq_len = args.seq_length
        self.reduce_bucket, self.num_in_reduce_bucket, self.max_reduce_bucket_size = (
            [],
            0,
            args.reduce_bucket_size,
        )
        self.allgather_bucket_size = args.allgather_bucket_size
        self.amp_enabled = args.amp_enabled
        self.dp_world_size = args.dp_num
        self.elem_size = 2
        self.all_params = list(self.model.parameters())

    def init(self):
        if not self.amp_enabled:
            for param in self.model.parameters():
                self.workload.append(
                    LogItem(
                        comm_type=CommType.broadcast,
                        comm_group=CommGroup.dp_group,
                        comm_group_size=self.dp_world_size,
                        msg_size=param.msg_size(),
                        stage="init",
                    )
                )

        self.workload.append(
            LogItem(
                comm_type=CommType.barrier,
                comm_group=CommGroup.all,
                comm_group_size=self.dp_world_size,
                msg_size=param.msg_size(),
                stage="init.__init__",
            )
        )

    def forward(self):
        if self.compute_enable:
            self.all_params = list(self.model.parameters())
            for param in self.all_params:
                if param.get_shape()[-1] != 1:
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.computation,
                            msg_size=(
                                (self.batch_size, self.seq_len, param.get_shape()[0]),
                                (param.get_shape()[0], param.get_shape()[1]),
                            ),
                            stage="forward.computation",
                        )
                    )

    def _reduce_ipg_grads(self):
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.dp_world_size,
                msg_size=self.num_in_reduce_bucket * self.elem_size,
                stage=f"{self.current_op}.allreduce_bucket",
            )
        )
        self.reduce_bucket, self.num_in_reduce_bucket = [], 0

    def backward(self):
        self.current_op = "backward"
        for param in self.all_params[::-1]:
            if param.numel() + self.num_in_reduce_bucket > self.max_reduce_bucket_size:
                self._reduce_ipg_grads()
            self.reduce_bucket.append(param)
            self.num_in_reduce_bucket += param.numel()
            if self.compute_enable:
                if param.get_shape()[-1] != 1:
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.computation,
                            msg_size=(
                                (self.batch_size, self.seq_len, param.get_shape()[0]),
                                (param.get_shape()[0], param.get_shape()[1]),
                            ),
                            stage=f"{self.current_op}.computation",
                        )
                    )
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.computation,
                            msg_size=(
                                (param.get_shape()[0], self.batch_size * self.seq_len),
                                (self.batch_size * self.seq_len, param.get_shape()[1]),
                            ),
                            stage=f"{self.current_op}.computation",
                        )
                    )

    def step(self):
        self.current_op = "step"
        self._reduce_ipg_grads()

        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.dp_world_size,
                msg_size=1,
                stage=f"{self.current_op}.has_overflow",
            )
        )
        num_params = sum([param.numel() for param in self.model.parameters()])
        num_shards = max(num_params // self.allgather_bucket_size, 1)
        shard_size = num_params // num_shards

        for i in range(num_shards):
            num_elements = (
                num_params - i * shard_size if i == (num_shards - 1) else shard_size
            )
            padding_size = (
                (self.dp_world_size - num_elements % self.dp_world_size)
                if num_elements % self.dp_world_size
                else 0
            )
            num_elements = num_elements + padding_size
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.dp_world_size,
                    msg_size=num_elements * self.elem_size,
                    stage=f"{self.current_op}.all_gather_dp_groups",
                )
            )


class DeepSpeedStage2(DeepSpeedStage1):
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.name = "deepspeed_stage2"

        self.param_range_map = self.build_model_gbuf_param_range_map(
            model, self.dp_world_size
        )

    def build_model_gbuf_param_range_map(self, model: MockedModel, dp_world_size: int):
        gbuf_size = sum([param.numel() for param in model.parameters()])

        gbuf_partition_size = int(math.ceil(gbuf_size / dp_world_size))
        gbuf_world_all_ranges = []
        for r in range(dp_world_size):
            gbuf_world_start = r * gbuf_partition_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + gbuf_partition_size)
            gbuf_world_all_ranges.append((gbuf_world_start, gbuf_world_end))

        start_idx, r = 0, 0
        gbuf_world_start, gbuf_world_end = gbuf_world_all_ranges[r]
        # record each param should be reduced to which rank(s)
        # param_id: int -> List[(rank: int, param_start_idx: int, param_end_idx: int)]
        param_range_map = {}
        for param in self.all_params:
            # current param in [start_idx, end_idx) range of gbuf
            param_id = id(param)
            param_range_map[param_id] = []
            end_idx = start_idx + param.numel()

            # current rank is in change of [gbuf_world_start, gbuf_world_end) of gbuf
            param_start_idx = start_idx
            # if current rank cannot fully cover this param, move to next rank
            while gbuf_world_end < end_idx:
                param_range_map[param_id].append((r, param_start_idx, gbuf_world_end))
                param_start_idx = gbuf_world_end
                r += 1
                gbuf_world_start, gbuf_world_end = gbuf_world_all_ranges[r]
            param_range_map[param_id].append((r, param_start_idx, end_idx))

            # for next param
            start_idx = end_idx
        return param_range_map

    def _reduce_ipg_grads(self):
        if not self.args.contiguous_gradients:
            super()._reduce_ipg_grads()
            return

        rank_start_end_idx = [[-1, -1, -1]]
        for param in self.reduce_bucket[::-1]:
            for rank, start_idx, end_idx in self.param_range_map[id(param)]:
                if rank == rank_start_end_idx[-1][0]:
                    if rank_start_end_idx[-1][-1] != start_idx:
                        print(f"WARNNING {rank_start_end_idx[-1]} - {start_idx}")
                    rank_start_end_idx[-1][-1] = end_idx
                else:
                    rank_start_end_idx.append([rank, start_idx, end_idx])

        for rank, start_idx, end_idx in rank_start_end_idx[1:]:
            self.workload.append(
                LogItem(
                    comm_type=CommType.reduce,
                    comm_group=CommGroup.dp_group,
                    msg_size=(end_idx - start_idx) * self.elem_size,
                    comm_group_size=self.dp_world_size,
                    stage=f"{self.current_op}.average_tensor",
                    dst=rank,
                )
            )
        self.reduce_bucket, self.num_in_reduce_bucket = [], 0


if __name__ == "__main__":
    args = get_params()
    print(args.__dict__)
    model = DeepspeedForCausalLM(args)
    if args.stage == 1:
        workload_generator = DeepSpeedStage1(args, model)
        filename = f"{workload_generator.name}_{args.model_name}_sp_{args.enable_sequence_parallel}_iteration_{args.epoch_num}_computationEnable_{args.computation_enable}_{args.world_size}n.csv"
    else:
        workload_generator = DeepSpeedStage2(args, model)
        filename = f"{workload_generator.name}_{args.model_name}_sp_{args.enable_sequence_parallel}_iteration_{args.epoch_num}_computationEnable_{args.computation_enable}_{args.world_size}n.csv"
    workload = workload_generator()
    workload.dump(filename)
    if args.enable_visual:
            try:
                from visualize.generate import visualize_output
                base_name = filename.split(".")[0]
                visualize_output(f"./results/mocked_workload/{base_name}_workload.csv",True)
            except ImportError: 
                print("visualize_output is not available because required library is not found")
    # WorkloadWriter.write_workload(workload, args, filename)
