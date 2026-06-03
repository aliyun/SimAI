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

"""
python -m workload_generator.generate_deepspeed_stage3_workload \
  --stage=3 --world_size=256 --global_batch=1024 --num_attention_heads=40 \
  --num_layers=40 --epoch_num=100 --hidden_size=5120 --ffn_hidden_size=13824 \
  --reduce_bucket_size=26214400 --allgather_bucket_size=500000000 --contiguous_gradients
"""

from workload_generator.mocked_model.training.MockedDeepspeed import DeepspeedForCausalLM
from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.workload_generator import WorkloadGenerator
from utils.utils import CommGroup, CommType, get_params, WorkloadWriter
from collections import deque, defaultdict
from log_analyzer.log import LogItem


class DeepSpeedStage3(WorkloadGenerator):
    """workload generator for deepspeed engine setup
    mock comm behavior of DeepSpeedEngine.__init__
    """

    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.name = "deepspeed_stage3"
        self.amp_enabled = args.amp_enabled
        self.dp_world_size = args.dp_num
        self.batch_size = args.micro_batch
        self.seq_len = args.seq_length
        self.compute_enable = args.computation_enable
        self.reduce_bucket, self.reduce_bucket_size = 0, args.reduce_bucket_size
        self.prefetch_bucket_size = args.prefetch_bucket_size
        self.max_live_parameters, self.current_live_parameters = (
            args.max_live_parameters,
            0,
        )
        self.stage, self._param_queue, self.all_params = (
            "init",
            deque(),
            list(self.model.parameters()),
        )
        self.__param_order = [
            (param, step_id)
            for step_id, param in enumerate(self.all_params + self.all_params[::-1])
        ]
        self.__most_recent_step_id_param_fetched_for = defaultdict(lambda: -1)
        self._mark_persistent_parameters(
            args.param_persistence_threshold, args.model_persistence_threshold
        )

    def _mark_persistent_parameters(self, param_threshold, model_threshold):
        self.persistent_params = []
        total_persistent_parameters = 0
        count = 0
        for param in self.model.parameters():
            param.id = count  # is also the step id
            count += 1
            param.ds_persist = False
            param.has_been_allgather = False
            if param.numel() + total_persistent_parameters > model_threshold:
                continue
            if param.numel() <= param_threshold:
                param.ds_persist = True
                self.persistent_params.append(param)
                total_persistent_parameters += param.numel()

    def init(self):
        if not self.amp_enabled:
            for param in self.model.parameters():
                self.workload.append(
                    LogItem(
                        comm_type=CommType.broadcast,
                        comm_group=CommGroup.dp_group,
                        comm_group_size=self.dp_world_size,
                        msg_size=param.msg_size(),
                        stage="init._broadcast_model",
                        src=0,
                    )
                )

        self.workload.append(
            LogItem(
                comm_type=CommType.barrier,
                comm_group=CommGroup.all,
                comm_group_size=self.dp_world_size,
                msg_size=0,
                stage="init._create_fp16_partitions_with_defragmentation",
            )
        )

        for _ in range(2):
            self.workload.append(
                LogItem(
                    comm_type=CommType.barrier,
                    comm_group=CommGroup.all,
                    comm_group_size=self.dp_world_size,
                    msg_size=0,
                    stage="init._setup_for_real_optimizer",
                )
            )

        for param in self.model.parameters():
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.dp_world_size,
                    msg_size=param.msg_size(),
                    stage="init._allgather_params",
                )
            )

    def _compute_for_param(self, param):
        if self.stage == "forward":
            if param.get_shape()[-1] != 1:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.computation,
                        msg_size=(
                            (self.batch_size, self.seq_len, param.get_shape()[0]),
                            (param.get_shape()[0], param.get_shape()[1]),
                        ),
                        stage=f"{self.stage}.computation",
                    )
                )
        if self.stage == "backward":
            # input grad
            if param.get_shape()[-1] != 1:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.computation,
                        msg_size=(
                            (self.batch_size, self.seq_len, param.get_shape()[0]),
                            (param.get_shape()[0], param.get_shape()[1]),
                        ),
                        stage=f"{self.stage}.computation",
                    )
                )

                # weight grad
                self.workload.append(
                    LogItem(
                        comm_type=CommType.computation,
                        msg_size=(
                            (param.get_shape()[0], self.batch_size * self.seq_len),
                            (self.batch_size * self.seq_len, param.get_shape()[1]),
                        ),
                    )
                )

    def _gather_param_directly(self, param):
        if not param.has_been_allgather:
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.dp_world_size,
                    msg_size=param.msg_size(),
                    stage=f"{self.stage}.allgather_fn",
                )
            )
            param.has_been_allgather = True
            self.current_live_parameters += param.numel()
        if self.compute_enable:
            self._compute_for_param(param)

    def _gather_param_prefetch(self, param, step_id):
        prefetch_bucket, prefetch_bucket_size = [], 0
        if not param.has_been_allgather:
            prefetch_bucket.append(param)
            prefetch_bucket_size += param.numel()
            future_param, future_step_id = self._param_queue.popleft()
            if future_param != param:
                print(
                    f"WARNING: expected {param.__dict__, step_id} but got {future_param.__dict__, future_step_id}"
                )
            param.has_been_allgather = True
            self.current_live_parameters += param.numel()

        while (
            self._param_queue
            and prefetch_bucket_size < self.prefetch_bucket_size
            and self.current_live_parameters < self.max_live_parameters
        ):
            future_param, step_id = self._param_queue.popleft()
            self.__most_recent_step_id_param_fetched_for[future_param.id] = max(
                step_id, self.__most_recent_step_id_param_fetched_for[future_param.id]
            )
            if future_param.has_been_allgather:
                continue
            prefetch_bucket.append(future_param)
            future_param.has_been_allgather = True
            self.current_live_parameters += future_param.numel()
            prefetch_bucket_size += future_param.numel()

        if prefetch_bucket:
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.dp_world_size,
                    msg_size=sum(param.msg_size() for param in prefetch_bucket),
                    stage=f"{self.stage}.allgather_fn",
                )
            )
            if self.compute_enable:
                for param in prefetch_bucket:
                    self._compute_for_param(param)

    def _partition_param(self, param, step_id):
        if len(self._param_queue) == 0:
            # 这里会错误的释放一些ds_persist的参数，但是不影响整体的模拟
            param.has_been_allgather = False
            self.current_live_parameters -= param.numel()
            return
        if param.ds_persist:
            return
        # 这里说明之后马上还会用到这个param
        if self.__most_recent_step_id_param_fetched_for[param.id] > step_id:
            return
        param.has_been_allgather = False
        self.current_live_parameters -= param.numel()

    def forward(self):
        self.stage = "forward"
        for i, param in enumerate(self.all_params):
            if len(self._param_queue) == 0:
                self._gather_param_directly(param)
            else:
                self._gather_param_prefetch(param, i)
            self._partition_param(param, i)

    def _reduce_param_with_bucket(self, param):
        if param.numel() + self.reduce_bucket > self.reduce_bucket_size:
            self.workload.append(
                LogItem(
                    comm_type=CommType.reduce_scatter,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.dp_world_size,
                    msg_size=self.reduce_bucket * param.elem_size(),
                    stage=f"{self.stage}.reduce_scatter_fn",
                )
            )
            self.reduce_bucket = param.numel()
        else:
            self.reduce_bucket += param.numel()

    def backward(self):
        self.stage = "backward"
        for i, param in enumerate(self.all_params[::-1]):
            if len(self._param_queue) == 0:
                self._gather_param_directly(param)
            else:
                self._gather_param_prefetch(param, i)
            self._partition_param(param, i + len(self.all_params))
            self._reduce_param_with_bucket(param)
        self._param_queue = deque(self.__param_order)
        self.__most_recent_step_id_param_fetched_for = defaultdict(lambda: -1)

    def step(self):
        self.stage = "step"
        self.workload.append(
            LogItem(
                comm_type=CommType.reduce_scatter,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.dp_world_size,
                msg_size=self.reduce_bucket * 2,
                stage=f"{self.stage}.reduce_scatter_fn",
            )
        )
        self.reduce_bucket = 0

        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.dp_world_size,
                msg_size=1,
                stage=f"{self.stage}.has_overflow",
            )
        )
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.dp_world_size,
                msg_size=8,
                stage=f"{self.stage}.get_grad_norm_direct",
            )
        )

        for param in self.model.parameters():
            param.has_been_allgather = False
        self.current_live_parameters = 0

        for param in self.persistent_params:
            self._gather_param_directly(param)


if __name__ == "__main__":
    args = get_params()
    model = DeepspeedForCausalLM(args)
    workload_generator = DeepSpeedStage3(args, model)
    workload = workload_generator()
    filename = f"{workload_generator.name}_{args.model_name}_sp_{args.enable_sequence_parallel}_iteration_{args.epoch_num}_computationEnable_{args.computation_enable}_{args.world_size}n.csv"
    workload.dump(filename)
    if args.enable_visual:
            try:
                from visualize.generate import visualize_output
                base_name = filename.split(".")[0]
                visualize_output(f"./results/mocked_workload/{base_name}_workload.csv",True)
            except ImportError: 
                print("visualize_output is not available because required library is not found")
    # WorkloadWriter.write_workload(workload, args, filename)
