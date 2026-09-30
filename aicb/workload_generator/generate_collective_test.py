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

from utils.utils import CommType, CommGroup, get_params, WorkloadWriter
from log_analyzer.log import LogItem, Workload
from workload_generator.workload_generator import WorkloadGenerator


class Collective_Test(WorkloadGenerator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.args = args
        self.name = "collective_test"

    def init(self):
        iter_num = self.args.iter_num
        for i in range(iter_num):
            # for warmup
            self.workload.append(
                LogItem(
                    comm_type=CommType.get_comm_type(self.args.test_comm),
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=self.args.begin_size,
                    stage="warmup",
                )
            )

    def step(self):
        test_comm = CommType.get_comm_type(self.args.test_comm)
        begin_size = self.args.begin_size
        end_size = self.args.end_size
        curr_size = begin_size
        iter_num = self.args.iter_num
        multi_all_reduce_enable = self.args.multi_all_reduce_enable

        while curr_size <= end_size:
            # self.workload.append(LogItem(comm_type=CommType.epoch_end))
            if not multi_all_reduce_enable:
                for i in range(iter_num):
                    self.workload.append(
                        LogItem(
                            comm_type=test_comm,
                            comm_group=CommGroup.dp_group,
                            comm_group_size=self.args.dp_num,
                            msg_size=curr_size,
                            stage="test_step",
                        )
                    )
                curr_size *= 2
            else:
                for i in range(iter_num):
                    self.workload.append(
                        LogItem(
                            comm_type=test_comm,
                            comm_group=CommGroup.pp_group,
                            comm_group_size=self.args.pipeline_model_parallel,
                            msg_size=curr_size,
                            stage="test_step",
                        )
                    )
                curr_size *= 2


if __name__ == "__main__":
    args = get_params()
    workload_generator = Collective_Test(args, None)
    workload = workload_generator()
    filename = "multi_all_reduce.csv"
    workload.dump(filename)
