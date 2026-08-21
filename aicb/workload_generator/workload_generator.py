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

from workload_generator.mocked_model.MockedModel import MockedModel
from utils.utils import CommGroup, CommType, RankGenerator
from utils.rank_mapper import get_rank_list_for_comm_group
from log_analyzer.log import Workload, LogItem


class WorkloadGenerator:
    DEFAULT_ORDER = 'tp-cp-ep-dp-pp'

    def __init__(self, args, model: MockedModel) -> None:
        self.name = "workload_generator"
        self.args = args
        self.model = model
        self.workload = Workload()
        self.epoch = 0

        order = getattr(args, 'order', None) or WorkloadGenerator.DEFAULT_ORDER
        self.rank_generator = RankGenerator(
            tp=args.tensor_model_parallel_size,
            ep=getattr(args, 'expert_model_parallel_size', 1),
            dp=args.dp_num,
            pp=args.pipeline_model_parallel,
            cp=getattr(args, 'context_parallel_size', 1),
            order=order,
        )
        self.workload.rank_generator = self.rank_generator

    def get_ranks(self, comm_group, comm_group_size=None):
        return get_rank_list_for_comm_group(
            self.rank_generator, comm_group, comm_group_size
        )

    def __call__(self):
        args = self.args
        self.workload = Workload()
        self.workload.rank_generator = self.rank_generator
        self.init()
        self.workload.append(LogItem(comm_type=CommType.epoch_end))
        for i in range(args.epoch_num):
            if args.pipeline_model_parallel > 1 and args.frame != "collective_test":
                self.with_pipeline_forward_backward()
                self.step()
            else:
                for _ in range(args.num_microbatches):
                    self.forward()
                    self.backward()
            self.step()
            self.workload.append(LogItem(comm_type=CommType.epoch_end))
        self._fill_ranks()
        return self.workload

    def _fill_ranks(self):
        for item in self.workload.workload:
            if item.ranks is not None:
                continue
            if item.comm_group is None:
                continue
            item.ranks = get_rank_list_for_comm_group(
                self.rank_generator, item.comm_group, item.comm_group_size
            )
            if item.comm_group_size is None and item.ranks:
                item.comm_group_size = len(item.ranks)

    def forward(self):
        pass

    def backward(self):
        pass

    def step(self):
        pass

    def with_pipeline_forward_backward(self):
        pass
