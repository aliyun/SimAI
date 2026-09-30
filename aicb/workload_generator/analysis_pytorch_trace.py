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

import json
from utils.utils import CommGroup, CommType
from log_analyzer.log import LogItem
from workload_generator.mocked_model.MockedModel import MockedModel
from workload_generator.workload_generator import WorkloadGenerator

comm_node = {}
workload = []


class Pytorch_trace_analyer(WorkloadGenerator):
    def __init__(self, args, model, filename):
        super().__init__(args, model)
        self.name = "pytorch_trace"
        self.filename = filename

    def init(self):
        pass

    def string2comm_type(self, s):
        if "all_gather" in s or "_all_gather_base" in s or "_allgather_base" in s:
            return CommType.all_gather
        if "reduce_scatter" in s or "_reduce_scatter_base" in s:
            return CommType.reduce_scatter
        if "all_reduce" in s:
            return CommType.all_reduce
        if "broadcast" in s:
            return CommType.broadcast
        if "barrier" in s:
            return CommType.barrier
        if "reduce" in s:
            return CommType.reduce
        else:
            print(f"can not convert {s} to any comm type")
            exit(0)

    def step(self):
        item = LogItem()
        with open(self.filename) as f:
            data = json.load(f)

        nodes_list = data["nodes"]
        for node in nodes_list:
            if node["name"].startswith("nccl:"):
                name = node["name"].split(":")[1]
                comm_type = self.string2comm_type(name)
                item.comm_type = comm_type
                # TODO: set group in default dp_group , get group in trace info
                item.comm_group = CommGroup.dp_group
                input = node.get("inputs")
                item.msg_size = input[0][3]
                item.item_size = input[0][4]
                self.workload.append(item)


# if __name__ == "__main__":
# a = parse_pytorch_trace_log("llama7b_zero3_rank8.json")
# json_parse("test_json.json")
# panda_parse("test_json.json")
