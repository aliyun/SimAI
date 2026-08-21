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

import re
from utils.utils import CommType, CommGroup, WorkloadWriter, get_params


class TraceParser:
    def __init__(self, input_file):
        self.input_file = input_file
        self.comm_workload = []

    def prase_trace(self):

        pattern = r"comm op: (.*?) \|.*?msg size: (.*?) \|.*?algbw \(Gbps\): (.*?) "

        with open(self.input_file, "r") as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    op = match.group(1)
                    comm_op = CommType.get_comm_type(op)
                    msg_size = match.group(2)
                    self.comm_workload.append(
                        {
                            "operation": "trace",
                            "comm_type": comm_op,
                            "msg_size": msg_size,
                            "comm_group": CommGroup.dp_group,
                            "bw": match.group(3),
                        }
                    )

    def get_trace_workload(self):
        return self.comm_workload


if __name__ == "__main__":
    args = get_params()
    output_file = "model_workload/deepspeed_trace.csv"
    paser = TraceParser(
        "llama-7b-ga8-seq2048-bs3_dlcfw77d07c87pho-master-0_2023-07-05 21_48_38.txt"
    )
    paser.prase_trace()
    workload = paser.get_trace_workload()
    WorkloadWriter().write_workload(workload, args, output_file)
