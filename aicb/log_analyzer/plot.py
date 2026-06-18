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

import numpy as np
import matplotlib.pyplot as plt
from log_analyzer.ds_comm_log_analyzer import parse_ds_comm_log
from utils.benchmark_logger import BenchLogger
from typing import Dict, List
from log_analyzer.utils import convert_size_to_msg


def log_boxplot(detailed_comm_info: Dict):
    MAX_ITEMS, COLS = 5, 2
    comm_type2msg_size2time_cost = {}
    for comm_type, comm_group, msg_size in sorted(detailed_comm_info.keys()):
        if (comm_type, comm_group) not in comm_type2msg_size2time_cost:
            comm_type2msg_size2time_cost[(comm_type, comm_group)] = {}
        elasped_time = np.array(
            detailed_comm_info[(comm_type, comm_group, msg_size)]["_elapsed_time"]
        )
        comm_type2msg_size2time_cost[(comm_type, comm_group)][
            msg_size
        ] = elasped_time  # [elasped_time < 3000]
    fig_num = sum(
        [
            (len(comm_info.keys()) + MAX_ITEMS - 1) // MAX_ITEMS
            for comm_info in comm_type2msg_size2time_cost.values()
        ]
    )

    fig_rows, fig_idx = (fig_num + COLS - 1) // COLS, 0
    fig, axes = plt.subplots(nrows=fig_rows, ncols=COLS, figsize=(8, 6))
    fig.tight_layout()
    fig.suptitle("for deepspeed Zero3 llama 13B")
    for (comm_type, comm_group), comm_info in comm_type2msg_size2time_cost.items():
        values, labels = list(comm_info.values()), [
            convert_size_to_msg(msg) for msg in comm_info.keys()
        ]
        for j in range(0, len(values), MAX_ITEMS):
            ax = axes[fig_idx // COLS][fig_idx % COLS]
            fig_idx += 1
            ax.set_title("%s %s msg info" % (comm_type.value, comm_group.value))
            ax.boxplot(
                values[j : j + MAX_ITEMS],
                labels=labels[j : j + MAX_ITEMS],
                flierprops=dict(
                    marker="o", markerfacecolor="black", markersize=2, linestyle="none"
                ),
            )
            for k in range(j, min(j + MAX_ITEMS, len(values))):
                ax.text(
                    x=k - j + 1,
                    y=np.max(values[k]) * 1.01,
                    s=len(values[k]),
                    horizontalalignment="center",
                    size="x-small",
                    color="r",
                    weight="semibold",
                )
    plt.show()


def log_time_plotter(epoch_times: List[float]):
    plt.plot(epoch_times)
    plt.show()


if __name__ == "__main__":
    filename = "/Users/yikaizhu/alicode/models-perf/deepspeed_baichuan_exp/results/baichuan_13B_zero3/55n_915_comm_log.txt"
    filename = "/Users/yikaizhu/Desktop/AIBC_clean.txt"
    comm_log = parse_ds_comm_log(filename)
    comm_info = comm_log.analyze()
    if "train" in comm_info:
        log_boxplot(comm_info["train"]["detailed_comm_type_info"])
    else:
        log_boxplot(comm_info["init"]["detailed_comm_type_info"])
