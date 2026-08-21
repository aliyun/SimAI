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

import os,math
import pickle
import csv
import dataclasses
import logging
import numpy as np
from typing import Union, Dict, List
from utils.utils import CommType, CommGroup
from log_analyzer.utils import convert_size_to_msg, calc_bw_log
import copy

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class LogItem:
    comm_type: CommType = dataclasses.field(default=None)
    comm_group: CommGroup = dataclasses.field(default=None)
    comm_group_size: int = dataclasses.field(default=None)
    msg_size: float = dataclasses.field(default=0)

    stage: str = dataclasses.field(default="")
    dst: int = dataclasses.field(default=None)
    src: int = dataclasses.field(default=None)
    additional: str = dataclasses.field(default="")

    _elapsed_time: float = dataclasses.field(default=None)
    algbw: float = dataclasses.field(default=None)
    busbw: float = dataclasses.field(default=None)
    count: float = dataclasses.field(default=1)
    ranks: list = dataclasses.field(default=None)

    @property
    def elapsed_time(self) -> float:
        return self._elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, elapsed_time):
        self._elapsed_time = elapsed_time
        self.algbw, self.busbw = calc_bw_log(
            self.comm_type, self.msg_size, elapsed_time, self.comm_group_size
        )

    def is_epoch_end(self):
        return self.comm_type == CommType.epoch_end

    def is_workload(self):
        return self.elapsed_time is None

    def view_as_ds_log(self):
        log_str = f"[RANK 0] comm op: {self.comm_type} | comm group: {self.comm_group}"
        log_str += " | time (ms): {:.2f}".format(self.elapsed_time)
        if self.comm_type == CommType.computation or self.additional == 'overlap':
            log_str += " | msg size: " + '0'
            log_str += " | algbw (GB): " + '0'
            log_str += " | busbw (GB): " + '0'
        else:
            log_str += " | msg size: " + convert_size_to_msg(self.msg_size)
            log_str += " | algbw (GB): {:.2f} ".format(self.algbw)
            log_str += " | busbw (GB): {:.2f} ".format(self.busbw)
        return log_str

    def csv_header(self):
        return ",".join([k for k in self.__dict__.keys()])

    def view_as_csv_line(self):
        parts = []
        for k in self.__dict__.keys():
            v = getattr(self, k)
            if k == 'ranks':
                if v is not None:
                    parts.append('"' + ','.join(str(r) for r in v) + '"')
                else:
                    parts.append('')
            else:
                parts.append(str(v))
        return ','.join(parts)

    def __str__(self):
        if self.is_workload():
            return "None"
        return "None"


def _print_stage_log(stage_name: str, stage_count: int, comm_type_info: Dict, primary_key: List[str], agg_key: List[str], performance_key: List[str], busbw_key: List[str]):
    header = f"{'Comm_Type':<15} {'Comm_Group':<12} {'Message_Size':<12} {'Count':<12} {'Avg_Elapsed_Time ± Std ':<24} {'Avg_BusBw ± Std':<24}\n"
    separator = "-" * len(header) + "\n"
    log_str = separator + header + separator

    for pkey in sorted(comm_type_info.keys()):
        row_str = ""
        values = {}
        for i, pkey_name in enumerate(primary_key):
            value = pkey[i] if pkey_name != "msg_size" else convert_size_to_msg(pkey[i])
            values[pkey_name] = value
        for key in agg_key:
            value = comm_type_info[pkey][key]
            value = convert_size_to_msg(value) if key == "msg_size" else f"{value:.2f}"
            values[key] = value
        for key in performance_key:
            performance_value_list = sorted(comm_type_info[pkey][key])
            values[f'avg_{key}'] = f"{np.mean(performance_value_list):.2f}±{np.std(performance_value_list):.2f}"
            values[f'min_{key}'] = f"{performance_value_list[0]:.2f}"
            values[f'max_{key}'] = f"{performance_value_list[-1]:.2f}"
        
        for key in busbw_key:
            busbw_value_list = sorted(comm_type_info[pkey][key])
            values[f'avg_{key}'] = f"{np.mean(busbw_value_list):.2f}±{np.std(busbw_value_list):.2f}"

        row_str += f"{values['comm_type']:<15} {values['comm_group']:<12} {values['msg_size']:<12} {values['count']:<16} {values['avg__elapsed_time']:<24} {values['avg_busbw']:<18}\n"
        log_str += row_str

    return log_str


def _analyze_stage_log(comm_log: List[Dict], stage: str, comm_info: Dict[str, Dict]):
    def __update_info(
        info_dict,
        log,
        primary_key: List[str],
        agg_key: List[str],
        performance_key: List[str],
        busbw_key: List[str],
    ):
        primary_key = tuple(log[key] for key in primary_key)
        if primary_key not in info_dict:
            info_dict[primary_key] = dict((key, 0) for key in agg_key)
            info_dict[primary_key].update(dict((key, []) for key in performance_key))
            info_dict[primary_key].update(dict((key, []) for key in busbw_key))
        for key in agg_key:
            info_dict[primary_key][key] += log[key]
        for key in performance_key:
            info_dict[primary_key][key].append(log[key])
        for key in busbw_key:
            info_dict[primary_key][key].append(log[key])

    if stage not in comm_info:
        comm_info[stage] = {
            "count": 0,
            "comm_type_info": {},
            "detailed_comm_type_info": {},
        }
    comm_info[stage]["count"] += 1
    # key: comm_type, value: count, time_ms
    comm_type_info = comm_info[stage]["comm_type_info"]
    # key: comm_type, msg_size, value: count, time_ms
    detailed_comm_type_info = comm_info[stage]["detailed_comm_type_info"]
    for log in comm_log:
        if log.comm_type != CommType.computation:
            __update_info(
                comm_type_info,
                log.__dict__,
                ["comm_type", "comm_group"],
                ["count", "msg_size"],
                ["_elapsed_time"],
                ["busbw"],
            )
            __update_info(
                detailed_comm_type_info,
                log.__dict__,
                ["comm_type", "comm_group", "msg_size"],
                ["count"],
                ["_elapsed_time"],
                ["busbw"],
            )


class Log:
    def __init__(self) -> None:
        self.comm_logs = []
        self.comm_log_each_epoch = [[]]
        self.epoch_times = []

    def add_comm_log(self, comm_log: LogItem):
        if (
            comm_log.is_epoch_end()
            and len(self.comm_logs) > 0
            and not self.comm_logs[-1].is_epoch_end()
        ):
            self.comm_logs.append(comm_log)
            self.comm_log_each_epoch.append([])
            self.epoch_times.append(comm_log.elapsed_time)
            return
        self.comm_logs.append(comm_log)
        self.comm_log_each_epoch[-1].append(comm_log)

    def analyze(self, print_fn=print):
        comm_info: Dict[str, Dict] = {}
        _analyze_stage_log(self.comm_log_each_epoch[0], "init", comm_info)
        for e_log in self.comm_log_each_epoch[1:]:
            _analyze_stage_log(e_log, "train", comm_info)
        for stage in comm_info.keys():
            if stage != "init":   
                stage_count = comm_info[stage]["count"]
                comm_type_info = comm_info[stage]["comm_type_info"]
                detailed_comm_type_info = comm_info[stage]["detailed_comm_type_info"]

                log_str = _print_stage_log(stage, stage_count, detailed_comm_type_info, ["comm_type", "comm_group", "msg_size"], ["count"], ["_elapsed_time"], ["busbw"])
                print_fn(f"\n\tDetailed comm info for AICB {stage} stage\n{log_str}")
        return comm_info

    def dump(self, filename):
        default_comm_folder_path = "results/comm_logs/"
        if not os.path.exists(default_comm_folder_path):
            os.makedirs(default_comm_folder_path, exist_ok=True)
        if "." in filename:
            filename = filename.split(".")[0]
        filename = os.path.join("results/comm_logs/", filename)
        csv_filename = filename + "_log.csv"
        with open(csv_filename, "w") as f:
            f.write(self.comm_logs[0].csv_header() + "\n")
            for log_item in self.comm_logs:
                log_item_write = copy.deepcopy(log_item)
                if(log_item_write.comm_type == CommType.computation):
                    msg_size_str = "("+' '.join(str(shape).replace(',', '') for shape in log_item_write.msg_size)+")"
                    log_item_write.msg_size = msg_size_str
                f.write(log_item_write.view_as_csv_line() + "\n")
                del log_item_write
        return csv_filename

    @staticmethod
    def load(filename):
        filename = filename.split(".")
        filename[-1] = "pkl"
        filename = ".".join(filename)
        return pickle.load(open(filename, "rb"))

    def _get_elapsed_time(self):
        return self.epoch_times

    def analyze_time(self, print_fn=print):
        self.epoch_times.pop(0)
        max_val = max(self.epoch_times)
        min_val = min(self.epoch_times)
        mean_val = sum(self.epoch_times) / len(self.epoch_times)

        variance = sum((x - mean_val) ** 2 for x in self.epoch_times) / len(
            self.epoch_times
        )
        variance = math.sqrt(variance)

        sorted_list = sorted(self.epoch_times)
        p90_val = sorted_list[int(len(sorted_list) * 0.9)]
        p99_val = sorted_list[int(len(sorted_list) * 0.99)]
        header = f"{'Init time':<18} {'Max iteration time':<20} {'Min iteration time':<20} {'Avg iteration time':<20} {'P90 iteration time ':<20} {'Iteration time Std ':<20}\n"
        separator = "-" * len(header) + "\n"
        log_str = separator + header + separator 
        iteration_result = f"{self.epoch_times[0]:<18.2f} {max_val:<20.2f} {min_val:<20.2f} {mean_val:<20.2f} {p90_val:<20.2f} {variance:<20.2f}\n"
        log_str += iteration_result
        print_fn(f"\n\tDetailed info for AICB iteration time\n{log_str}")


class Workload:
    def __init__(self) -> None:
        self.workload = []

    def append(self, log_item: Union[LogItem, Dict]):
        if isinstance(log_item, LogItem):
            self.workload.append(log_item)
            return
        if "stage" not in log_item:
            log_item["stage"] = log_item["operation"] if "operation" in log_item else ""
        if "comm_group" not in log_item:
            assert (
                log_item["comm_type"] == CommType.computation
            ), "comm_group is required for non-computation comm_type"
            log_item["comm_group"] = CommGroup.all
        self.workload.append(
            LogItem(
                comm_type=log_item["comm_type"],
                comm_group=log_item["comm_group"],
                comm_group_size=log_item["comm_group_size"],
                msg_size=log_item["msg_size"],
                stage=log_item["stage"],
                src=log_item.get("src", None),
                dst=log_item.get("dst", None),
                additional=log_item.get("additional", None),
            )
        )

    def extend(self, new_workload):
        self.workload.extend(new_workload.workload)

    def dump(self, filename):
        folder_path = os.path.dirname(filename)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        default_folder_path = "results/mocked_workload/"
        if not os.path.exists(default_folder_path):
            os.makedirs(default_folder_path, exist_ok=True)
        if "." in filename:
            filename = os.path.basename(filename).split(".")[0]
        filename = os.path.join("results/mocked_workload/", filename)
        csv_filename = filename + "_workload.csv"

        rank_gen = getattr(self, 'rank_generator', None)

        if rank_gen is not None:
            from utils.rank_mapper import get_rank_list_for_comm_group, build_rank_mapping_table
            for log_item in self.workload:
                if log_item.ranks is not None:
                    continue
                if log_item.comm_group is None:
                    continue
                log_item.ranks = get_rank_list_for_comm_group(
                    rank_gen, log_item.comm_group, log_item.comm_group_size
                )
                if log_item.comm_group_size is None and log_item.ranks:
                    log_item.comm_group_size = len(log_item.ranks)
                logger.warning(
                    "Rank auto-populated at dump time for LogItem stage=%s, comm_group=%s.",
                    log_item.stage, log_item.comm_group.value,
                )

            mapping_rows = build_rank_mapping_table(rank_gen)
            sidecar_filename = filename + "_rank_mapping.csv"
            with open(sidecar_filename, "w") as sf:
                sf.write("group,size,rank_groups\n")
                for row in mapping_rows:
                    rank_groups_quoted = '"' + row["rank_groups"] + '"'
                    sf.write(f"{row['group']},{row['size']},{rank_groups_quoted}\n")
            print(f"Rank mapping file generated:{sidecar_filename}")
        else:
            logger.warning("No rank_generator available; skipping rank population and sidecar.")

        with open(csv_filename, "w") as f:
            f.write(self.workload[0].csv_header() + "\n")
            for log_item in self.workload:
                log_item_write = copy.deepcopy(log_item)
                if(log_item_write.comm_type == CommType.computation):
                    msg_size_str = "("+' '.join(str(shape).replace(',', '') for shape in log_item_write.msg_size)+")"
                    log_item_write.msg_size = msg_size_str
                f.write(log_item_write.view_as_csv_line() + "\n")
                del log_item_write
        print(f"Workload file generated:{csv_filename}")
        
    @staticmethod
    def load(filename):
        filename = filename.split(".")
        filename[-1] = "pkl"
        filename = ".".join(filename)
        workload, args = pickle.load(open(filename, "rb"))
        return workload, args
