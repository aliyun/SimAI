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

# /usr/bin/python3
from utils.utils import CommType, CommGroup
from log_analyzer.utils import convert_msg_to_size, convert_size_to_msg
from utils.benchmark_logger import BenchLogger
from log_analyzer.log import LogItem, Log

COMM_OP = "comm op"
CALLER_FUNC = "Caller Func"
TIME_MS = "time (ms)"
MSG_SIZE = "msg size"
LOG_STARTER = "[rank 0]"
WORLD_SIZE = 16
TP_SIZE = 4
DP_SIZE = 4
# LOG_STARTER = "[INFO] "


def clean_s(s):
    return s.strip("[]\n\t ")


def string2comm_type(s):
    if "all_gather" in s:
        return CommType.all_gather
    if "reduce_scatter" in s:
        return CommType.reduce_scatter
    if "all_reduce" in s:
        return CommType.all_reduce
    if "broadcast" in s:
        return CommType.broadcast
    if "barrier" in s:
        return CommType.barrier
    if "reduce" in s:
        return CommType.reduce
    print(f"WARNING cannot convert {s} to CommType")
    return CommType.epoch_end


def parse_ds_log_item(line):
    index = line.lower().find(LOG_STARTER)
    if index == -1:
        return None
    item_list = line[index + len(LOG_STARTER) :].split("|")
    item = {}
    for raw_item in item_list:
        if "epoch" in raw_item:
            split_text = raw_item.split()
            numbers = [word for word in split_text if word.isdigit()]
            item["epoch_num"] = int(numbers[0])
            continue
        if "micro_step" in raw_item:
            split_text = raw_item.split()
            numbers = [word for word in split_text if word.replace(".", "").isdigit()]
            item["iter_time"] = float(numbers[0])
            continue
        if ":" not in raw_item:
            continue
        key, value = raw_item.split(":")
        key, value = clean_s(key), clean_s(value)
        if key == COMM_OP:
            item["comm_type"] = string2comm_type(value)
        elif key == MSG_SIZE or MSG_SIZE in key:
            item["msg_size"] = convert_msg_to_size(value)
        elif key == CALLER_FUNC:
            item["stage"] = value
        elif key == TIME_MS or TIME_MS in key:
            item["elapsed_time"] = float(value)
        if key == "group":
            group = eval(value)
            if len(group) == WORLD_SIZE:
                item["group"] = CommGroup.all
            elif len(group) == TP_SIZE:
                item["group"] = CommGroup.tp_group
            elif len(group) == DP_SIZE:
                item["group"] = CommGroup.dp_group
        elif "algbw" in key:
            item["algbw"] = float(value)
        elif "busbw" in key:
            item["busbw"] = float(value)
        else:
            try:
                item[key] = float(value)
            except:
                item[key] = value
    return item


def parse_ds_comm_log(filename):
    comm_log = Log()
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            if "After initializing ZeRO optimizer" in line:
                comm_log.add_comm_log(LogItem(comm_type=CommType.epoch_end))
                continue
            elif "microstep" in line:
                comm_log.add_comm_log(LogItem(comm_type=CommType.epoch_end))
                continue
            log = parse_ds_log_item(line)
            if log is None:
                continue
            if "comm_type" in log:
                log_item = LogItem(
                    comm_type=log["comm_type"],
                    comm_group=log.get("group", CommGroup.dp_group),
                    msg_size=log["msg_size"],
                )
                log_item._elapsed_time = log.get("elapsed_time", -1)
                log_item.algbw, log_item.busbw = log.get("algbw", -1), log.get(
                    "busbw", -1
                )
                comm_log.add_comm_log(log_item)
    return comm_log


if __name__ == "__main__":
    import sys

    filename = sys.argv[1]
    comm_log = parse_ds_comm_log(filename)
    comm_log.analyze()
