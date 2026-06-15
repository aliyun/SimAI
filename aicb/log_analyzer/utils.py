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

import math
from utils.utils import CommGroup, CommType, get_args


def convert_size_to_msg(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def convert_msg_to_size(msg):
    if msg == "0B":
        return 0
    try:
        num, name = msg.split(" ")
    except:
        print(f"cannot convert msg into int")
        return 0
    num, name = float(num), name.strip()
    size_name = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    if name not in size_name:
        return None
    p = math.pow(1024, size_name.index(name))
    return num * p


def calc_bw_log(comm_type: CommType, size, duration,group_size):  # size: Bytes; duration: ms
    n = group_size if group_size else 1
    duration /= 1000
    if comm_type in [CommType.all_gather, CommType.reduce_scatter]:
        # size *= n
        tput = size / duration
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_type == CommType.all_reduce:
        tput = size / duration
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_type in [CommType.barrier, CommType.computation]:
        return 0, 0
    else:  # [CommType.broadcast, CommType.reduce, "gather", "scatter", "isend", "irecv"]
        tput = size / duration
        busbw = tput
    tput /= 1024*1024*1024
    busbw /= 1024*1024*1024
    tput = round(tput, 2)
    busbw = round(busbw, 2)
    return tput, busbw
