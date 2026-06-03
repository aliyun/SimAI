#!/usr/bin/python3
""" 
Usage [{filename}]:
1. Change IMAGE_NAME from DUMMY to your real image name 
2. Change IPLIST from DUMMY to your real /path/to/iplist (absolute path)
3. Change AICB_DIR from DUMMY to your real /path/to/aicb (absolute path)
4. Change the settings in run_suites.py to select the workload you want
5. Copy iplist and aicb to all participating servers at /path/to/iplist and /path/to/aicb, e.g., using `pscp` command like `pscp.pssh -h iplist iplist /path/to/iplist` and `pscp.pssh -h iplist -r aicb /path/to/aicb`
6. Run simulation on all participating servers, e.g., using `pssh` command like `pssh -i -h /path/to/iplist -o out -e err -t 0 "cd /path/to/aicb && python scripts/run_in_cluster.py"`
"""

import subprocess
import os
import re
import sys

filename = os.path.basename(__file__)
__doc__ = __doc__.format(filename=filename)


def get_local_ip():
    output = os.popen("ifconfig").read().strip()
    pattern = r"inet (\d+.\d+.\d+.\d+) "
    return re.findall(pattern, output)


def get_world_id_list(filename):
    with open(filename, "r") as f:
        return f.read().strip().split("\n")


def get_docker_env_rank(filename):
    ip_list = get_world_id_list(filename)
    local_ip = get_local_ip()
    for ip in local_ip:
        if ip in ip_list:
            return len(ip_list), ip_list.index(ip), ip_list[0], 12345
    return -1, -1, -1, -1


IPLIST = "DUMMY_IPLIST"  # Change it to /path/to/iplist, e.g., /root/iplist
AICB_DIR = "DUMMY_AICB_DIR" # Change it to /path/to/aicb, e.g., /root/aicb
IMAGE_NAME = "DUMMY_IMAGE_NAME"  # Change it to your docker image name, e.g., nvcr.io/nvidia/pytorch:xx.xx-py3

if IPLIST == "DUMMY_IPLIST" or AICB_DIR == "DUMMY_AICB_DIR" or IMAGE_NAME == "DUMMY_IMAGE_NAME":
    sys.stderr.write(__doc__)
    sys.exit(1)

WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT = get_docker_env_rank(IPLIST)
AICB_DIR_base = os.path.basename(AICB_DIR)
command = f"""docker run --name aicb_test --gpus all --privileged \
--ulimit memlock=-1 --ulimit stack=67108864 \
--init -i --shm-size=4g --network=host --rm \
-e WORLD_SIZE={WORLD_SIZE} \
-e RANK={RANK} \
-e MASTER_ADDR={MASTER_ADDR} \
-e MASTER_PORT={MASTER_PORT} \
-v {AICB_DIR}:/workspace/{AICB_DIR_base} \
{IMAGE_NAME} /bin/sh -c 'cd /workspace/{AICB_DIR_base} && pwd && python run_suites.py'
""" # Change the settings in run_suites.py to select the workload you want

ret = subprocess.run(command, shell=True)
print(ret)
