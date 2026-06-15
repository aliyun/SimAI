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
import sys
import subprocess
import os
import configparser
import argparse

running_command = {}
default_config = {
    "deepspeed": {
        "llama7b_zero2": 0,
        "llama65b_zero3": 0,
    },
    "megatron": {
        "llama_7B": 0,
        "gpt_13B_sp": 0,
        "gpt_175B_tp":0,
        "gpt_175B": 0,
        "gpt_22B": 0,
        "llama_405B": 0,
        "Mixtral_8*7B": 0,
    },
    "aiob" : {  #aicb workload suites with computation
        "llama_7B_aiob": 0,
        "gpt_13B_sp_aiob": 0,
        "gpt_175B_aiob": 0,
        "gpt_22B_aiob": 0,
        "gpt_175B_tp_aiob": 0,
        "llama_405B_aiob": 0,
        "Mixtral_8*7B_aiob": 0, 
        "llama7B_zero2_aiob": 0,
        "llama65B_zero3_aiob": 0,
    },
    "coll_comm_check": {"all_reduce": 0, "all_gather": 0, "muti_all_reduce": 0},
}


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output directory", default="./results")


def read_config(config):
    ds_conf = config["deepspeed"]
    megatron_conf = config["megatron"]
    cc_conf = config["coll_comm_check"]
    aiob_conf = config["aiob"]
    if int(ds_conf["llama7b_zero2"]):
        running_command["deepspeed2_llama13b"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 2 -m 13 --epoch_num 10 "
        )
    if int(ds_conf["llama65b_zero3"]):
        running_command["deepspeed3_llama65b"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 3 -m 65 --epoch_num 10  \
          --reduce_bucket_size 1000000000 --allgather_bucket_size 500000000 \
          --param_persistence_threshold 1000000"
        )
    if int(megatron_conf["llama_7B"]):
        running_command["megatron_llama7B"] = (
            f"bash scripts/megatron_gpt.sh -m 7 --tensor_model_parallel_size 1 --epoch_num 10  --seq_length 4096"
        )
    if int(megatron_conf["gpt_13B_sp"]):
        running_command["megatron_gpt13b_sp"] = (
            f"bash scripts/megatron_gpt.sh -m 13 --tensor_model_parallel_size 2 --epoch_num 10 --sp"
        )
    if int(megatron_conf["gpt_175B"]):
        running_command["megatron_gpt175B"] = (
            f"bash scripts/megatron_gpt.sh -m 175 --tensor_model_parallel_size 8 --epoch_num 10 --pipeline_model_parallel 2 --sp"
        )
    if int(megatron_conf["gpt_175B_tp"]):
        running_command["megatron_gpt175B_tp"] = (
            f"bash scripts/megatron_gpt.sh -m 175 --tensor_model_parallel_size 8 --epoch_num 10 --pipeline_model_parallel 2"
        )
    if int(megatron_conf["gpt_22B"]):
        running_command["megatron_gpt_22B"] = (
            f"bash scripts/megatron_gpt.sh -m 22 --tensor_model_parallel_size 4 --epoch_num 10 --sp"
        )
    if int(megatron_conf["Mixtral_8*7B"]):
        running_command["megatron_moe"] = (
            f"bash scripts/megatron_gpt.sh -m moe --tensor_model_parallel_size 2 --epoch_num 10 --sp --ep 4 --num_experts 16 --topk 4 "
        )
    if int(megatron_conf["llama_405B"]):
        running_command["megatron_llama_405B"] = (
            f"bash scripts/megatron_gpt.sh -m 405 --tensor_model_parallel_size 8 --epoch_num 10 --sp --seq_length 8192"
        )
    if int(aiob_conf["llama_7B_aiob"]):
        running_command["megatron_llama7b_aiob"] = (
            f"bash scripts/megatron_gpt.sh -m 7 --epoch_num 10 --aiob_enable "
        )
    if int(aiob_conf["gpt_13B_sp_aiob"]):
        running_command["megatron_gpt13b_sp_aiob"] = (
            f"bash scripts/megatron_gpt.sh -m 13 --tensor_model_parallel_size 4 --epoch_num 10 --aiob_enable --sp"
        )
    if int(aiob_conf["gpt_175B_aiob"]):
        running_command["megatron_gpt175B_aiob"] = (
            f"bash scripts/megatron_gpt.sh -m 175 --tensor_model_parallel_size 8 --epoch_num 10 --aiob_enable --pipeline_model_parallel 2 --sp"
        )
    if int(aiob_conf["gpt_175B_tp_aiob"]):
        running_command["megatron_gpt175B_tp_aiob"] = (
            f"bash scripts/megatron_gpt.sh -m 175 --tensor_model_parallel_size 8 --epoch_num 10 --aiob_enable --pipeline_model_parallel 2 "
        )
    if int(aiob_conf["gpt_22B_aiob"]):
        running_command["megatron_gpt_22B_aiob"] = (
            f"bash scripts/megatron_gpt.sh -m 22 --tensor_model_parallel_size 4 --epoch_num 10 --aiob_enable --sp"
        )
    if int(aiob_conf["Mixtral_8*7B_aiob"]):
        running_command["megatron_moe_aiob"] = (
            f"bash scripts/megatron_gpt.sh -m moe --tensor_model_parallel_size 2 --epoch_num 10 --sp --ep 4 --num_experts 16 --topk 4 --aiob_enable "
        )
    if int(aiob_conf["llama_405B_aiob"]):
        running_command["megatron_llama_405B"] = (
            f"bash scripts/megatron_gpt.sh -m 405 --tensor_model_parallel_size 8 --epoch_num 10 --sp --seq_length 8192 --aiob_enable "
        )
    if int(aiob_conf["llama7B_zero2_aiob"]):
        running_command["deepspeed2_llama7b_aiob"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 2 -m 7 --epoch_num 10 --aiob_enable "
        )
    if int(aiob_conf["llama65B_zero3_aiob"]):
        running_command["deepspeed3_llama65b_aiob"] = (
            f"bash scripts/deepspeed_llama.sh --zero_stage 3 -m 65 --epoch_num 10  \
          --reduce_bucket_size 1000000000 --allgather_bucket_size 500000000 \
          --param_persistence_threshold 1000000 --aiob_enable"
        )
    if int(cc_conf["all_reduce"]):
        running_command["all_reduce_check"] = (
            f"bash scripts/coll_comm_check.sh --iter_num 100 --test_comm all_reduce --model_name all_reduce"
        )
    if int(cc_conf["all_gather"]):
        running_command["all_gather_check"] = (
            f"bash scripts/coll_comm_check.sh --iter_num 100 --test_comm all_gather --model_name all_gather"
        )
    if int(cc_conf["muti_all_reduce"]):
        running_command["muti_all_reduce_check"] = (
            f"bash scripts/coll_comm_check.sh --iter_num 100 --test_comm all_reduce --model_name muti_all_reduce --muti_all_reduce_enable 1"
        )
    


if __name__ == "__main__":
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    read_config(config=default_config)
    result = {}
    print(running_command)
    for name, command in running_command.items():
        result_dir = "./results/log/"
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        output_file = f"./results/log/{name}.txt"
        
        command += f" 2>&1 | tee {output_file}"
        print(name)
        ret = subprocess.run(command, shell=True, text=True)

        # if ret.returncode != 0:
        #     print(f"ERROR when running {name}: {command}")
        #     print(
        #         f"return state is {ret.returncode}, got err{ret.stderr}, get output{ret.stdout}"
        #     )
        #     exit(-1)
        # command_out = ret.stdout
        # print(command_out)
