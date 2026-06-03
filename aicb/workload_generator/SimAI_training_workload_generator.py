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

from workload_generator.mocked_model.training import MockedDeepSeek
import workload_generator.mocked_model.training.MockedDeepspeed
from workload_generator.mocked_model.training.MockedMegatron import *
from workload_generator.mocked_model.training.MockedDeepSeek import *
from workload_generator.mocked_model.MockedModel import MockedParam, MockedModel
from utils.utils import CommType, get_params, get_comp_out, extract_averages
import os
from typing import List, Tuple
from collections import deque
import dataclasses
from enum import Enum

try:
    import torch
except ImportError as e:
    torch = None
    print("Failed to import 'torch'.")
import math
import re





@dataclasses.dataclass
class Work_Item:
    name: str = dataclasses.field(default="none")
    placeholder: int = dataclasses.field(default=-1)
    forward_compute_time: int = dataclasses.field(default=0)
    forward_comm: str = dataclasses.field(default="NONE")
    forward_comm_size: int = dataclasses.field(default=0)
    backward_compute_time: int = dataclasses.field(default=0)
    backward_comm: str = dataclasses.field(default="NONE")
    backward_comm_size: int = dataclasses.field(default=0)
    dp_compute_time: int = dataclasses.field(default=0)
    dp_comm: str = dataclasses.field(default="NONE")
    dp_comm_size: int = dataclasses.field(default=0)
    process_time: int = dataclasses.field(default=100)



def _get_aiob_compute_time(compute_cache, forward_or_backward, stage, dowarn=True):
    compute_time_map = compute_cache

    # if compute time with exact layer name exist, use it without prefix
    if stage in compute_time_map.keys():
        return compute_time_map[stage]

    if stage == "grad":
        prefix = stage + "_" + forward_or_backward
    elif stage == "embedding":
        prefix = "Emb"
    elif stage == "final":
        prefix = "attention" + "_" + forward_or_backward
    else:
        prefix = stage + "_" + forward_or_backward

    for key, value in compute_time_map.items():
        if prefix == key:

            compute_time = compute_time_map.get(key)
            return compute_time
    # just so it doesn't spam warning when trying to get per-layer comp time
    if dowarn:
        print("[warn] can't match any stage", stage)
    return 1


class LayerInfo:
    def __init__(self, layer_id, layer_name, param_count):
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.param_count = param_count


class SIMAI_workload:
    def __init__(self, model, args, compute_cache=None):
        self.model = model
        self.args = args
        self.compute_cache = compute_cache
        self.workload = []
        self.seq_len = args.seq_length
        self.tp = args.tensor_model_parallel_size
        self.mbs = args.micro_batch
        if args.moe_enable:
            self.expert_model_parallel_size = args.expert_model_parallel_size
            self.num_experts = args.num_experts
            self.topk = args.moe_router_topk

    def get_model_details(self):
        layers = []
        visited = set()

        def traverse_model(model):
            if id(model) in visited:
                return
            visited.add(id(model))

            if self.args.enable_sequence_parallel:
                if (
                    isinstance(model, MegatronColumnLinear)
                    or isinstance(model, MegatronRowLinear)
                    or isinstance(model, MegatronEmbedding)
                    or isinstance(model, FusedLayernorm)
                    or isinstance(model, DeepSeekLinear)
                ):
                    params = model.parameters()
                    param_count = sum(p.numel() for p in params)
                    layers.append(LayerInfo(model.layer_id, model.name, param_count))
                if isinstance(model, MOEMLP) or isinstance(model, DeepSeekMoE):
                    moe_params = model.parameters()
                    moe_param_count = sum(p.numel() for p in moe_params)
                    layers.append(LayerInfo(model.layer_id, model.name, moe_param_count))

            else:
                if (
                    isinstance(model, MegatronAttention)
                    or isinstance(model, MegatronMlp)
                    or isinstance(model, MegatronEmbedding)
                    or isinstance(model, DeepSeekMLA)
                    or isinstance(model, DeepSeekMoE)
                ):
                    params = model.parameters()
                    param_count = sum(p.numel() for p in params)
                    layers.append(LayerInfo(model.layer_id, model.name, param_count))

            for child in model.child_modules():
                traverse_model(child)

        traverse_model(model)

        return layers

    def _get_total_params(self):
        total_params = 0
        moe_param_count = 0
        layers = self.get_model_details()
        for layer in layers:
            total_params += layer.param_count
            if "moe" in layer.layer_name:
                moe_param_count += layer.param_count

        return total_params, moe_param_count

    def workload_generate_aiob(self):
        # args.world_size --> total gpus number
        self.ga_num = self.args.global_batch // (self.args.micro_batch * self.args.dp_num)
        if self.ga_num < 1:
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"
            )
        default_compute_time = 1
        compute_time = 0
        tp_comm_size = (
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size
        )
        layers = self.get_model_details()
        total_params, moe_param_count = self._get_total_params()
        # self.workload.append(Work_Item(name="norm", forward_compute_time=0,
        #                         forward_comm = "BROADCAST", forward_comm_size= 8*self.args.micro_batch*self.args.seq_length,
        #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
        #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
        #                         ))
        forward_compute_time = _get_aiob_compute_time(
            self.compute_cache, "forward", "grad"
        )
        backward_compute_time = _get_aiob_compute_time(
            self.compute_cache, "backward", "grad"
        )
        self.workload.append(
            Work_Item(
                name="grad_gather",
                forward_compute_time=default_compute_time,
                forward_comm="NONE",
                forward_comm_size=0,
                backward_compute_time=default_compute_time,
                backward_comm="NONE",
                backward_comm_size=0,
                dp_compute_time=default_compute_time,
                dp_comm="ALLGATHER",
                dp_comm_size=2 * (total_params-moe_param_count),
            )
        )
        self.workload.append(
            Work_Item(
                name="grad_param_comm",
                forward_compute_time=default_compute_time,
                forward_comm="NONE",
                forward_comm_size=0,
                backward_compute_time=default_compute_time,
                backward_comm="NONE",
                backward_comm_size=0,
                dp_compute_time=default_compute_time,
                dp_comm="REDUCESCATTER",
                dp_comm_size=4 * (total_params-moe_param_count),
            )
        )
        self.workload.append(
            Work_Item(
                name="grad_param_compute",
                forward_compute_time=default_compute_time,
                forward_comm="NONE",
                forward_comm_size=0,
                backward_compute_time=forward_compute_time + backward_compute_time,
                backward_comm="NONE",
                backward_comm_size=0,
                dp_compute_time=default_compute_time,
                dp_comm="NONE",
                dp_comm_size=0,
            )
        )

        if not self.args.enable_sequence_parallel:
            self.workload.append(
                Work_Item(
                    name="layernorm",
                    forward_compute_time=default_compute_time,
                    forward_comm="NONE",
                    forward_comm_size=0,
                    backward_compute_time=default_compute_time,
                    backward_comm="ALLREDUCE",
                    backward_comm_size=2 * total_params,
                    dp_compute_time=default_compute_time,
                    dp_comm="NONE",
                    dp_comm_size=0,
                )
            )
        if args.tensor_model_parallel_size == 1 :
            emd_backward_comm = "NONE"
        else:
            emd_backward_comm = "ALLREDUCE"
        self.workload.append(
            Work_Item(
                name="embedding_grads",
                forward_compute_time=default_compute_time,
                forward_comm="NONE",
                forward_comm_size=0,
                backward_compute_time=default_compute_time,
                backward_comm=emd_backward_comm,
                backward_comm_size=tp_comm_size,
                dp_compute_time=default_compute_time,
                dp_comm="NONE",
                dp_comm_size=0,
            )
        )
        if self.args.expert_model_parallel_size != self.args.dp_num:
            self.workload.append(Work_Item(name="moe_grad_norm1", forward_compute_time=default_compute_time,
                                    forward_comm = "NONE", forward_comm_size= 0,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm="ALLGATHER_DP_EP", dp_comm_size=2*moe_param_count
                                    ))
            self.workload.append(Work_Item(name="moe_grad_norm2", forward_compute_time=default_compute_time,
                                    forward_comm = "NONE", forward_comm_size= 0,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm="REDUCESCATTER_DP_EP", dp_comm_size=4*moe_param_count
                                    ))
        for _ in range(self.ga_num):
            for layer in layers:
                name = layer.layer_name
                forward_comm = backward_comm = backward_comm_2 = "NONE"
                forward_comm_size = tp_comm_size
                emb_comm_size = tp_comm_size
                backward_comm_size = 0
                dp_comm = "NONE"
                dp_comm_size = 0

                # try get layer specific compute time
                # e.g. from AiobDeepSeek.DeepSeekMLA's compute times
                layer_comp_time = _get_aiob_compute_time(
                    self.compute_cache, "", name, False
                )
                # _get_aiob_compute_time return 1 in case no compute time found
                if layer_comp_time == 1:
                    layer_comp_time = None

                if self.args.enable_sequence_parallel:
                    if "embedding" in name:
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"
                            backward_comm = "NONE"
                        else:
                            forward_comm = "ALLREDUCE"
                            backward_comm = "NONE"
                        emb_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "", "embedding"
                        )
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=emb_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=emb_comm_size ,
                                backward_compute_time=default_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
                    if "attention_linear" in name:
                        # for non-shareded linear in attention block

                        # similar to row linear but without comms
                        if layer_comp_time != None:
                            forward_compute_time  = layer_comp_time
                            backward_compute_time = layer_comp_time
                        else:
                            forward_compute_time=_get_aiob_compute_time(
                                self.compute_cache, "forward", name.split("_")[0]
                            )
                            backward_compute_time = _get_aiob_compute_time(
                                self.compute_cache, "backward", name.split("_")[0]
                            )
                        if self.args.recompute_activations:
                            forward_compute_time *= 2
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=forward_compute_time,
                                forward_comm="NONE",
                                forward_comm_size=0,
                                backward_compute_time=backward_compute_time,
                                backward_comm="NONE",
                                backward_comm_size=0,#sp overlap allgather
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
                    if "row" in name:
                        
                        if layer_comp_time != None:
                            forward_compute_time  = layer_comp_time
                            backward_compute_time = layer_comp_time
                        else:
                            forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                            )
                            backward_compute_time = _get_aiob_compute_time(
                                self.compute_cache, "backward", name.split("_")[0]
                            )

                        if self.args.recompute_activations and 'attention' in name:
                            forward_compute_time *= 2
                        forward_compute_time = int(forward_compute_time / 2)
                        backward_compute_time = int(backward_compute_time / 2)
                        forward_comm_size_sp = tp_comm_size
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"
                            backward_comm = "NONE"
                        else:
                            forward_comm = "REDUCESCATTER"
                            backward_comm = "ALLGATHER"
                        self.workload.append(
                                Work_Item(
                                    name=name,
                                    forward_compute_time=forward_compute_time,
                                    forward_comm=forward_comm,
                                    forward_comm_size=forward_comm_size,
                                    backward_compute_time=backward_compute_time,
                                    backward_comm=backward_comm,
                                    backward_comm_size=forward_comm_size_sp,#sp overlap allgather
                                    dp_compute_time=backward_compute_time,
                                    dp_comm=dp_comm,
                                    dp_comm_size=dp_comm_size,
                                )
                            )

                    elif "column" in name:
                        if layer_comp_time != None:
                            forward_compute_time  = layer_comp_time
                            backward_compute_time = layer_comp_time
                        else:
                            forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                            )
                            backward_compute_time = _get_aiob_compute_time(
                                self.compute_cache, "backward", name.split("_")[0]
                            )

                        if self.args.recompute_activations and 'attention' in name:
                            forward_compute_time *= 2
                        forward_compute_time = int(forward_compute_time / 2)
                        backward_compute_time = int(backward_compute_time / 2)
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"
                            backward_comm = "NONE"
                            backward_comm_2 = "NONE"
                        else:
                            forward_comm = "ALLGATHER"
                            backward_comm = "REDUCESCATTER"
                            backward_comm_2 = "ALLGATHER"
                        self.workload.append(
                                Work_Item(
                                    name=name,
                                    forward_compute_time=forward_compute_time,
                                    forward_comm=forward_comm,
                                    forward_comm_size=forward_comm_size,
                                    backward_compute_time=backward_compute_time,
                                    backward_comm=backward_comm,
                                    backward_comm_size=backward_comm_size,
                                    dp_compute_time=backward_compute_time,
                                    dp_comm=dp_comm,
                                    dp_comm_size=dp_comm_size,
                                )
                            )
                    elif "moelayer" in name:
                        if layer_comp_time != None:
                            forward_compute_time  = layer_comp_time
                            backward_compute_time = layer_comp_time
                        else:
                            forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                            )
                            backward_compute_time = _get_aiob_compute_time(
                                self.compute_cache, "backward", name.split("_")[0]
                            )
                        
                        # MoE Communication is based on Megatron core_v0.13.0:https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0
                        forward_comm1 = "ALLGATHER" # for EP
                        forward_comm2 = "ALLTOALL_EP"
                        forward_comm3 = "ALLGATHER"
                        forward_comm4 = "REDUCESCATTER"
                        forward_comm5 = "ALLTOALL_EP"
                        if args.expert_model_parallel_size == 1:
                            forward_comm2 = "NONE"
                            forward_comm5 = "NONE"
                        if args.tensor_model_parallel_size == 1:
                            if args.expert_model_parallel_size == 1:
                                forward_comm1 = "NONE"
                            forward_comm3 = "NONE"
                            forward_comm4 = "NONE"

                        # if args.expert_model_parallel_size != 1:
                        ep_allgather_size = 2 * self.expert_model_parallel_size * self.num_experts * self.tp
                        fwd_ep_dispatch_size = tp_comm_size * self.topk // self.tp
                        bkwd_ep_dispatch_size = tp_comm_size * self.topk // self.tp
                        ep_combine_size = tp_comm_size * self.topk // self.tp

                        if self.args.frame == "DeepSeek":
                            # for DeepEP based on https://github.com/parthpower/DeepEP/commit/50aee15f592bc22142eb04b7d718296b19613ae9
                            # only fprop does the FP8
                            fwd_ep_dispatch_size = int(fwd_ep_dispatch_size * MockedDeepSeek.FP8_FACTOR)
                            # rest of the comm shapes are similar to megatron
                        # EP All gather
                        self.workload.append(Work_Item(name=name, forward_compute_time=forward_compute_time,
                                    forward_comm = forward_comm1, forward_comm_size=ep_allgather_size,
                                    backward_compute_time=backward_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        # EP dispatch
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm2, forward_comm_size=fwd_ep_dispatch_size,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm2, backward_comm_size=bkwd_ep_dispatch_size,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        # TP All reduce
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm3, forward_comm_size=tp_comm_size*self.topk,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm4, backward_comm_size=tp_comm_size*self.topk,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm4, forward_comm_size=tp_comm_size*self.topk,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm3, backward_comm_size=tp_comm_size*self.topk,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        # EP combine
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm5, forward_comm_size=ep_combine_size,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm5, backward_comm_size=ep_combine_size,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                else:
                    if args.tensor_model_parallel_size == 1 :
                        forward_comm = "NONE"
                        backward_comm = "NONE"
                    else:

                        forward_comm = "ALLREDUCE"
                        backward_comm = "NONE"
                    if self.args.recompute_activations and 'attention' in name:
                        forward_compute_time *= 2
                    if "embedding" in name:
                        emb_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "", "embedding"
                        )
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=emb_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=forward_comm_size,
                                backward_compute_time=default_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
                    else:
                        if layer_comp_time != None:
                            forward_compute_time  = layer_comp_time
                            backward_compute_time = layer_comp_time
                        else:
                            forward_compute_time = _get_aiob_compute_time(
                                self.compute_cache, "forward", name.split("_")[0]
                            )
                            backward_compute_time = _get_aiob_compute_time(
                                self.compute_cache, "backward", name.split("_")[0]
                            )
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=forward_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=forward_comm_size,
                                backward_compute_time=backward_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
            # compute_time = _get_aiob_compute_time(self.compute_cache, "forward", "embedding")
            # self.workload.append(Work_Item(name="embedding_norm", forward_compute_time=compute_time,
            #                         forward_comm = "ALLREDUCE", forward_comm_size= self.args.vocab_size*self.args.hidden_size*2,
            #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
            #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
            #                         ))
        for i in range(3):
            self.workload.append(
                Work_Item(
                    name="cross_entropy" + str(i + 1),
                    forward_compute_time=compute_time,
                    forward_comm="ALLREDUCE",
                    forward_comm_size=self.args.seq_length * self.args.micro_batch * 4,
                    backward_compute_time=compute_time,
                    backward_comm="NONE",
                    backward_comm_size=0,
                    dp_compute_time=compute_time,
                    dp_comm="NONE",
                    dp_comm_size=0,
                )
            )

        for i in range(4):
            self.workload.append(
                Work_Item(
                    name="optimizer" + str(i + 1),
                    forward_compute_time=compute_time,
                    forward_comm="ALLREDUCE",
                    forward_comm_size=4,
                    backward_compute_time=compute_time,
                    backward_comm="NONE",
                    backward_comm_size=0,
                    dp_compute_time=compute_time,
                    dp_comm="NONE",
                    dp_comm_size=0,
                )
            )

    def workload_generate(self):
        # args.world_size --> total gpus number
        self.ga_num = self.args.global_batch // (self.args.micro_batch * self.args.dp_num)
        if self.ga_num < 1:
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"
            )
        default_compute_time = 1
        compute_time = 0
        tp_comm_size = (
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size
        )
        layers = self.get_model_details()
        total_params, moe_param_count = self._get_total_params()
        # print(f"Total params is {total_params}, moe params is {moe_param_count}")
        # self.workload.append(Work_Item(name="norm", forward_compute_time=0,
        #                         forward_comm = "BROADCAST", forward_comm_size= 8*self.args.micro_batch*self.args.seq_length,
        #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
        #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
        #                         ))
        forward_compute_time = default_compute_time
        backward_compute_time = default_compute_time
        self.workload.append(
            Work_Item(
                name="grad_norm",
                forward_compute_time=forward_compute_time,
                forward_comm="ALLGATHER",
                forward_comm_size=2 * total_params,
                backward_compute_time=backward_compute_time,
                backward_comm="NONE",
                backward_comm_size=0,
                dp_compute_time=default_compute_time,
                dp_comm="REDUCESCATTER",
                dp_comm_size=4 * total_params,
            )
        )
        if not self.args.enable_sequence_parallel:
            self.workload.append(
                Work_Item(
                    name="layernorm",
                    forward_compute_time=default_compute_time,
                    forward_comm="NONE",
                    forward_comm_size=0,
                    backward_compute_time=default_compute_time,
                    backward_comm="ALLREDUCE",
                    backward_comm_size=2 * total_params,
                    dp_compute_time=default_compute_time,
                    dp_comm="NONE",
                    dp_comm_size=0,
                )
            )
        if args.expert_model_parallel_size != args.dp_num:
            self.workload.append(Work_Item(name="moe_grad_norm1", forward_compute_time=default_compute_time,
                                    forward_comm = "NONE", forward_comm_size= 0,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm="ALLGATHER_DP_EP", dp_comm_size=2*moe_param_count
                                    ))
            self.workload.append(Work_Item(name="moe_grad_norm2", forward_compute_time=default_compute_time,
                                    forward_comm = "NONE", forward_comm_size= 0,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm="REDUCESCATTER_DP_EP", dp_comm_size=4*moe_param_count
                                    ))
        for _ in range(self.ga_num):
            for layer in layers:
                name = layer.layer_name
                forward_comm = backward_comm = backward_comm_2 = "NONE"
                forward_comm_size = tp_comm_size
                backward_comm_size = tp_comm_size
                dp_comm = "NONE"
                dp_comm_size = 0
                if self.args.enable_sequence_parallel:
                    if "embedding" in name:
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=default_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=forward_comm_size,
                                backward_compute_time=default_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
                    if "attention_linear" in name:
                        # for non-shareded linear in attention block

                        # similar to row linear but without comms
                        forward_compute_time=_get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )
                        if self.args.recompute_activations:
                            forward_compute_time *= 2
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=forward_compute_time,
                                forward_comm="NONE",
                                forward_comm_size=0,
                                backward_compute_time=backward_compute_time,
                                backward_comm="NONE",
                                backward_comm_size=0,#sp overlap allgather
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
                    if "row" in name:
                        if self.args.recompute_activations and 'attention' in name:
                            forward_comm_size *= 2
                        forward_comm = "REDUCESCATTER"
                        backward_comm = "ALLGATHER"
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm, forward_comm_size= forward_comm_size,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=tp_comm_size,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                    if "column" in name:
                        if self.args.recompute_activations and 'attention' in name:
                            forward_comm_size *= 2
                        forward_comm = "ALLGATHER"
                        forward_comm2 = "NONE"
                        backward_comm = "REDUCESCATTER"
                        backward_comm_2 = "ALLGATHER"
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm, forward_comm_size= forward_comm_size,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                    if "moelayer" in name:
                        forward_comm1 = "ALLGATHER" # for EP
                        forward_comm2 = "ALLTOALL_EP"
                        forward_comm3 = "ALLGATHER"
                        forward_comm4 = "REDUCESCATTER"
                        forward_comm5 = "ALLTOALL_EP"
                        # if args.expert_model_parallel_size != 1:
                        ep_allgather_size = 2 * self.expert_model_parallel_size * self.num_experts * self.tp
                        fwd_ep_dispatch_size = tp_comm_size * self.topk // self.tp
                        bkwd_ep_dispatch_size = tp_comm_size * self.topk // self.tp
                        ep_combine_size = tp_comm_size * self.topk // self.tp

                        if self.args.frame == "DeepSeek":
                            # for DeepEP based on https://github.com/parthpower/DeepEP/commit/50aee15f592bc22142eb04b7d718296b19613ae9
                            # only fprop does the FP8
                            fwd_ep_dispatch_size = int(fwd_ep_dispatch_size * MockedDeepSeek.FP8_FACTOR)
                            # rest of the comm shapes are similar to megatron
                        # EP All gather
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm1, forward_comm_size=ep_allgather_size,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        # EP dispatch
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm2, forward_comm_size=fwd_ep_dispatch_size,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm2, backward_comm_size=bkwd_ep_dispatch_size,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        # TP All reduce
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm3, forward_comm_size=tp_comm_size*self.topk,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm4, backward_comm_size=tp_comm_size*self.topk,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm4, forward_comm_size=tp_comm_size*self.topk,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm3, backward_comm_size=tp_comm_size*self.topk,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        # EP combine
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,
                                    forward_comm = forward_comm5, forward_comm_size=ep_combine_size,
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm5, backward_comm_size=ep_combine_size,
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size
                                    ))
                        
                else:
                    forward_comm = "ALLREDUCE"
                    backward_comm = "ALLREDUCE"
                    if self.args.recompute_activations and 'attention' in name:
                        forward_comm_size *= 2
                    if "embedding" in name:
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=default_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=forward_comm_size,
                                backward_compute_time=default_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
                    else:
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=default_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=forward_comm_size,
                                backward_compute_time=default_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=default_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
            self.workload.append(
                Work_Item(
                    name="embedding_norm",
                    forward_compute_time=default_compute_time,
                    forward_comm="ALLREDUCE",
                    forward_comm_size=self.args.vocab_size * self.args.hidden_size * 2,
                    backward_compute_time=default_compute_time,
                    backward_comm="NONE",
                    backward_comm_size=0,
                    dp_compute_time=default_compute_time,
                    dp_comm="NONE",
                    dp_comm_size=0,
                )
            )
        for i in range(3):
            self.workload.append(
                Work_Item(
                    name="cross_entropy" + str(i + 1),
                    forward_compute_time=compute_time,
                    forward_comm="ALLREDUCE",
                    forward_comm_size=self.args.seq_length * self.args.micro_batch * 4,
                    backward_compute_time=compute_time,
                    backward_comm="NONE",
                    backward_comm_size=0,
                    dp_compute_time=compute_time,
                    dp_comm="NONE",
                    dp_comm_size=0,
                )
            )

        for i in range(4):
            self.workload.append(
                Work_Item(
                    name="optimizer" + str(i + 1),
                    forward_compute_time=compute_time,
                    forward_comm="ALLREDUCE",
                    forward_comm_size=4,
                    backward_compute_time=compute_time,
                    backward_comm="NONE",
                    backward_comm_size=0,
                    dp_compute_time=compute_time,
                    dp_comm="NONE",
                    dp_comm_size=0,
                )
            )

    def dump_file(self, filename):
        filename = filename + ".txt"

        pp_comm_value = 2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size
        if self.args.enable_sequence_parallel:
            pp_comm_value /= self.args.tensor_model_parallel_size

        pp_comm = (
            f"pp_comm: {pp_comm_value}"
            if self.args.pipeline_model_parallel != 1
            else "pp_comm: 0"
        )
        with open(filename, "w") as f:
            f.write((
                f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {self.args.tensor_model_parallel_size} "
                f"ep: {self.args.expert_model_parallel_size} "
                f"pp: {self.args.pipeline_model_parallel} "
                f"vpp: {self.args.num_layers} "
                f"ga: {self.ga_num} all_gpus: {self.args.world_size} "
                f"checkpoints: 0 checkpoint_initiates: 0 "
            ) + pp_comm + "\n")

            f.write(str(len(self.workload)) + "\n")
            for item in self.workload:
                f.write(
                    "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])
                    + "\n"
                )


class simAI_MicroTest:
    def __init__(self, args):
        self.args = args
        self.workload = []

    def _simAI_microtest_convert(self, comm_type):
        if comm_type == "all_reduce" or comm_type == "allreduce":
            return "ALLREDUCE"
        elif comm_type == "all_gather" or comm_type == "allgather":
            return "ALLGATHER"
        elif comm_type == "reduce_scatter" or comm_type == "reducescatter":
            return "REDUCESCATTER"
        elif comm_type == "all_to_all" or comm_type == "alltoall":
            return "ALLTOALL"
        else:
            return

    def workload_generator(self):
        curr_size = self.args.begin_size
        default_compute_time = 1
        while curr_size <= self.args.end_size:
            self.workload.append(
                Work_Item(
                    name="micro_test",
                    forward_compute_time=default_compute_time,
                    forward_comm="NONE",
                    forward_comm_size=0,
                    backward_compute_time=default_compute_time,
                    backward_comm="NONE",
                    backward_comm_size=0,
                    dp_compute_time=default_compute_time,
                    dp_comm=self._simAI_microtest_convert(self.args.test_comm),
                    dp_comm_size=curr_size,
                    process_time=1,
                )
            )
            curr_size *= 2

    def dump_file(self, filename):
        filename = filename + ".txt"
        with open(filename, "w") as f:
            if not self.args.multi_all_reduce_enable:
                f.write(f"MICRO" + "\n")
                f.write(str(len(self.workload)) + "\n")
                for item in self.workload:
                    f.write(
                        "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])
                        + "\n"
                    )
            else:
                f.write(
                    f"HYBRID_TRANSFORMER_FWD_IN_BCKWD	model_parallel_NPU_group: {self.args.tensor_model_parallel_size} \
                        expert_parallel_npu_group: {self.args.expert_model_parallel_size} pp: {self.args.pipeline_model_parallel} \
                        ga: {self.ga_num} all_gpus: {self.args.world_size} checkpoints: 0 checkpoint_initiates: 0"
                    + "\n"
                )
                f.write(str(len(self.workload)) + "\n")
                for item in self.workload:
                    f.write(
                        "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])
                        + "\n"
                    )


if __name__ == "__main__":
    args = get_params()
    print(args)
    if args.frame == "DeepSeek":
        model = DeepSeekV3Model(args)
    else:
        model = MegatronModel(args)
    result_dir = "results/workload/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    filename = f"{args.gpu_type}-{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-MOE-{args.moe_enable}-GEMM-{args.moe_grouped_gemm}-flash_attn-{args.use_flash_attn}"
    filepath = os.path.join(result_dir, filename)
    params = model.parameters()
    # work = SIMAI_workload(model, args, GPU_Tensor_core.A100, "gpt13B")
    # name_layers = work.workload_generate()
    # work.dump_file("test")
    print(sum(p.numel() for p in params))
    if args.aiob_enable:
        params = model.parameters()
        args.model_param = sum(p.numel() for p in params)
        if args.comp_filepath == None:

            comp_filepath = get_comp_out(args)

            compute_cache = extract_averages(comp_filepath,args)
        else:
            print("comp_filepath:", args.comp_filepath)
            comp_filepath = args.comp_filepath
            compute_cache = extract_averages(comp_filepath,args)

        print("compute_cache = {")
        for key, value in compute_cache.items():
            print(f"    '{key}' : {value},")
        print("}")
        work = SIMAI_workload(
            model, args,compute_cache
        )
        name_layers = work.workload_generate_aiob()

        # set comm_size = 0 for any comm_type == NONE
        for i in range(len(work.workload)):
            if work.workload[i].forward_comm == "NONE":
                work.workload[i].forward_comm_size = 0
            if work.workload[i].backward_comm == "NONE":
                work.workload[i].backward_comm_size = 0

        work.dump_file(filepath)
        print("workload save in :", filepath)
    # print(args)
    else:

        work = SIMAI_workload(model, args, None)
        name_layers = work.workload_generate()
        work.dump_file(filepath)
        print(f"workload save in : {filepath}.txt")
