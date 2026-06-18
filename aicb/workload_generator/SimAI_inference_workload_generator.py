import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workload_generator.mocked_model.MockedModel
import workload_generator.mocked_model.inference.MockedDeepSeek as MockedDeepSeek
import workload_generator.mocked_model.inference.MockedQwen3Moe as MockedQwen3Moe
import workload_generator.mocked_model.inference.MockedQwen3Next as MockedQwen3Next
from workload_generator.mocked_model.MockedModel import InferencePhase
from utils.utils import CommType, get_params, get_comp_out, extract_inference_averages, Strategy, get_ep_expected_m_per_group
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
    forward_comm_size: str = dataclasses.field(default="0")
    backward_compute_time: int = dataclasses.field(default=0)
    backward_comm: str = dataclasses.field(default="NONE")
    backward_comm_size: int = dataclasses.field(default=0)
    dp_compute_time: int = dataclasses.field(default=0)
    dp_comm: str = dataclasses.field(default="NONE")
    dp_comm_size: int = dataclasses.field(default=0)
    process_time: int = dataclasses.field(default=100)



def _get_aiob_compute_time(compute_cache, forward_or_backward, stage, dowarn=True, aiob_enable=True):
    default_compute_time = 1
    if not aiob_enable:
        return default_compute_time
    compute_time_map = compute_cache
    if stage == "shared_experts" or stage == "dense_mlp":
        prefix = "mlp"
    else:
        prefix = stage

    for key, value in compute_time_map.items():
        if prefix == key:

            compute_time = compute_time_map.get(key)
            return compute_time

    if dowarn:
        print(f"[warn] can't match any stage with {stage}, using default_compute_time {default_compute_time}")
    return default_compute_time

class LayerInfo:
    def __init__(self, layer_id, layer_name):
        self.layer_id = layer_id
        self.layer_name = layer_name
def _get_model_details(model):
    layers = []
    visited = set()

    def traverse_model(model, child_id=0):
        if id(model) in visited:
            return
        visited.add(id(model))
        if (
                isinstance(model, MockedDeepSeek.DeepSeekAttention)
                or isinstance(model, MockedDeepSeek.DeepSeekMLP)
                # or isinstance(model, MockedDeepSeek.DeepSeekMOE)
                # or isinstance(model, MockedDeepSeek.DeepSeekColumnLinear)
                # or isinstance(model, MockedDeepSeek.DeepSeekRowLinear)
                or isinstance(model, MockedQwen3Moe.Qwen3MoeRMSNorm)
                or isinstance(model, MockedQwen3Moe.Qwen3MoeAttention)
                or isinstance(model, MockedQwen3Moe.Qwen3MoeRoute)
                or isinstance(model, MockedQwen3Moe.Qwen3MoeExpert)
                or isinstance(model, MockedQwen3Next.Qwen3NextRMSNorm)
                or isinstance(model, MockedQwen3Next.Qwen3NextAttention)
                or isinstance(model, MockedQwen3Next.Qwen3NextGatedDeltaNet)
                or isinstance(model, MockedQwen3Next.Qwen3NextRoute)
                or isinstance(model, MockedQwen3Next.Qwen3NextExpert)
            ):
                layers.append(LayerInfo(model.layer_id, model.name))
        for child in model.child_modules():
            traverse_model(child, child_id+1)

    traverse_model(model)

    return layers

class SimAIWorkload():
    def __init__(self, model, args, compue_cache=None):
        self.model = model
        self.args = args
        self.compute_cache = compute_cache
        self.workload = []
        self.seq_len = args.seq_length
        self.tp = args.tensor_model_parallel_size
        self.mbs = args.micro_batch
        self.batch_size = args.micro_batch  # 添加缺失的 batch_size 属性
        if args.moe_enable:
            self.expert_model_parallel_size = args.expert_model_parallel_size
            self.num_experts = args.num_experts
            #如果有moe_router_topk则用，否则用num_experts_per_tok
            self.topk = args.moe_router_topk if hasattr(args,"moe_router_topk") else args.num_experts_per_tok

    
    def get_comm_size(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.batch_size
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
        tp_comm_size = 2 * m * self.args.hidden_size
        ep_combine_size = tp_comm_size * self.topk // self.tp
        ep_dispatch_size = ep_combine_size

        if any(s in self.args.frame for s in ('DeepSeek', 'Qwen3')): 
            # for DeepEP based on https://github.com/parthpower/DeepEP/commit/50aee15f592bc22142eb04b7d718296b19613ae9
            ep_dispatch_size = int(ep_dispatch_size * MockedDeepSeek.FP8_FACTOR)

        return tp_comm_size, ep_dispatch_size, ep_combine_size
        

    def workload_generate_aiob(self):
        default_compute_time = 1
        compute_time = 0
        tp_comm_size, ep_dispatch_size, ep_combine_size = self.get_comm_size()
        layers = _get_model_details(self.model)
        print("layer length:" , len(layers))

        for layer in layers:
            name = layer.layer_name
            if args.tensor_model_parallel_size == 1 :
                forward_comm = "NONE"
            else:
                forward_comm = "ALLREDUCE"

            if "norm" in name or "shared_expert" in name: # comp only layers
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )
                self.workload.append(
                    Work_Item(
                        name=name,
                        forward_compute_time=compute_time,
                        forward_comm="NONE",
                        forward_comm_size="0",
                        backward_compute_time=0,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=0,
                        dp_comm="NONE",
                        dp_comm_size=0,
                    )
                )
            elif "attention" in name or "dense_mlp" in name: # comp & ALLREDUCE comm layers
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )
                self.workload.append(
                    Work_Item(
                        name=name,
                        forward_compute_time=compute_time,
                        forward_comm=forward_comm,
                        forward_comm_size=tp_comm_size,
                        backward_compute_time=0,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=0,
                        dp_comm="NONE",
                        dp_comm_size=0,
                    )
                )
            elif "moe_route" in name: # moe route
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )
                self.workload.append(
                    Work_Item(
                        name="moe_route",
                        forward_compute_time=compute_time,
                        forward_comm="ALLTOALL_EP",
                        # forward_comm_size = "2*" + str(token_num) + "*" + str(self.args.moe_router_topk) + "*" + str(self.args.expert_dim),
                        forward_comm_size = ep_dispatch_size,
                        backward_compute_time=default_compute_time,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=default_compute_time,
                        dp_comm="NONE",
                        dp_comm_size=0
                    )
                )

            elif "moe_expert" in name: # moe experrt
                compute_time = _get_aiob_compute_time(
                        self.compute_cache, "forward", name
                        )

                if self.args.frame == "DeepSeek":
                    #TODO currently AiobDeepSeek doesn't support moe_route, should be fixed.
                    self.workload.append(
                        Work_Item(
                            name="moe_route",
                            forward_compute_time=default_compute_time,
                            forward_comm="ALLTOALL_EP",
                            # forward_comm_size = "2*" + str(token_num) + "*" + str(self.args.moe_router_topk) + "*" + str(self.args.expert_dim),
                            forward_comm_size = ep_dispatch_size,
                            backward_compute_time=default_compute_time,
                            backward_comm="NONE",
                            backward_comm_size=0,
                            dp_compute_time=default_compute_time,
                            dp_comm="NONE",
                            dp_comm_size=0
                        )
                    )
                self.workload.append(
                    Work_Item(
                        name="moe_expert",
                        forward_compute_time=compute_time,
                        forward_comm="ALLTOALL_EP",
                        forward_comm_size = ep_combine_size,
                        backward_compute_time=default_compute_time,
                        backward_comm="NONE",
                        backward_comm_size=0,
                        dp_compute_time=default_compute_time,
                        dp_comm="NONE",
                        dp_comm_size=0
                    )
                )
    def dump_file(self, filename):
        filename = filename + ".txt"

        pp_comm_value = 2 * self.mbs * self.seq_len * self.args.hidden_size * (1 if self.args.pipeline_model_parallel > 1 else 0)

        pp_comm = (
            f"pp_comm: {pp_comm_value}"
        )
        with open(filename, "w") as f:
            f.write((
                f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {self.args.tensor_model_parallel_size} "
                f"ep: {self.args.expert_model_parallel_size} "
                f"pp: {self.args.pipeline_model_parallel} "
                f"all_gpus: {self.args.world_size} "
                f"mode: 1 " # inference
                f"vpp: 1 ga: 1 checkpoints: 0 checkpoint_initiates: 0 "
            ) + pp_comm + "\n")

            f.write(str(len(self.workload)) + "\n")
            for item in self.workload:
                f.write(
                    "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])
                    + "\n"
                )
                
                

if __name__ == "__main__":
    import sys
    import argparse
    from utils.utils import get_params
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate inference workload for AI models")
    parser.add_argument("model_name", help="Model name (e.g., DeepSeek-671B, Qwen3-Moe-235B, Qwen3-Next-80B)")
    parser.add_argument("config_file", nargs="?", help="Path to JSON config file")
    
    # Add arguments that has default value
    parser.add_argument("--aiob_enable", action="store_true", default=False, help="Enable AIOB")
    parser.add_argument("--aiob_forward_loops", type=int, default=1, help="Number of AIOB forward loops")
    parser.add_argument("--moe_routing_strategy", default=Strategy.RoundRobin.value, choices=[s.value for s in Strategy], help="MoE routing strategy")
    parser.add_argument("--seq_length", type=int, default=1, help="Sequence length")
    parser.add_argument("--micro_batch", type=int, default=1, help="Micro batch size")
    parser.add_argument("--world_size", type=int, default=1, help="World size")
    parser.add_argument("--tensor_model_parallel_size", default=1, type=int, help="Tensor model parallel size")
    parser.add_argument("--expert_model_parallel_size", default=1, type=int, help="Expert model parallel size")
    parser.add_argument("--pipeline_model_parallel", default=1, type=int, help="Pipeline model parallel size")
    parser.add_argument("--moe_enable", default=True, action="store_true", help="Enable MoE")
    parser.add_argument("--result_dir", default="results/workload/", help="Result directory")
    parser.add_argument("--phase",
                        choices=[InferencePhase.DECODE.value, InferencePhase.PREFILL.value],
                        default=InferencePhase.DECODE.value, help="Inference phase")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    config_file = args.config_file
    
    if "Qwen3-Moe" in model_name:
        args = MockedQwen3Moe.Qwen3MoeParams(config_file, args)
        model = MockedQwen3Moe.Qwen3MoeModel(args)
    elif "Qwen3-Next" in model_name:
        args = MockedQwen3Next.Qwen3NextParams(config_file, args)
        model = MockedQwen3Next.Qwen3NextModel(args)
    elif "DeepSeek" in model_name:
        args = MockedDeepSeek.DeepSeekParams(config_file, args)
        model = MockedDeepSeek.DeepSeekModel(args)
    else:
        print(f"Invalid model name: {model_name}")
        sys.exit(1)
    # args = MockedDeepSeek.DeepSeekParams(config_file)
    # model = MockedDeepSeek.DeepSeekModel(args)
    phase = getattr(args, "phase", InferencePhase.DECODE.value)
    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-bs{args.micro_batch}-seq{args.seq_length}-{phase}"
    
    if args.aiob_enable:
        if "Qwen3-Moe" in model_name:
            import workload_generator.mocked_model.inference.AiobQwen3Moe as AiobQwen3Moe
            aiob_model = AiobQwen3Moe.Qwen3MoeModel(args)
            aiob_output_filepath = aiob_model()
        elif "Qwen3-Next" in model_name:
            import workload_generator.mocked_model.inference.AiobQwen3Next as AiobQwen3Next
            aiob_model = AiobQwen3Next.Qwen3NextModel(args)
            aiob_output_filepath = aiob_model()
        elif "DeepSeek" in model_name:
            import workload_generator.mocked_model.inference.AiobDeepSeek as AiobDeepSeek
            aiob_model = AiobDeepSeek.DeepSeekModel(args)
            aiob_output_filepath = aiob_model()
        else:
            print(f"Invalid model name: {model_name}")
            sys.exit(1)

    else:
        aiob_dir = "results/aiob_outputs"
        aiob_output_filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-bpg{args.micro_batch}-seq{args.seq_length}-{phase}.txt"
        aiob_output_filepath = os.path.join(aiob_dir,aiob_output_filename)
        if not os.path.exists(aiob_output_filepath):
            print(f"aiob not enabled, and {aiob_output_filepath} not found. Using default compute time.")
            aiob_output_filepath = ""
        else:
            print(f"aiob not enabled, using existing file {aiob_output_filepath}.")
    compute_cache = extract_inference_averages(aiob_output_filepath,args)
    print("compute_cache = {")
    for key, value in compute_cache.items():
        print(f"    '{key}' : {value},")
    print("}")
    
    work = SimAIWorkload(
        model, args,compute_cache
    )
    name_layers = work.workload_generate_aiob()
    # set comm_size = 0 for any comm_type == NONE
    for i in range(len(work.workload)):
        if work.workload[i].forward_comm == "NONE":
            work.workload[i].forward_comm_size = 0
        if work.workload[i].backward_comm == "NONE":
            work.workload[i].backward_comm_size = 0

    filepath = os.path.join(result_dir, filename)
    work.dump_file(filepath)
    print("workload save in :", filepath)
    print("Finish Model initialization")