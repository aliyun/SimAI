import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workload_generator.mocked_model.MockedModel
import workload_generator.mocked_model.inference.MockedDeepSeek as MockedDeepSeek
import workload_generator.mocked_model.inference.MockedQwen3Moe as MockedQwen3Moe
import workload_generator.mocked_model.inference.MockedQwen3Next as MockedQwen3Next
from workload_generator.mocked_model.MockedModel import InferencePhase
from workload_generator.SimAI_inference_workload_generator import _get_aiob_compute_time, _get_model_details, LayerInfo
from utils.utils import CommType, get_params, get_comp_out, extract_inference_averages
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
from collections import OrderedDict
'''
可以用字典按键(layer_id, layer_type)聚合，把layer_name丢弃，只累加comp_time与comm_size。示例函数如下：

def merge_layers(rows, sort=False):
    """
    rows: 列表，每项为 [layer_id, layer_name, layer_type, layer_comp_time, layer_comm_size]
    返回: 合并后的列表，每项为 [layer_id, layer_type, sum_comp_time, sum_comm_size]
    sort: 是否按 (layer_id, layer_type) 排序输出；默认保持首次出现的顺序
    """
    from collections import OrderedDict

    totals = OrderedDict()  # 保持首次出现的顺序
    for layer_id, _, layer_type, comp, comm in rows:
        key = (layer_id, layer_type)
        if key in totals:
            totals[key][0] += comp
            totals[key][1] += comm
        else:
            totals[key] = [comp, comm]

    items = totals.items()
    if sort:
        items = sorted(items, key=lambda kv: (kv[0][0], kv[0][1]))

    return [[layer_id, layer_type, comp_sum, comm_sum]
            for (layer_id, layer_type), (comp_sum, comm_sum) in items]

示例
rows = [
    [0, "x", "A", 0, 0],
    [0, "y", "A", 0, 0],
    [1, "foo", "B", 3.2, 10],
    [1, "bar", "B", 1.8, 5],
]
print(merge_layers(rows))
# 输出: [[0, 'A', 0, 0], [1, 'B', 5.0, 15]]

如果你用 pandas，也可一行完成：
import pandas as pd
df = pd.DataFrame(rows, columns=["layer_id","layer_name","layer_type","layer_comp_time","layer_comm_size"])
out = (df
       .drop(columns=["layer_name"])
       .groupby(["layer_id","layer_type"], as_index=False)
       .sum())
# out.values.tolist() 得到同样的结果列表
'''

class LayerType(Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    MOE = "moe"

def _get_layer_comm_size(layer_type, tp_size, ep_size):
    if layer_type == LayerType.MOE:
        return ep_size
    else:
        return tp_size
    
def _extract_layer_type(layer_name):
    if "attention" in layer_name:
        return LayerType.ATTENTION
    elif "mlp" in layer_name:
        return LayerType.MLP
    elif "moe" in layer_name or "expert" in layer_name:
        return LayerType.MOE
    else:
        return None


class VidurWorkload():
    def __init__(self, model, args, compute_cache=None):
        self.model = model
        self.args = args
        self.compute_cache = compute_cache
        self.vidur_layers = []
        self.seq_len = args.seq_length
        self.tp = args.tensor_model_parallel_size
        self.mbs = args.micro_batch
        self.batch_size = args.micro_batch  # 添加缺失的 batch_size 属性
        self.num_layers = 0
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
        vidur_layers_tmp = []
        layers = _get_model_details(self.model)
        for layer in layers:
            # print(f'Layer {layer.layer_id} {layer.layer_name}')
            layer_type = _extract_layer_type(layer.layer_name)
            comp_time = _get_aiob_compute_time(
                    self.compute_cache, "forward", layer.layer_name
                    )
            vidur_layers_tmp.append(
                [layer.layer_id,
                layer.layer_name,
                layer_type,
                comp_time,
                0]
            )
        totals = OrderedDict()
        for layer_id, _, layer_type, comp, comm in vidur_layers_tmp:
            key = (layer_id, layer_type)
            if key in totals:
                totals[key][0] += comp
                totals[key][1] += comm
            else:
                totals[key] = [comp, comm]

        items = totals.items()

        self.vidur_layers = [[layer_id, layer_type, comp_sum, comm_sum]
                for (layer_id, layer_type), (comp_sum, comm_sum) in items]
        
        # Comm part
        tp_comm_size, ep_dispatch_size, ep_combine_size = self.get_comm_size()
        tp_comm_size = tp_comm_size if self.tp > 1 else 0
        ep_comm_size = ep_dispatch_size + ep_combine_size
        for vidur_layer in self.vidur_layers:
            comm_size = _get_layer_comm_size(vidur_layer[1], tp_comm_size, ep_comm_size)
            vidur_layer[3] = comm_size

    def dump_file(self, filename):
        filename = filename + ".csv"
        with open(filename, "w") as f:
            f.write("layer_id\tlayer_name\tcomp_time\tcomm_size\n")
            # for layer_name, layer_info in self.workload.items():
            #     if layer_info.layer_comp_time == 0:
            #         continue
            #     f.write(f"{layer_name}\t{layer_info.layer_comp_time}\t{layer_info.layer_comm_size}\n")
            for vidur_layer in self.vidur_layers:
                f.write(f'{vidur_layer[0]}\t{vidur_layer[1].value}\t{vidur_layer[2]}\t{vidur_layer[3]}\n')

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
        
    phase = getattr(args, "phase", InferencePhase.DECODE.value)
    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    filename = f"vidur-{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-bs{args.micro_batch}-seq{args.seq_length}-{phase}"
    
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
    
    work = VidurWorkload(
        model, args, compute_cache
    )
    work.workload_generate_aiob()
    filepath = os.path.join(result_dir, filename)
    work.dump_file(filepath)
    print("workload save in :", filepath)
    print("Finish Model initialization")
    
    # TODO: 实现Vidur工作负载生成和保存逻辑
    # print("VidurWorkload class initialized. Implementation pending.")