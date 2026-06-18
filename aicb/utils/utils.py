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

from typing import List, Dict, Tuple
import pandas as pd
import pickle
from enum import Enum
import argparse
import sys
import time
import os
import json
from collections import defaultdict
import math
import re
import numpy as np

try:
    import torch
except ImportError as e:
    torch = None
    print("Failed to import 'torch'.")

def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool],
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example, 
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then 
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the 
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the 
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).
        
        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks
class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        if 'ep' in order:
            if 'ep-dp' not in order and 'dp-ep' not in order:
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        '''
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)
        return ranks

def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


def Comp_with_aiob(workload, compute_cache):
    for item in workload.workload:
        if item.comm_type == CommType.computation:
            for key in compute_cache:
                key_temp = key.split("_")[0]
                if key_temp in item.stage:
                    item._elapsed_time = compute_cache[key]
                    break
    return workload


def get_comp_out(args):
    vocab_size = args.vocab_size
    batch_size = args.micro_batch
    seq_len = args.seq_length
    tp = args.tensor_model_parallel_size
    vocab_size = args.padded_vocab_size
    if "Megatron" or "DeepSeek" in args.frame:
        device = torch.cuda.current_device()
        from workload_generator.mocked_model.training.AiobMegatron import MegatronModel
        measure_model = MegatronModel(args)
        measure_model.train()
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        # total_input_1 = torch.rand(args.seq_len,
        #                                       args.batch_size,
        #                                       args.hidden_size,
        #                                       device=device).to(dtype)
        masked_input = torch.randint(
            0,
            math.ceil(vocab_size / tp),
            (batch_size, seq_len),
            device=device,
            dtype=torch.int64,
        )
        filepath = measure_model(masked_input)
        return filepath

    

def extract_inference_averages(file_path,args):
    if file_path is None or file_path == "":
        return {}
    attention_norm_avg_sum = 0.0
    attention_gdn_avg_sum = 0.0
    attention_avg_sum = 0.0
    mlp_avg_sum = 0.0
    moe_norm_avg_sum = 0.0
    moe_route_avg_sum = 0.0
    moe_expert_sum = 0.0

    section_header_re = re.compile(r"^(\w+):")
    time_gpu_avg_re = re.compile(r"time_gpu_avg:\s+(\d+(\.\d+)?)")
    # time_gpu_min_re = re.compile(r"time_gpu_min:\s+(\d+(\.\d+)?)")

    with open(file_path, "r") as file:
        current_section = None

        for line in file:
            header_match = section_header_re.match(line)
            if header_match:
                current_section = header_match.group(1).strip()

            avg_match = time_gpu_avg_re.search(line)
            # min_match = time_gpu_min_re.search(line)
            if avg_match and current_section:
                avg_value = float(avg_match.group(1)) * 1000
                if "atten_norm" in current_section:
                    attention_norm_avg_sum += avg_value
                elif "gdn" in current_section:
                    attention_gdn_avg_sum += avg_value
                elif "atten" in current_section:
                    attention_avg_sum += avg_value
                elif "mlp" in current_section:
                    mlp_avg_sum += avg_value
                elif "moe_norm" in current_section:
                    moe_norm_avg_sum += avg_value
                elif "moe_route" in current_section:
                    moe_route_avg_sum += avg_value
                elif "moe" in current_section:
                    moe_expert_sum += avg_value

    full_cache = {
        "attention_norm": round(attention_norm_avg_sum),
        "attention_gdn": round(attention_gdn_avg_sum),
        "attention_layer": round(attention_avg_sum),
        "mlp": round(mlp_avg_sum),
        "moe_norm": round(moe_norm_avg_sum),
        "moe_route": round(moe_route_avg_sum),
        "moe_expert": round(moe_expert_sum),
    }
    compute_cache = {key: value for key, value in full_cache.items() if value != 0}

    return compute_cache


def get_compute_path(args):
    result_dir = "./results/aiob_outputs"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    phase = args.phase
    filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp1-ep{args.expert_model_parallel_size}-bpg{args.micro_batch}-seq{args.seq_length}-{phase}.txt"
    filepath = os.path.join(result_dir, filename)
    return filepath

def write_time(time_list, args):
    filepath = get_compute_path(args)
    with open(filepath, "w") as file:
        file.write(f"inference_term:{args.phase}\n")
        data_str = json.dumps(time_list, indent=4)

        file.write(data_str)
    return filepath

def calculate_stats(time_list, file_name):
    stats = {}
    for key, values in time_list.items():
        times = [entry['time_gpu'] for entry in values]
        variance = np.var(times)
        p99 = np.percentile(times, 99)
        stats[key] = {'variance': variance, 'P99': p99}
    with open(file_name, 'w') as file:
        json.dump(stats, file, indent=4)
    print("output stats successfully")

def extract_averages(file_path,args):
    attention_avg_sum = 0.0
    mlp_avg_sum = 0.0
    other_avgs = {}
    grad_forward = 0.0
    grad_backward = 0.0

    section_header_re = re.compile(r"^(\w+):")
    time_gpu_avg_re = re.compile(r"time_gpu_avg:\s+(\d+(\.\d+)?)")
    time_gpu_min_re = re.compile(r"time_gpu_min:\s+(\d+(\.\d+)?)")

    # from DeepSeek's MLA per-layers compute
    # must match with that of AiobDeepSeek->DeepSeekMLA's return from forward()
    # FIXME: this should be more generic
    per_layer_time_map = {
        "attention_linear_q_lora": 0,
        "attention_q_column": 0,
        "attention_linear_kv_lora": 0,
        "attention_kv_column": 0,
        "attention_o_row": 0
    }

    with open(file_path, "r") as file:
        current_section = None

        for line in file:
            header_match = section_header_re.match(line)
            if header_match:
                current_section = header_match.group(1).strip()

            avg_match = time_gpu_avg_re.search(line)
            min_match = time_gpu_min_re.search(line)
            if current_section == "param_time":
                if min_match:
                    grad_forward = float(min_match.group(1)) * 1000 #us
                if avg_match:
                    grad_backward = float(avg_match.group(1)) * 1000
            elif avg_match and current_section:
                avg_value = float(avg_match.group(1)) * 1000
                if "atten" in current_section or current_section == "layernorm":

                    if args.recompute_activations and 'flash' in current_section:
                        attention_avg_sum += avg_value*2
                    else:
                        attention_avg_sum += avg_value
                elif "mlp" in current_section or current_section == "layernorm2":
                    mlp_avg_sum += avg_value
                else:
                    other_avgs[current_section] = avg_value

                if current_section in per_layer_time_map.keys():
                    per_layer_time_map[current_section] = round(avg_value)


    # 四舍五入并转换为整数
    attention_forward = round(attention_avg_sum)
    attention_backward = attention_forward
    mlp_forward = round(mlp_avg_sum)
    mlp_backward = mlp_forward
    grad_backward = round(grad_backward)
    grad_forward = round(grad_forward)
    other_avgs_int = {k: round(v) for k, v in other_avgs.items() if k != "param_time"}

    a100_compute_cache = {
        "attention_forward": attention_forward,
        "attention_backward": attention_backward,
        "mlp_forward": mlp_forward,
        "mlp_backward": mlp_backward,
        "grad_forward": grad_forward,
        "grad_backward": grad_backward,
    }
    a100_compute_cache.update(other_avgs_int)
    a100_compute_cache.update(per_layer_time_map)

    return a100_compute_cache


def process_all_keys(input_file):

    with open(input_file, "r") as file:
        first_line_str = file.readline().strip()
        remaining_content = file.read().strip()
    # 尝试修复和构建合法的 JSON 字符串
    corrected_content = remaining_content.replace("}{", "},{").replace("]}{", "]},{")

    # 构建 JSON 数组
    json_str = f"[{corrected_content}]"

    try:
        data = json.loads(json_str)

        processed_results = defaultdict(lambda: defaultdict(list))
        for entry in data:
            for key, measurements in entry.items():
                for measure in measurements:
                    measure_key, measure_value = next(iter(measure.items()))
                    if "time_gpu" in measure_key:
                        processed_results[key]["time_gpu"].append(measure["time_gpu"])
                    else:
                        processed_results[key][measure_key] = measure_value

        for key, values in processed_results.items():
            if "time_gpu" in values:
                gpu_times = values["time_gpu"]
                min_time_gpu = min(gpu_times)
                gpu_times_filtered = [t for t in gpu_times if t <= 3 * min_time_gpu]
                values["time_gpu_max"] = max(gpu_times_filtered)
                values["time_gpu_min"] = min_time_gpu
                values["time_gpu_avg"] = sum(gpu_times_filtered) / len(
                    gpu_times_filtered
                )
                del values["time_gpu"]

        with open(input_file, "w") as outfile:
            outfile.write(first_line_str + "\n")
            for key, values in processed_results.items():
                outfile.write(f"{key}:\n")
                for subkey, subvalue in values.items():
                    outfile.write(f"    {subkey}: {subvalue}\n")
        print(f"Compute-results save in:{input_file}")

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print("Invalid JSON content:\n", corrected_content)


def cuda_timing_decorator(func):
    def wrapper(*args, **kwargs):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event) * 1000  # 时间以毫秒为单位
        return result, elapsed_time_ms

    return wrapper
def get_aiob_path(args):
    result_dir = "./results/aiob_outputs"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-flash_attn-{args.use_flash_attn}.txt"
    filepath = os.path.join(result_dir, filename)
    return filepath

def write_op(time_list, args):
    filepath = get_aiob_path(args)
    with open(filepath, "w") as file:
        file.write(f"train_iter:{args.epoch_num}\n")
        data_str = json.dumps(time_list, indent=4)

        file.write(data_str)
    return filepath


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    UNUSED = 8


class CommType(str, Enum):
    """Enum class for possible comm types"""

    all_reduce = "all_reduce"
    isend = "isend"
    irecv = "irecv"
    broadcast = "broadcast"
    all_gather = "all_gather"
    reduce_scatter = "reduce_scatter"
    barrier = "barrier"
    reduce = "reduce"
    reduce_scatter_tensor = "reduce_scatter_tensor"
    all_gather_into_tensor = "all_gather_into_tensor"
    computation = "computation"
    epoch_end = "epoch_end"
    all_to_all = "all_to_all"

    @classmethod
    def get_comm_type(cls, value):
        for comm_type in cls:
            if comm_type.value == value:
                return comm_type
        raise ValueError("Invailid communication type")


class CommGroup(str, Enum):
    """Enum class for possible comm groups"""

    dp_group = "dp_group"
    pp_group = "pp_group"
    tp_group = "tp_group"
    ep_group = "ep_group"
    ep_dp_group = "ep_dp_group"
    ep_tp_group = "ep_tp_group"
    embedding_group = "embedding_group"
    all = "all_nodes"


class WorkloadWriter:
    @staticmethod
    def write_workload(workload: List[Dict], args, filename: str):
        df = pd.DataFrame.from_dict(workload)
        df = df.fillna(-1)
        df.to_csv(filename, index=False)

    @staticmethod
    def load_workload(filename: str) -> List[Dict]:
        filename = filename.split(".")
        filename[-1] = "pkl"
        filename = ".".join(filename)
        workload, args = pickle.load(open(filename, "rb"))
        return workload, args


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame",
        help="communication framework",
        choices=["Megatron", "DeepSpeed", "collective_test", "DeepSeek"],
        default="Megatron",
    )
    parser.add_argument("--gpu_type", type=str, default=None),
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1,
                        help='Degree of tensor model parallelism.')
    parser.add_argument("--pipeline_model_parallel", type=int, default=1,
                        help='Degree of pipeline model parallelism.')
    parser.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')
    parser.add_argument("--pp_rank", type=int, default=-1,
                        help='Rank where encoder and decoder should be split.')
    parser.add_argument("--global_batch", type=int, default=4,
                        help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    parser.add_argument("--micro_batch", type=int, default=1,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.'
                        )
    parser.add_argument("--epoch_num", type=int, default=1,
                        help="Number of iterations")
    parser.add_argument("--computation_enable", action="store_true", help="Enable computation")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=None,
        help="Transformer Feed-Forward Network hidden size. "
        "This is set to 4*hidden-size if not provided",
    )
    parser.add_argument(
        "--enable_visual",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument("--workload_only", action="store_true", help="Only generate workload")
    get_model_params(parser)
    get_ds_params(parser)
    get_megatron_params(parser)
    get_collective_test_params(parser)
    get_moe_params(parser)
    get_simAI_workload_params(parser)
    get_aiob_params(parser)
    get_deepseek_params(parser)
    args = parser.parse_args()

    assert (
        args.world_size % (args.tensor_model_parallel_size * args.pipeline_model_parallel) == 0
    ), f"world size: {args.world_size}, tp: {args.tensor_model_parallel_size}, pp: {args.pipeline_model_parallel}"
    if args.moe_enable:
        assert (
            args.moe_enable and args.enable_sequence_parallel
        ), f"moe must be enabled with sequence parallel"
    args.dp_num = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel)
    # assert args.global_batch % (args.dp_num * args.micro_batch) == 0, \
    #     f"global_batch: {args.global_batch}, dp: {args.dp_num}, micro_batch: {args.micro_batch}"
    args.num_microbatches = args.global_batch // (args.dp_num * args.micro_batch)
    if args.aiob_enable and not args.computation_enable:
            args.computation_enable = True
            
    if args.num_attention_heads is None:
        args.num_attention_heads = args.num_layers

                    
    args.padded_vocab_size = get_padded_vocab_size(args)
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

        else:
            args.ffn_hidden_size = 4 * args.hidden_size
    if args.swiglu:
        args.gated_linear_unit = True
        args.bias_gelu_fusion = False
    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.dtype == "float16", \
            "Expert parallelism is not supported with fp16 training."
    if args.moe_grouped_gemm:
        assert args.dtype == "bfloat16", 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
    if args.pipeline_model_parallel > 1 :
        args.num_layers = int(args.num_layers//args.pipeline_model_parallel)
    return args


ARGS = None


def get_args():
    global ARGS
    if ARGS is not None:
        return ARGS
    ARGS = get_params()
    return ARGS


def get_aiob_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--aiob_enable",
        action="store_true",
        help="Enable aiob to get operation real compute time",
    )
    parser.add_argument("--comp_filepath", type=str, default=None,
                        help="Use aiob_lib to get operation real compute time",)
    parser.add_argument("--gated_linear_unit", default=False)
    parser.add_argument("--bias_gelu_fusion", action="store_true",
                        help='Enable bias and gelu fusion.')
    parser.add_argument("--openai_gelu", action="store_true",
                         help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    parser.add_argument("--onnx_safe", action="store_true",
                        help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    parser.add_argument("--squared_relu", action="store_true",
                        help='Use squared relu activation instead of default gelu')
    parser.add_argument('--recompute_activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')


def get_model_params(parser: argparse.ArgumentParser):
    parser.add_argument("--model_name", help="Model for training")
    parser.add_argument(
        "--hidden_size", type=int, help='Tansformer hidden size.', default=1024
    )
    parser.add_argument("--num_layers", type=int, help='Number of transformer layers.', default=24)
    parser.add_argument(
        "--seq_length", type=int, help='Maximum sequence length to process.', default=2048
    )
    parser.add_argument("--num_attention_heads", help='Number of transformer attention heads.',type=int, default=None)
    parser.add_argument("--vocab_size", type=int, help='Size of vocab before EOD or padding.', default=32000)
    parser.add_argument("--max_position_embeddings", type=int,help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.', default=4096)
    parser.add_argument("--add_bias_linear",help='Enable bias in the linear layers', action="store_true")
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use FlashAttention implementation of attention.",
    )
    parser.add_argument(
        "--swiglu",
        action="store_true",
        help="Use gated linear units and SiLU activation instead of default gelu",
    )


def get_ds_params(parser: argparse.ArgumentParser):
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--amp_enabled", action="store_true")
    parser.add_argument("--reduce_bucket_size", type=int, default=int(5e8))

    # for stage1/2 only
    parser.add_argument("--allgather_bucket_size", type=int, default=int(5e8))
    parser.add_argument("--contiguous_gradients", action="store_true")

    # for stage 3 only
    parser.add_argument("--param_persistence_threshold", type=int, default=int(1e5))
    parser.add_argument(
        "--model_persistence_threshold", type=int, default=int(sys.maxsize)
    )
    parser.add_argument("--max_live_parameters", type=int, default=int(1e9))
    parser.add_argument("--prefetch_bucket_size", type=int, default=int(1e9))


def get_megatron_params(parser: argparse.ArgumentParser):
    parser.add_argument("--enable_sequence_parallel",help='Enable sequence parallel optimization.',action="store_true")
    parser.add_argument(
        "--use-distributed-optimizer",
        action="store_true",
        help="Use distributed optimizer.",
    )
    parser.add_argument("--make_vocab_size_divisible_by", help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.',type=int, default=128)
    parser.add_argument(
        "--overlap_grad_reduce",
        action="store_true",
        default=False,
        help="If set, overlap DDP grad reduce. (Not implement yet)",
    )


def get_collective_test_params(parser: argparse.ArgumentParser):
    parser.add_argument("--begin_size", type=int, default=1048576)
    parser.add_argument("--end_size", type=int, default=1048576)
    parser.add_argument("--test_comm", type=str, default="all_reduce")
    parser.add_argument("--iter_num", type=int, default=500)
    parser.add_argument("--multi_all_reduce_enable", type=int, default=0)


def get_simAI_workload_params(parser: argparse.ArgumentParser):
    parser.add_argument("--overlap_version", action="store_true")

def get_moe_params(parser: argparse.ArgumentParser):
    parser.add_argument('--moe_enable', action="store_true")
    parser.add_argument('--expert_model_parallel_size', type=int, default=1, help='Degree of expert model parallelism.')
    parser.add_argument('--num_experts', type=int, default=1, help='Number of Experts in MoE (None means no MoE)')
    parser.add_argument('--moe_router_topk', type=int, default=1, help='Number of experts to route to for each token. The default is 2.')
    parser.add_argument('--moe_grouped_gemm', action='store_true',
                       help='When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).')
    parser.add_argument('--activation_func', type=str,help='activation_func for mlp')

def get_deepseek_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--n_dense_layers", type=int, default=3, help="Number of dense (non-MoE) layers"
    )
    parser.add_argument(
        "--n_shared_expert",
        type=int,
        default=2,
        help="Number of shared experts for DeepSeek model",
    )
    parser.add_argument(
        "--qk_rope_dim",
        type=int,
        default=64,
        help="Dimention of QK with positional embeddings",
    )
    parser.add_argument(
        "--qk_nope_dim",
        type=int,
        default=128,
        help="Dimention of QK without positional embeddings",
    )
    parser.add_argument(
        "--q_lora_rank", type=int, default=1536, help="Q down projection size"
    )
    parser.add_argument(
        "--kv_lora_rank", type=int, default=512, help="KV down projection size"
    )
    parser.add_argument(
        "--v_head_dim", type=int, default=128, help="Dimention for value projection"
    )

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def get_padded_vocab_size(args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = args.vocab_size

    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while (after % multiple) != 0:

        after += 1

    return after


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def num_parameters_to_bytes(args, params: int) -> str:
    """convert parameters to MBs or GBs"""
    bytes_per_param = 1
    if args.dtype == "bfloat16" or args.dtype == "float16":
        bytes_per_param = 2
    else:
        # default to float32, similart to AiobMegatron
        bytes_per_param = 4
    b = params * bytes_per_param
    gb = b / 1e9
    # if less than 1GB, print MB
    if gb < 0:
        return f"{b/1e6:.2f} MB"
    return f"{gb:.2f} GB"

class Strategy(str, Enum):
    RoundRobin = "RoundRobin"
    UniformRandom = "UniformRandom"

def get_ep_expected_m_per_group(m, num_groups, topk, ep, strategy = "RoundRobin"):
    if strategy == Strategy.RoundRobin:
        return max(round(m * topk / num_groups), 1)
    elif strategy == Strategy.UniformRandom:
        # Simulate Global distribution to find the busiest RANK (Straggler Rank).
        global_total_tokens = m * topk * ep
        global_total_experts = num_groups * ep
        
        rng = np.random.default_rng()
        
        # 1. Distribute tokens to ALL global experts
        all_experts_load = rng.multinomial(global_total_tokens, [1.0/global_total_experts]*global_total_experts)
        
        # 2. Aggregate loads per Rank
        # Reshape to (ep, num_groups) and sum across experts within each rank
        rank_loads = all_experts_load.reshape(ep, num_groups).sum(axis=1)
            
        # 3. Find the busiest Rank
        max_rank_load = rank_loads.max()
        
        # 4. Return average load per expert on the busiest rank
        return int(max_rank_load / num_groups)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")