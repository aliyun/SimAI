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
import torch
from utils.utils import get_args, get_comp_out, extract_averages, Comp_with_aiob
from utils.benchmark_logger import bench_logger
from workload_applyer import WorkloadApplyer
from utils.utils import *

if __name__ == "__main__":
    # Import bootstrap FIRST to populate MODEL_REGISTRY before arg parsing.
    # This must happen before get_args() so that --frame choices are dynamic.
    import workload_generator._bootstrap as _  # noqa: F401 -- side-effect import
    from workload_generator.registry import lookup

    args = get_args()
    if not hasattr(args, "backend"):
        args.backend = "nccl"
    torch.distributed.init_process_group(backend=args.backend)
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

    entry = lookup(args.frame)
    if entry.model_cls is not None:
        model = entry.model_cls(args)
    else:
        model = None
    workload_generator = entry.wl_cls(args, model)

    workload = workload_generator()
    if args.aiob_enable and (args.frame == "Megatron" or args.frame == "DeepSeek"):

        params = model.parameters()
        args.model_param = sum(p.numel() for p in params)
        args.activation_memory = 0
        for sub_module in model.child_modules():
            if hasattr(sub_module, "activation_memory"):
                args.activation_memory += sub_module.activation_memory()
        print("model_param:", args.model_param)
        if args.comp_filepath == None:
            local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
            if local_rank == 0:
                filepath = get_comp_out(args)
            else:
                filepath = get_aiob_path(args)
            torch.distributed.barrier()
            compute_cache = extract_averages(filepath,args)
        else:
            print("comp_filepath:", args.comp_filepath)
            compute_cache = extract_averages(args.comp_filepath,args)
        workload = Comp_with_aiob(workload, compute_cache)
    if torch.distributed.get_rank() == 0:
        filename = f"{workload_generator.name}_{args.model_name}_sp_{args.enable_sequence_parallel}_iteration_{args.epoch_num}_computationEnable_{args.computation_enable}_{args.world_size}n.csv"
        workload.dump(filename)
    if not args.workload_only :
        applyer = WorkloadApplyer(workload=workload, args=args)
        cpu_time = applyer.apply_workload()
        if torch.distributed.get_rank() == 0:
            bench_logger.analyze_comm_log()
            if args.frame != "collective_test":
                bench_logger.analyze_comm_time()
            csv_filename = bench_logger.dump_log(filename)
            if args.enable_visual:
                try:
                    from visualize.generate import visualize_output
                    visualize_output(csv_filename,False)
                except ImportError:
                    print("visualize_output is not available because required library is not found")

            print(
                f"total time for {args.frame} and {args.epoch_num} iterations is {cpu_time:.4f} s"
            )
