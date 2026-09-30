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
from workload_generator.mocked_model.training.MockedDeepspeed import DeepspeedForCausalLM
from workload_generator.mocked_model.training.MockedDeepSeek import DeepSeekV3Model
from workload_generator.mocked_model.training.MockedMegatron import MegatronModel
from workload_generator.generate_deepspeed_stage1_2_workload import (
    DeepSpeedStage1,
    DeepSpeedStage2,
)
from workload_generator.generate_deepspeed_stage3_workload import DeepSpeedStage3
from workload_generator.generate_megatron_workload import MegatronWorkload
from workload_generator.generate_collective_test import Collective_Test
from workload_applyer import WorkloadApplyer
from utils.utils import *

if __name__ == "__main__":
    args = get_args()
    if not hasattr(args, "backend"):
        args.backend = "nccl"
    torch.distributed.init_process_group(backend=args.backend)
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
    if args.frame == "Megatron":
        model = MegatronModel(args)
        workload_generator = MegatronWorkload(args, model)
    elif args.frame == "DeepSpeed":
        model = DeepspeedForCausalLM(args)
        if args.stage == 1:
            workload_generator = DeepSpeedStage1(args, model)
        elif args.stage == 2:
            workload_generator = DeepSpeedStage2(args, model)
        elif args.stage == 3:
            workload_generator = DeepSpeedStage3(args, model)
    elif args.frame == "collective_test":
        workload_generator = Collective_Test(args, None)
    elif args.frame == "DeepSeek":
        model = DeepSeekV3Model(args)
        workload_generator = MegatronWorkload(args, model)
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
