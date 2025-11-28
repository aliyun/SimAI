import argparse
import datetime
import gc
import os
from itertools import product
from typing import Any, List

import pandas as pd
import ray
from tqdm import tqdm

from vidur.logger import init_logger
from vidur.profiling.cpu_overhead.benchmark_runner import BenchmarkRunner
from vidur.profiling.utils import get_cpu_overhead_batch_sizes_to_profile, hex_to_binary

logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CPU Overhead Profiling")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum batch size to profile",
    )
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/cpu_overhead/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    model_name: str,
    batch_sizes_to_profile: List[int],
    tensor_parallel_degrees: List[int],
    output_dir: str,
    pbar: Any,
) -> dict:
    results = []

    for tensor_parallel_degree in tensor_parallel_degrees:
        for batch_index, batch_size in enumerate(batch_sizes_to_profile):
            try:
                runner = create_runner(
                    model_name, batch_size, tensor_parallel_degree, output_dir
                )
                results.append(ray.get(runner.run.remote()))
                del runner
                # trigger garbage collection
                gc.collect()
            except Exception as e:
                logger.error(
                    f"Failed to run {model_name}_{batch_size}_{tensor_parallel_degree} due to {e}"
                )
                # update progress bar
                pbar.update(len(batch_sizes_to_profile) - batch_index)
                break

            pbar.update(1)

    df = pd.DataFrame(results)
    os.makedirs("f{output_dir}/{model_name}", exist_ok=True)
    df.to_csv(f"{output_dir}/{model_name}/cpu_overhead.csv")


def create_runner(
    model_name: str, batch_size: int, tensor_parallel_degree: int, output_dir: str
) -> BenchmarkRunner:
    placement_group_ids = list(ray.util.placement_group_table().keys())
    for placement_group_id in placement_group_ids:
        ray._private.worker.global_worker.core_worker.remove_placement_group(
            ray.PlacementGroupID(hex_to_binary(placement_group_id))
        )

    runner_class = (
        ray.remote(num_gpus=0)(BenchmarkRunner)
        .options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})
        .remote
    )

    return runner_class(model_name, batch_size, tensor_parallel_degree, output_dir)


def main():
    args = parse_args()

    batch_sizes_to_profile = get_cpu_overhead_batch_sizes_to_profile(
        args.max_batch_size
    )

    input_combos = product(
        args.models, args.num_tensor_parallel_workers, batch_sizes_to_profile
    )

    pbar = tqdm(total=len(list(input_combos)))

    for model_name in args.models:
        profile_model(
            model_name,
            batch_sizes_to_profile,
            args.num_tensor_parallel_workers,
            args.output_dir,
            pbar,
        )


if __name__ == "__main__":
    main()
