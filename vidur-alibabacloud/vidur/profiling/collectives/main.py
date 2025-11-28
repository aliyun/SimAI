import argparse
import datetime
import os

import pandas as pd
import ray
from tqdm import tqdm

from vidur.logger import init_logger
from vidur.profiling.collectives.benchmark_runner import BenchmarkRunner
from vidur.profiling.utils import get_collectives_inputs

logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MLP Profiling")
    parser.add_argument(
        "--num_workers_per_node_combinations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--max_collective_size",
        type=int,
        default=4096 * 8192,
        help="Maximum number of elements involved in the collective",
    )
    parser.add_argument(
        "--collective",
        default="all_reduce",
        # choices=["all_reduce", "send_recv"],
        # > add
        choices=["all_reduce", "send_recv","all_gather","reduce_scatter","all_to_all","broadcast"],
        help="Collective to profile",
    )
    args = parser.parse_args()

    # 比如: args.output_dir = 'profiling_outputs/collective/2025-10-15_08-22-57'
    args.output_dir = f"{args.output_dir}/collective/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def create_runner_pool():
    total_gpus_available = int(ray.cluster_resources()["GPU"])
    logger.info(f"Total GPUs available: {total_gpus_available}")

    assert total_gpus_available > 0, "No GPUs available"

    all_node_ips = [x["NodeName"] for x in ray.nodes()]
    logger.info(f"All node IPs: {all_node_ips}")

    assert len(all_node_ips) > 0, "No nodes available"

    num_nodes = len(all_node_ips)
    gpus_per_node = total_gpus_available // len(all_node_ips)

    runner_pool = []
    for gpu_id in range(total_gpus_available):
        node_ip = all_node_ips[gpu_id // gpus_per_node]
        runner_pool.append(
            BenchmarkRunner.options(
                resources={
                    f"node:{node_ip}": 0.01,
                }
            ).remote(gpu_id, gpus_per_node, all_node_ips[0])
        )
    return total_gpus_available, num_nodes, runner_pool


def main():
    args = parse_args()

    ray.init()

    total_gpus_available, num_nodes, runner_pool = create_runner_pool()

    all_results = []

    collectives_inputs = get_collectives_inputs(
        num_nodes,
        args.num_workers_per_node_combinations,
        args.max_collective_size,
        args.collective,
        total_gpus_available,
    )

    for collectives_input in tqdm(collectives_inputs):
        promises = []
        for gpu_id in range(total_gpus_available):
            promise = runner_pool[gpu_id].run_collective.remote(collectives_input)
            promises.append(promise)

        for gpu_id in range(int(total_gpus_available)):
            result = ray.get([promises[gpu_id]])[0]
            if result and gpu_id == 0:
                all_results.append(result)

        ray.get(promises)

    # filter none results
    all_results = [x for x in all_results if x is not None]

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix

    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    # write results to a csv file
    df.to_csv(f"{args.output_dir}/{args.collective}.csv")


if __name__ == "__main__":
    main()
