import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

from vidur.config_optimizer.analyzer.constants import CPU_MACHINE_COST, GPU_COSTS
from vidur.logger import init_logger

logger = init_logger(__name__)


def extract_stat_from_request_metrics(
    request_metrics_df: pd.DataFrame,
    stat_name: str,
    stat_short_name: str = None,
):
    if stat_short_name is None:
        stat_short_name = stat_name

    stats = request_metrics_df[stat_name].describe().to_dict()
    # add 95th and 99th percentile
    stats["90%"] = request_metrics_df[stat_name].quantile(0.90)
    stats["95%"] = request_metrics_df[stat_name].quantile(0.95)
    stats["99%"] = request_metrics_df[stat_name].quantile(0.99)

    stats_dict = {f"{stat_short_name}_{k}": v for k, v in stats.items()}
    return stats_dict


def extract_stats_from_cdf_df(
    cdf_df: pd.DataFrame,
    stat_name: str,
    stat_short_name: str = None,
    extract_all: bool = False,
):
    if stat_short_name is None:
        stat_short_name = stat_name

    if extract_all:
        cdf_df["cdf_rounded"] = cdf_df["cdf"].round(2)
        cdf_df = cdf_df.drop_duplicates(subset="cdf_rounded", keep="first")
        return {f"{stat_short_name}_cdf": cdf_df[stat_name].tolist()[1:]}

    percentile_map = {
        "min": 0.0,
        "25%": 0.25,
        "50%": 0.5,
        "75%": 0.75,
        "90%": 0.90,
        "95%": 0.95,
        "99%": 0.99,
        "max": 1.0,
    }
    stats = {
        k: cdf_df[cdf_df["cdf"] == v][stat_name].iloc[0]
        for k, v in percentile_map.items()
    }
    stats_dict = {f"{stat_short_name}_{k}": v for k, v in stats.items()}
    return stats_dict


def extract_utilization_stats(run_dir: str, stat_name: str):
    stat_files = glob.glob(f"{run_dir}/plots/replica_*{stat_name}.json")
    vals = []
    for stat_file in stat_files:
        stat = json.load(open(stat_file))
        for k, v in stat.items():
            if k.endswith("weighted_mean"):
                vals.append(v)

    if len(vals) == 0:
        return {f"{stat_name}_mean": np.nan}

    return {f"{stat_name}_mean": sum(vals) / len(vals)}


def process_run(run_dir: str):
    config_file = f"{run_dir}/config.yml"
    request_metrics_file = f"{run_dir}/request_metrics.csv"
    tbt_file = f"{run_dir}/plots/batch_execution_time.csv"
    ttft_file = f"{run_dir}/plots/prefill_e2e_time.csv"
    batch_size_file = f"{run_dir}/plots/batch_size.csv"
    batch_num_tokens_file = f"{run_dir}/plots/batch_num_tokens.csv"
    request_completion_time_series_file = (
        f"{run_dir}/plots/request_completion_time_series.csv"
    )

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        request_metrics_df = pd.read_csv(request_metrics_file)
        tbt_df = pd.read_csv(tbt_file)
        ttft_df = pd.read_csv(ttft_file)
        batch_size_df = pd.read_csv(batch_size_file)
        batch_num_tokens_df = pd.read_csv(batch_num_tokens_file)
        request_completion_time_series_df = pd.read_csv(
            request_completion_time_series_file
        )
    except FileNotFoundError as e:
        # TODO(amey): Add a better error handling approach
        # we can run into this issue if the run was not successful
        # either due to actual failure or due to model OOMing in simulation
        # later is okay, while the former is not
        return None

    request_scheduling_delay_stats = extract_stat_from_request_metrics(
        request_metrics_df, "request_scheduling_delay"
    )
    request_e2e_time_normalized_stats = extract_stat_from_request_metrics(
        request_metrics_df, "request_e2e_time_normalized"
    )
    ttft_stats = extract_stat_from_request_metrics(
        request_metrics_df, "prefill_e2e_time", "ttft"
    )
    ttft_cdf = extract_stats_from_cdf_df(
        ttft_df, "prefill_e2e_time", "ttft", extract_all=True
    )
    tbt_stats = extract_stats_from_cdf_df(tbt_df, "batch_execution_time", "tbt")
    tbt_cdf = extract_stats_from_cdf_df(
        tbt_df, "batch_execution_time", "tbt", extract_all=True
    )
    batch_size_stats = extract_stats_from_cdf_df(batch_size_df, "batch_size")
    batch_size_cdf = extract_stats_from_cdf_df(
        batch_size_df, "batch_size", extract_all=True
    )
    batch_num_tokens_cdf = extract_stats_from_cdf_df(
        batch_num_tokens_df, "batch_num_tokens", extract_all=True
    )
    memory_usage_stats = extract_utilization_stats(run_dir, "memory_usage")
    mfu_stats = extract_utilization_stats(run_dir, "mfu")
    busy_time_percent_stats = extract_utilization_stats(run_dir, "busy_time_percent")
    runtime = request_completion_time_series_df["Time (sec)"].max()

    if (
        config["replica_scheduler_provider"] == "sarathi"
        and config["sarathi_scheduler_chunk_size"] == 4096
    ):
        config["replica_scheduler_provider"] = "orca+"

    config.update(
        {
            **request_scheduling_delay_stats,
            **request_e2e_time_normalized_stats,
            **tbt_stats,
            **ttft_stats,
            **memory_usage_stats,
            **mfu_stats,
            **busy_time_percent_stats,
            **ttft_cdf,
            **tbt_cdf,
            **batch_size_stats,
            **batch_size_cdf,
            **batch_num_tokens_cdf,
            "runtime": runtime,
        }
    )

    return config


def get_sim_time(sim_results_dir: str):
    output_file = f"{sim_results_dir}/output.log"

    with open(output_file, "r") as f:
        lines = f.readlines()

    # search for Simulation took time: xxx
    for line in lines:
        if "Simulation took time" in line:
            return float(line.split(":")[-1].strip())


def process_trace(sim_results_dir: str):
    analysis_dir = f"{sim_results_dir}/analysis"

    # check the results already exists
    if os.path.exists(f"{analysis_dir}/stats.csv") and os.path.exists(
        f"{analysis_dir}/simulation_stats.yml"
    ):
        return

    os.makedirs(analysis_dir, exist_ok=True)

    # the dir structure is sim_results_dir/runs/<config_hash>/<qps>/<date-string>/
    run_dirs = glob.glob(f"{sim_results_dir}/runs/*/*/*/")

    num_cores = os.cpu_count() - 2

    with Pool(num_cores) as p:
        all_results = p.map(process_run, run_dirs)

    # filer out None values
    all_results = [r for r in all_results if r is not None]
    logger.info(f"Total number of runs: {len(run_dirs)} valid runs: {len(all_results)}")

    df = pd.DataFrame(all_results)

    df["num_gpus"] = (
        df["cluster_num_replicas"]
        * df["replica_num_tensor_parallel_workers"]
        * df["replica_num_pipeline_stages"]
    )
    df["cost"] = (
        df["runtime"] * df["num_gpus"] * df["replica_device"].map(GPU_COSTS) / 3600
    )
    df["capacity_per_dollar"] = df["poisson_request_interval_generator_qps"] / (
        df["num_gpus"] * df["replica_device"].map(GPU_COSTS)
    )
    df["gpu_hrs"] = df["runtime"] * df["num_gpus"] / 3600

    df["num_replica_gpus"] = (
        df["replica_num_tensor_parallel_workers"] * df["replica_num_pipeline_stages"]
    )
    df["hour_cost_per_replica"] = (
        df["replica_device"].map(GPU_COSTS) * df["num_replica_gpus"]
    )
    df["capacity_per_replica"] = (
        df["poisson_request_interval_generator_qps"] / df["cluster_num_replicas"]
    )

    # store the df
    df.to_csv(f"{analysis_dir}/stats.csv", index=False)

    gpu_cost = df["cost"].sum()
    total_gpu_hrs = df["gpu_hrs"].sum()

    sim_time = get_sim_time(sim_results_dir)
    cpu_hrs = sim_time / 3600
    cpu_cost = cpu_hrs * CPU_MACHINE_COST

    simulation_stats = {
        "gpu_cost": gpu_cost,
        "sim_cpu_cost": cpu_cost,
        "total_gpu_hrs": total_gpu_hrs,
        "sim_time": sim_time,
        "total_runs": len(run_dirs),
        "valid_runs": len(all_results),
    }

    json.dump(
        simulation_stats, open(f"{analysis_dir}/simulation_stats.json", "w"), indent=4
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-results-dir", type=str, required=True)
    args = parser.parse_args()

    process_trace(args.sim_results_dir)


if __name__ == "__main__":
    main()
