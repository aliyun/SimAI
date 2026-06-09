"""
    Automated search for capacity for different systems via latency vs qps data.
    A system is characterised by:
    1. trace
    2. model
    3. sku
    4. scheduler
"""

import argparse
import json
import os
import time

import yaml

from vidur.config_optimizer.config_explorer.config_explorer import ConfigExplorer
from vidur.logger import init_logger

logger = init_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--min-search-granularity",
        type=float,
        default=2.5,
        help="Minimum search granularity for capacity (%)",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache_tmpfs")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--scheduling-delay-slo-value", type=float, default=5.0)
    parser.add_argument("--scheduling-delay-slo-quantile", type=float, default=0.99)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--time-limit", type=int, default=30, help="Time limit in minutes"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-cache-warmup", action="store_true")

    args = parser.parse_args()

    default_num_threads = os.cpu_count() - 2
    if args.num_threads is not None:
        args.num_threads = min(args.num_threads, default_num_threads)
    else:
        args.num_threads = default_num_threads

    return args


if __name__ == "__main__":
    args = get_args()

    config = yaml.safe_load(open(args.config_path))

    assert (
        args.scheduling_delay_slo_quantile >= 0
        and args.scheduling_delay_slo_quantile <= 1
    )

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting config optimizer")
    logger.info(f"Args: {args}")
    logger.info(f"Config: {config}")

    # store the config and args
    json.dump(vars(args), open(f"{args.output_dir}/args.json", "w"))
    json.dump(config, open(f"{args.output_dir}/config.json", "w"))

    multiple_capacity_search = ConfigExplorer(args, config)

    start_time = time.time()

    all_results = multiple_capacity_search.run()

    end_time = time.time()

    logger.info(f"Simulation took time: {end_time - start_time}")
