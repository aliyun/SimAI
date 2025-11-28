import argparse
import copy

import ray

from vidur.config_optimizer.config_explorer.capacity_search import CapacitySearch
from vidur.config_optimizer.config_explorer.config import JobConfig
from vidur.config_optimizer.config_explorer.ray_utils import (
    CpuAssignmentManager,
    RayParallelRunner,
    run_on_each_node,
)


def run_search(
    job_config: JobConfig,
    args: argparse.Namespace,
    cpu_core_assignment_manager: CpuAssignmentManager = None,
    cpu_core_id: int = None,
):
    capacity_search = CapacitySearch(
        job_config,
        args,
        cpu_core_assignment_manager,
        cpu_core_id,
    )
    return capacity_search.search()


class ConfigExplorer:
    def __init__(
        self,
        args: argparse.Namespace,
        config: dict,
    ):
        self.args = args
        self.config = config

        ray.init(ignore_reinit_error=True)

    def _warmup_cache(self):
        job_configs = JobConfig.generate_unique_model_job_configs(self.config)

        args_for_warmup = copy.deepcopy(self.args)
        args_for_warmup.max_iterations = 1

        for job_config in job_configs:
            all_node_results = run_on_each_node(
                run_search,
                job_config,
                args_for_warmup,
            )
            assert all(all_node_results) or not any(
                all_node_results
            ), "All nodes should have the same result"

    def run(self):
        if not self.args.skip_cache_warmup:
            self._warmup_cache()

        job_configs = JobConfig.generate_job_configs(self.config)

        ray_parallel_runner = RayParallelRunner()

        remote_func = (
            lambda cpu_core_assignment_manager, cpu_core_id, job_config: run_search(
                job_config,
                self.args,
                cpu_core_assignment_manager,
                cpu_core_id,
            )
        )
        all_results = ray_parallel_runner.map(
            remote_func,
            job_configs,
        )
        return all_results
