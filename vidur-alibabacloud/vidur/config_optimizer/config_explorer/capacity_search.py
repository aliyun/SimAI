import argparse
import glob
import os
import platform
import shlex
from subprocess import Popen

import pandas as pd
import ray

from vidur.config_optimizer.config_explorer.config import JobConfig, SimulationConfig
from vidur.config_optimizer.config_explorer.ray_utils import (
    CpuAssignmentManager,
    get_ip,
)
from vidur.logger import init_logger

logger = init_logger(__name__)


class CapacitySearch:
    def __init__(
        self,
        job_config: JobConfig,
        args: argparse.Namespace,
        cpu_core_assignment_manager: CpuAssignmentManager = None,
        cpu_core_id: int = None,
    ):
        self.node_ip = get_ip()
        self.cpu_core_id = None
        self.job_config = job_config
        self.args = args
        self.cpu_core_assignment_manager = cpu_core_assignment_manager
        self.cpu_core_id = cpu_core_id

    def release_cpu_core_id(self):
        if self.cpu_core_id is None:
            return

        ray.get(
            self.cpu_core_assignment_manager.release_cpu_core_id.remote(
                self.node_ip,
                self.cpu_core_id,
            )
        )

    def _generate_run_command(
        self,
        scheduler_config: SimulationConfig,
    ):
        cpu_affinity_command = ""
        if self.cpu_core_id is not None and platform.system() != "Darwin":
            cpu_affinity_command = f"taskset --cpu-list {self.cpu_core_id}"

        command = f"nice -n 1 {cpu_affinity_command} python -m vidur.main {scheduler_config.to_args()}"
        logger.debug(f"Running command: {command}")

        return command

    def _get_result_file(self, run_dir: str) -> str:
        scheduling_delay_file = glob.glob(
            f"{run_dir}/*/plots/request_scheduling_delay.csv"
        )
        if len(scheduling_delay_file) == 0:
            return

        return scheduling_delay_file[0]

    def _is_under_sla(
        self,
        result_file: str,
        simulator_config: SimulationConfig,
    ) -> tuple[bool, float]:
        scheduling_delay_df = pd.read_csv(result_file)
        scheduling_delay = scheduling_delay_df["request_scheduling_delay"].quantile(
            self.args.scheduling_delay_slo_quantile
        )
        is_under_scheduling_delay_sla = (
            scheduling_delay <= self.args.scheduling_delay_slo_value
        )

        logger.info(
            f"{simulator_config.to_human_readable_name()} - Scheduling delay (P{self.args.scheduling_delay_slo_quantile}): {scheduling_delay}",
        )
        return is_under_scheduling_delay_sla, scheduling_delay

    def is_under_sla(self, qps: float) -> tuple[bool, float]:
        simulator_config = SimulationConfig(
            output_dir=self.args.output_dir,
            cache_dir=self.args.cache_dir,
            qps=qps,
            time_limit=self.args.time_limit,
            job_config=self.job_config,
        )
        run_dir = simulator_config.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        cached_result_file = self._get_result_file(run_dir)
        if cached_result_file:
            return self._is_under_sla(cached_result_file, simulator_config)

        command = self._generate_run_command(simulator_config)

        output_file = open(f"{run_dir}/output.log", "w")

        # write command to a file
        output_file.write(f"Running command: {command}\n")

        try:
            args = shlex.split(command)
            p = Popen(args, stdout=output_file, stderr=output_file)
            p.wait()

            result_file = self._get_result_file(run_dir)
            assert (
                result_file is not None
            ), f"Result file not found for {simulator_config.to_human_readable_name()}"
            return self._is_under_sla(result_file, simulator_config)
        except Exception as e:
            logger.error(
                f"Error running: {self.job_config.get_human_readable_name()}, failed with error: {e}",
            )
            return False, None

    def search(self):
        """
        Perform binary search to find the maximum QPS under the SLO
        """
        logger.info(
            f"Starting search for {self.job_config.get_human_readable_name()}",
        )

        left = 0
        right = self.job_config.start_qps * 2
        qps = 0
        max_qps_under_sla = None
        min_qps_over_sla = 2**32

        for _ in range(self.args.max_iterations):
            # stopping condition - we have reached the minimum granularity
            if abs(left - right) < self.args.min_search_granularity * qps / 100:
                break

            qps = (left + right) / 2

            is_under_sla, scheduling_delay = self.is_under_sla(qps)

            if scheduling_delay is None:
                break

            if is_under_sla:
                max_qps_under_sla = qps

                if scheduling_delay < self.args.scheduling_delay_slo_value / 8:
                    # if the scheduling delay is very low, we can increase the QPS more aggressively
                    right = min(right * 4, min_qps_over_sla)
                elif scheduling_delay < self.args.scheduling_delay_slo_value / 4:
                    right = min(right * 2, min_qps_over_sla)
                elif qps > 0.8 * right:
                    right = min(right * 2, min_qps_over_sla)

                left = qps
            else:
                if scheduling_delay > 500:
                    right = qps / 2
                elif scheduling_delay > 1000:
                    right = qps / 4
                else:
                    right = qps

                min_qps_over_sla = min(min_qps_over_sla, qps)

        logger.info(
            f"Max QPS under SLO for {self.job_config.get_human_readable_name()}: {max_qps_under_sla}",
        )

        self.release_cpu_core_id()

        return {
            **self.job_config.to_config_dict(),
            "max_qps_under_sla": max_qps_under_sla,
        }
