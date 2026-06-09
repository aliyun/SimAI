import gc
import time

import numpy as np
from sarathi import LLMEngine, SamplingParams
from sarathi.metrics.constants import CpuOperationMetrics
from tqdm import tqdm

from vidur.logger import init_logger

logger = init_logger(__name__)


NUM_PREFILL_TOKEN = 256
NUM_DECODE_TOKEN_AMPLIFICATION_FACTOR = 3


class BenchmarkRunner:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        tensor_parallel_degree: int,
        output_dir: str,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._tensor_parallel_degree = tensor_parallel_degree
        self._output_dir = output_dir

        self._config_name = f"{model_name}_{batch_size}_{tensor_parallel_degree}"

        self._llm_engine = LLMEngine.from_engine_args(
            replica_id=0,
            # model config
            model=model_name,
            tokenizer=model_name,
            tensor_parallel_size=tensor_parallel_degree,
            dtype="float16",
            load_format="dummy",
            # scheduler config
            scheduler_type="vllm",
            max_num_seqs=batch_size,
            write_metrics=True,
            output_dir=output_dir,
            enable_op_level_metrics=False,
            enable_cpu_op_level_metrics=True,
            keep_individual_batch_metrics=False,
            trust_remote_code=True,
        )

    def _get_input_params(self) -> SamplingParams:
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=self._batch_size * NUM_DECODE_TOKEN_AMPLIFICATION_FACTOR,
        )
        prompt_token_ids = (
            np.random.default_rng()
            .integers(low=0, high=10000, size=NUM_PREFILL_TOKEN)
            .tolist()
        )

        return {
            "prompt": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
        }

    def _warmup(self) -> None:
        # warmup the engine
        self._llm_engine.add_request(**self._get_input_params())

        is_completed = False
        while not is_completed:
            step_outputs = self._llm_engine.step()
            is_completed = step_outputs[0].finished

        self._llm_engine.reset_metrics()

    def run(self):
        self._warmup()

        for _ in range(self._batch_size):
            self._llm_engine.add_request(**self._get_input_params())

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(
            total=self._batch_size, desc=f"{self._config_name} processed requests"
        )

        self._llm_engine.reset_metrics()

        start_time = time.monotonic()

        # Run the engine.
        while num_processed_requests < self._batch_size:
            step_outputs = self._llm_engine.step()
            num_steps += 1

            for output in step_outputs:
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)

        end_time = time.monotonic()
        pbar.close()

        logger.info(
            f"{self._config_name} exiting after processing {self._batch_size} ({num_steps} iterations),"
            f" Total time taken: {end_time - start_time:.2f} seconds"
        )

        self._llm_engine.pull_worker_metrics()

        metric_store = self._llm_engine.get_metric_store()

        metrics_means = {
            f"{k.name.lower()}_mean": v.mean
            for k, v in metric_store.cpu_operation_metrics.items()
        }

        metrics_medians = {
            f"{k.name.lower()}_median": v.median
            for k, v in metric_store.cpu_operation_metrics.items()
        }

        metrics = {**metrics_means, **metrics_medians}

        total_recorded_cpu_time = (
            metric_store.cpu_operation_metrics[CpuOperationMetrics.SCHEDULE].sum
            + metric_store.cpu_operation_metrics[
                CpuOperationMetrics.PROCESS_MODEL_OUTPUTS
            ].sum
            + metric_store.cpu_operation_metrics[
                CpuOperationMetrics.MODEL_EXECUTION_E2E
            ].sum
            + metric_store.cpu_operation_metrics[CpuOperationMetrics.SAMPLER_E2E].sum
            + metric_store.cpu_operation_metrics[
                CpuOperationMetrics.PREPARE_INPUTS_E2E
            ].sum
        )

        total_recorded_cpu_time *= 1e-3  # convert to seconds
        ray_comm_time_mean = (
            (end_time - start_time) - total_recorded_cpu_time
        ) / num_steps
        ray_comm_time_mean *= 1e3  # convert to ms

        metrics.update(
            {
                "model_name": self._model_name,
                "batch_size": self._batch_size,
                "tensor_parallel_degree": self._tensor_parallel_degree,
                "ray_comm_time_mean": ray_comm_time_mean,
            }
        )

        del self._llm_engine
        # trigger garbage collection
        gc.collect()

        return metrics
