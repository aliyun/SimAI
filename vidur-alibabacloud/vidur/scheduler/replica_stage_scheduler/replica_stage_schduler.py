from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.execution_time_predictor import BaseExecutionTimePredictor


class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor

        # 在这个PP stage上等待执行的micro-batch队列
        # Queue of micro-batches waiting for execution on this PP stage
        self._batch_queue = []
        self._is_busy = False

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    def on_stage_end(self) -> None:
        self._is_busy = False

    # sw： p batch：time；      flow：time；     d batch： time
    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        if self._is_busy or not self._batch_queue:
            return None, None, None

        self._is_busy = True
        batch = self._batch_queue.pop(0)
        # 模拟micro-batch在PP stage上的执行
        # TODO: 这块接入simai
        
        # Simulate micro-batch execution on PP stage
        # TODO: Integrate with simai
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        

        # >: total_time =  self.model_time + self._get_cpu_overhead() * 1e-3
    
        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )

        return batch, batch_stage, execution_time
