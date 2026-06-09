from sklearn.ensemble import RandomForestRegressor

from vidur.config import (
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)

# > add simulation_config=self._config
class RandomForrestExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: RandomForrestExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        # will trigger model training
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
            simulation_config=simulation_config,
        )
        # self.simulation_config = simulation_config

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }

    def _get_estimator(self):
        return RandomForestRegressor()
