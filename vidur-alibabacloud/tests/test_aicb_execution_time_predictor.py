"""Regression tests for the AICB execution-time predictor."""

from types import SimpleNamespace

from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


def test_aicb_compute_predictions_include_kv_cache_save():
    """AICB batches must be able to account for KV-cache-save execution time.

    This guards the failure reported in SimAI issue #266, where the AICB
    prediction table omitted the key consumed by
    ``_get_attention_kv_cache_save_execution_time``.
    """
    predictor = object.__new__(SklearnExecutionTimePredictor)
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        tensor_parallel_size=1,
    )
    predictor._max_tokens = 4

    predictions = predictor._predict_for_compute_models_by_aicb()

    assert "attn_kv_cache_save" in predictions
    assert set(predictions["attn_kv_cache_save"]) == {
        (1,),
        (2,),
        (3,),
        (4,),
    }
    assert all(
        prediction >= 0
        for prediction in predictions["attn_kv_cache_save"].values()
    )
