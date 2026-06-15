"""
Model registration bootstrap for AICB.

This module imports all known model classes and workload generators,
then registers them in the global MODEL_REGISTRY. It is imported by
aicb.py once at startup to populate the registry before argument parsing.

WHY THIS FILE EXISTS (circular-import avoidance):
    MockedMegatron.py <--> generate_megatron_workload.py would form a
    circular import if registration calls were placed in either file.
    This module sits above both and nothing imports from it, so each
    import resolves cleanly.

To add a new model framework:
    1. Import the model class and workload generator.
    2. Call register_model() with the frame name.
    3. Done -- aicb.py and utils.py pick it up automatically.
"""

from workload_generator.registry import register_model

# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------
from workload_generator.mocked_model.training.MockedMegatron import MegatronModel
from workload_generator.mocked_model.training.MockedDeepSeek import DeepSeekV3Model
from workload_generator.mocked_model.training.MockedDeepspeed import DeepspeedForCausalLM
from workload_generator.mocked_model.training.MockedLlama import LlamaModel

# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------
from workload_generator.generate_megatron_workload import MegatronWorkload
from workload_generator.generate_deepspeed_stage1_2_workload import (
    DeepSpeedStage1,
    DeepSpeedStage2,
)
from workload_generator.generate_deepspeed_stage3_workload import DeepSpeedStage3
from workload_generator.generate_collective_test import Collective_Test


# ---------------------------------------------------------------------------
# DeepSpeed workload-generator factory
# ---------------------------------------------------------------------------

def _deepspeed_wl_factory(args, model):
    """Select the correct DeepSpeed workload generator based on args.stage.

    This factory decouples the registry (which stores a single wl_cls per
    frame) from the stage-dependent dispatch that was previously hardcoded
    in aicb.py.
    """
    stage = getattr(args, "stage", 3)
    if stage == 1:
        return DeepSpeedStage1(args, model)
    elif stage == 2:
        return DeepSpeedStage2(args, model)
    elif stage == 3:
        return DeepSpeedStage3(args, model)
    else:
        raise ValueError(
            f"Unknown DeepSpeed stage: {stage}. Must be 1, 2, or 3."
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_model(
    "Megatron",
    MegatronModel,
    MegatronWorkload,
    "Megatron-LM training workload (TP/PP/DP/EP with sequence parallelism)",
)

register_model(
    "DeepSpeed",
    DeepspeedForCausalLM,
    _deepspeed_wl_factory,
    "DeepSpeed ZeRO stages 1-3 training workload",
)

register_model(
    "DeepSeek",
    DeepSeekV3Model,
    MegatronWorkload,
    "DeepSeek-V3 training workload (MLA + DeepSeekMoE + FP8)",
)

register_model(
    "LLaMA",
    LlamaModel,
    MegatronWorkload,
    "LLaMA 2/3/4 training workload (GQA + SwiGLU + RMSNorm pre-norm)",
)

register_model(
    "collective_test",
    None,
    Collective_Test,
    "Collective communication micro-benchmark patterns",
)
