"""
Model Registry for AICB Workload Generator.

Provides a centralized registry that maps frame names (e.g. "Megatron", "LLaMA")
to ModelEntry records containing the model class, workload generator class, and
a human-readable description. This replaces the hardcoded if/elif chain in aicb.py,
enabling new model frameworks to be added without modifying the entry point.

Usage:
    # In a MockedModel file (e.g., MockedLlama.py):
    from workload_generator.registry import register_model
    register_model("LLaMA", LlamaModel, MegatronWorkload, "LLaMA decoder-only model")

    # In aicb.py:
    from workload_generator.registry import lookup, model_registry
    entry = lookup(args.frame)
    model = entry.model_cls(args)

    # In utils.py (for dynamic --frame choices):
    from workload_generator.registry import get_registered_frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ModelEntry:
    """Describes a registered model framework.

    Attributes:
        name: Frame name used as --frame argument value (e.g. "Megatron", "LLaMA").
        model_cls: Callable that constructs the MockedModel instance.
                   May be None for frameworks that don't require a model (e.g. collective_test).
        wl_cls: WorkloadGenerator class, or a callable factory that returns one.
                For DeepSpeed, this is a factory that inspects args.stage.
        description: Human-readable description of the framework.
    """

    name: str
    model_cls: Optional[Callable[..., Any]]
    wl_cls: Callable[..., Any]
    description: str = ""


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

model_registry: Dict[str, ModelEntry] = {}


def register_model(
    name: str,
    model_cls: Optional[Callable[..., Any]],
    wl_cls: Callable[..., Any],
    description: str = "",
) -> None:
    """Register a model framework in the global registry.

    Args:
        name: Frame name. Must be unique across all registered models.
        model_cls: Model class (or None if the framework has no model, e.g. collective_test).
        wl_cls: WorkloadGenerator class or factory callable(args, model) -> WorkloadGenerator.
        description: Short human-readable description.

    Raises:
        ValueError: If a model with the same name is already registered.
    """
    if name in model_registry:
        raise ValueError(
            f"Model frame '{name}' is already registered. "
            f"Existing: {model_registry[name].description}"
        )
    model_registry[name] = ModelEntry(
        name=name,
        model_cls=model_cls,
        wl_cls=wl_cls,
        description=description,
    )


def lookup(name: str) -> ModelEntry:
    """Look up a registered model by frame name.

    Args:
        name: The frame name (--frame argument value).

    Returns:
        ModelEntry with model_cls and wl_cls.

    Raises:
        KeyError: If the frame name is not registered, with a message listing
                  all available frame names.
    """
    if name not in model_registry:
        available = get_registered_frames()
        raise KeyError(
            f"Unknown frame '{name}'. "
            f"Available frames: {', '.join(available) if available else '(none registered)'}"
        )
    return model_registry[name]


def get_registered_frames() -> List[str]:
    """Return a sorted list of all registered frame names."""
    return sorted(model_registry.keys())


def is_registered(name: str) -> bool:
    """Return True if a model with the given frame name is registered."""
    return name in model_registry


def clear_registry() -> None:
    """Clear all entries from the registry. Intended for testing only."""
    model_registry.clear()
