"""AICB Tunability Package (F003).

Decorator-pattern wrappers for injecting performance variability
into generated workloads without modifying core mocked_model logic.

Components:
- TunabilityWrapper: orchestrates straggler + scaling + variability
- WorkloadScaler: post-generation DP/TP dimension reconfiguration
- VariabilityInjector: Monte Carlo noise on operation durations

Reference: .omc/research/aicb-model-extensibility.md, Section 4.4 (F003).
"""

from .wrapper import TunabilityWrapper
from .scaler import WorkloadScaler
from .variability import VariabilityInjector

__all__ = ["TunabilityWrapper", "WorkloadScaler", "VariabilityInjector"]
