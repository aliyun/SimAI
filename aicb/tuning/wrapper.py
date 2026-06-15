"""F003: Tunability Wrapper (Decorator Pattern).

Orchestrates straggler injection, workload scaling, and variability
modeling as a decorator layer between mocked_model output and the
workload writer. No changes to core mocked_model logic.

Design: Non-invasive wrapper that iterates over LogItems after
workload generation. Applies tunability transforms in order:
  1. Scale (recalculate parallelism dimensions)
  2. Inject stragglers (probabilistic delays)
  3. Add variability (duration noise)

Reference: MLSynth Performance Wrapper pattern (NAIC '25).
"""

import random
from typing import List, Optional
from log_analyzer.log import LogItem, CommType, CommGroup


class TunabilityWrapper:
    """Decorator-pattern tunability orchestrator.

    Wraps a list of LogItems (generated workload) and applies
    straggler injection, workload scaling, and variability modeling.

    Usage:
        items = generate_workload(...)
        wrapper = TunabilityWrapper(items)
        wrapper.inject_stragglers(gpu_rate=0.02, nic_rate=0.01)
        wrapper.add_variability(std_ratio=0.05)
        wrapper.apply()
        # items now have straggler delays and duration variability
    """

    def __init__(self, log_items: List[LogItem]):
        self._items = log_items
        self._straggler_config: Optional[dict] = None
        self._variability_config: Optional[dict] = None

    # ------------------------------------------------------------------
    # Straggler Injection
    # ------------------------------------------------------------------

    def inject_stragglers(
        self,
        gpu_rate: float = 0.0,
        nic_rate: float = 0.0,
        gpu_delay_us: float = 100.0,
        nic_delay_us: float = 50.0,
        seed: Optional[int] = None,
    ) -> "TunabilityWrapper":
        """Configure probabilistic straggler delays.

        Args:
            gpu_rate: Probability [0,1] that a compute op is delayed.
            nic_rate: Probability [0,1] that a comm op is delayed.
            gpu_delay_us: Base delay for compute stragglers (microseconds).
            nic_delay_us: Base delay for comm stragglers (microseconds).
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self._straggler_config = {
            "gpu_rate": gpu_rate,
            "nic_rate": nic_rate,
            "gpu_delay_us": gpu_delay_us,
            "nic_delay_us": nic_delay_us,
        }
        return self

    # ------------------------------------------------------------------
    # Variability Modeling
    # ------------------------------------------------------------------

    def add_variability(
        self,
        std_ratio: float = 0.0,
        seed: Optional[int] = None,
    ) -> "TunabilityWrapper":
        """Configure Monte Carlo duration noise.

        Args:
            std_ratio: Std dev as fraction of elapsed_time (e.g., 0.05 = 5%).
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self._variability_config = {
            "std_ratio": std_ratio,
        }
        return self

    # ------------------------------------------------------------------
    # Apply all configured transforms
    # ------------------------------------------------------------------

    def apply(self) -> List[LogItem]:
        """Apply all configured tunability transforms in order.

        Returns:
            The modified LogItem list (same reference as input).
        """
        # 1. Straggler injection
        if self._straggler_config:
            self._apply_stragglers()

        # 2. Variability noise
        if self._variability_config:
            self._apply_variability()

        return self._items

    def _apply_stragglers(self):
        cfg = self._straggler_config
        for item in self._items:
            if item.elapsed_time is None:
                continue

            # Compute ops (comp_type) vs communication ops
            if item.comm_type == CommType.computation:
                if random.random() < cfg["gpu_rate"]:
                    delay = cfg["gpu_delay_us"] * random.uniform(0.5, 2.0)
                    item.straggler_delay_us = delay
            elif item.comm_type not in (
                CommType.computation,
                CommType.epoch_begin,
                CommType.epoch_end,
            ):
                if random.random() < cfg["nic_rate"]:
                    delay = cfg["nic_delay_us"] * random.uniform(0.5, 2.0)
                    item.straggler_delay_us = delay

    def _apply_variability(self):
        cfg = self._variability_config
        for item in self._items:
            if item.elapsed_time is not None and item.elapsed_time > 0:
                noise = random.gauss(0, cfg["std_ratio"])
                item.elapsed_time = max(
                    0,
                    item.elapsed_time * (1.0 + noise),
                )

    # ------------------------------------------------------------------
    # Chained scaling support (delegates to WorkloadScaler)
    # ------------------------------------------------------------------

    def scale(
        self,
        new_dp: Optional[int] = None,
        new_tp: Optional[int] = None,
        new_pp: Optional[int] = None,
        total_gpus: Optional[int] = None,
    ) -> "TunabilityWrapper":
        """Scale workload to new parallelism dimensions.

        Delegates to WorkloadScaler. Changes communication topology
        without regenerating layer structure.

        Args:
            new_dp: New data-parallel size.
            new_tp: New tensor-parallel size.
            new_pp: New pipeline-parallel size.
            total_gpus: Total GPU count (auto-computed if not given).
        """
        from .scaler import WorkloadScaler

        scaler = WorkloadScaler(self._items)
        scaler.scale(
            new_dp=new_dp,
            new_tp=new_tp,
            new_pp=new_pp,
            total_gpus=total_gpus,
        )
        return self
