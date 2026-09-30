"""F003: Variability Injector.

Standalone Monte Carlo noise injector for LogItem durations.
Can be used independently of TunabilityWrapper for targeted
variability studies.

Supports:
- Gaussian noise: item.elapsed_time *= (1 + N(0, std_ratio))
- Uniform noise: item.elapsed_time *= (1 + U(-range, +range))
- Per-op-type filtering (compute-only, comm-only, all)
"""

import random
from typing import List, Optional
from log_analyzer.log import LogItem, CommType


class VariabilityInjector:
    """Adds controlled noise to operation durations.

    Usage:
        injector = VariabilityInjector(log_items)
        injector.apply_gaussian(std_ratio=0.05, seed=42)
        # All op durations now have 5% std Gaussian noise.
    """

    def __init__(self, log_items: List[LogItem]):
        self._items = log_items

    def apply_gaussian(
        self,
        std_ratio: float = 0.05,
        op_filter: str = "all",
        seed: Optional[int] = None,
        min_duration_us: float = 1.0,
    ) -> List[LogItem]:
        """Apply Gaussian noise: duration *= (1 + N(0, std_ratio)).

        Args:
            std_ratio: Std dev as fraction of the duration.
            op_filter: "all", "compute", or "comm".
            seed: Random seed for reproducibility.
            min_duration_us: Floor duration in microseconds.

        Returns:
            The modified LogItem list.
        """
        if seed is not None:
            random.seed(seed)

        for item in self._items:
            if not self._matches_filter(item, op_filter):
                continue
            if item.elapsed_time is None or item.elapsed_time <= 0:
                continue

            noise = random.gauss(0.0, std_ratio)
            item.elapsed_time = max(
                min_duration_us,
                item.elapsed_time * (1.0 + noise),
            )

        return self._items

    def apply_uniform(
        self,
        range_pct: float = 0.10,
        op_filter: str = "all",
        seed: Optional[int] = None,
        min_duration_us: float = 1.0,
    ) -> List[LogItem]:
        """Apply uniform noise: duration *= (1 + U(-range, +range)).

        Args:
            range_pct: Maximum deviation as fraction (e.g., 0.10 = +/-10%).
            op_filter: "all", "compute", or "comm".
            seed: Random seed for reproducibility.
            min_duration_us: Floor duration in microseconds.

        Returns:
            The modified LogItem list.
        """
        if seed is not None:
            random.seed(seed)

        for item in self._items:
            if not self._matches_filter(item, op_filter):
                continue
            if item.elapsed_time is None or item.elapsed_time <= 0:
                continue

            noise = random.uniform(-range_pct, +range_pct)
            item.elapsed_time = max(
                min_duration_us,
                item.elapsed_time * (1.0 + noise),
            )

        return self._items

    def _matches_filter(self, item: LogItem, op_filter: str) -> bool:
        if op_filter == "all":
            return True
        is_compute = item.comm_type == CommType.computation
        if op_filter == "compute":
            return is_compute
        if op_filter == "comm":
            return not is_compute and item.comm_type not in (
                CommType.epoch_begin,
                CommType.epoch_end,
            )
        return True
