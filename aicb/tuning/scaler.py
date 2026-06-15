"""F003: Workload Scaler.

Post-generation parallelism reconfiguration. Changes DP/TP/PP
dimensions without regenerating layer structure, preserving total
FLOP count while recalculating communication sizes and participant
groups.

Key invariant: total_gpus = new_dp * new_tp * new_pp
"""

import math
from typing import List, Optional
from log_analyzer.log import LogItem, CommType, CommGroup


class WorkloadScaler:
    """Reconfigures parallelism dimensions on an existing workload.

    Recalculates communication topology (comm_group, comm_group_size,
    msg_size) for changed DP/TP/PP dimensions. Preserves compute ops
    unchanged (same FLOPs per layer regardless of parallelism).
    """

    def __init__(self, log_items: List[LogItem]):
        self._items = log_items
        self._original_config: Optional[dict] = None

    def scale(
        self,
        new_dp: Optional[int] = None,
        new_tp: Optional[int] = None,
        new_pp: Optional[int] = None,
        total_gpus: Optional[int] = None,
    ) -> List[LogItem]:
        """Apply parallelism scaling.

        At least one of new_dp, new_tp, new_pp must be provided.
        Missing dimensions are derived from total_gpus constraint.

        Args:
            new_dp: New data-parallel size.
            new_tp: New tensor-parallel size.
            new_pp: New pipeline-parallel size.
            total_gpus: Total GPU count override.

        Returns:
            The modified LogItem list.
        """
        # Detect original parallelism dimensions from comm groups
        orig = self._detect_original_parallelism()
        if orig is None:
            return self._items  # Cannot detect, skip scaling

        # Fill missing dimensions
        dp = new_dp if new_dp is not None else orig["dp"]
        tp = new_tp if new_tp is not None else orig["tp"]
        pp = new_pp if new_pp is not None else orig["pp"]
        gpus = total_gpus if total_gpus is not None else dp * tp * pp

        if gpus != dp * tp * pp:
            raise ValueError(
                f"total_gpus ({gpus}) != dp*tp*pp ({dp}*{tp}*{pp} = {dp*tp*pp})"
            )

        if dp == orig["dp"] and tp == orig["tp"] and pp == orig["pp"]:
            return self._items  # No change needed

        # Recalculate communication topology for each item
        for item in self._items:
            self._rescale_item(item, orig, dp, tp, pp, gpus)

        return self._items

    def _detect_original_parallelism(self) -> Optional[dict]:
        """Detect original DP/TP/PP from LogItem comm groups."""
        dp_sizes = set()
        tp_sizes = set()

        for item in self._items:
            if item.comm_group == CommGroup.dp:
                dp_sizes.add(item.comm_group_size)
            elif item.comm_group in (
                CommGroup.tp,
                CommGroup.tp_pp,
            ):
                tp_sizes.add(item.comm_group_size)

        # Heuristic: largest DP group is the DP size, largest TP group is TP size
        dp = max(dp_sizes) if dp_sizes else 1
        tp = max(tp_sizes) if tp_sizes else 1

        if dp == 1 and tp == 1:
            return None

        return {"dp": dp, "tp": tp, "pp": 1}

    def _rescale_item(
        self,
        item: LogItem,
        orig: dict,
        dp: int,
        tp: int,
        pp: int,
        gpus: int,
    ):
        """Recalculate communication parameters for a single LogItem."""
        orig_dp = orig["dp"]
        orig_tp = orig["tp"]

        # Compute scaling ratios
        tp_ratio = tp / orig_tp
        dp_ratio = dp / orig_dp

        # TP communication: msg_size scales inversely with tp_size
        # (more GPUs in TP group = smaller per-GPU message)
        if item.comm_group in (CommGroup.tp, CommGroup.tp_pp):
            item.comm_group_size = tp
            item.msg_size = item.msg_size / tp_ratio
            # More TP groups across the cluster = more TP collectives
            # but each is smaller. Recalculate elapsed_time via bandwidth.
            if item.elapsed_time is not None:
                item.elapsed_time = item.elapsed_time / tp_ratio

        # DP communication: group_size changes, msg_size is unchanged
        elif item.comm_group == CommGroup.dp:
            item.comm_group_size = dp
            # DP AllReduce: same msg_size (gradients), different group
            if item.elapsed_time is not None:
                # DP communication time scales with log2(dp) for ring
                item.elapsed_time = item.elapsed_time * (
                    math.log2(max(dp, 2)) / math.log2(max(orig_dp, 2))
                )

        # DP+TP combined: update both
        if item.comm_group == CommGroup.dp:
            parts = item.stage.split("_")
            for i, part in enumerate(parts):
                if part.startswith("dp"):
                    parts[i] = f"dp{dp}"
                elif part.startswith("tp"):
                    parts[i] = f"tp{tp}"
            item.stage = "_".join(parts)
