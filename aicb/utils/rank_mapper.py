"""CommGroup-to-RankGenerator mapping utility for communication domain (rank) information.

This module bridges the CommGroup enum values to RankGenerator.get_ranks() tokens,
providing the actual GPU rank IDs participating in each communication operation.
"""

import logging
from typing import List, Dict

from utils.utils import CommGroup, RankGenerator

logger = logging.getLogger(__name__)

_COMM_GROUP_TOKEN_MAP = {
    CommGroup.tp_group:      ("tp",     False),
    CommGroup.pp_group:      ("pp",     False),
    CommGroup.dp_group:      ("dp",     False),
    CommGroup.ep_group:      ("ep",     True),
    CommGroup.ep_dp_group:   ("ep-dp",  True),
    CommGroup.ep_tp_group:   ("ep-tp",  True),
    CommGroup.embedding_group: ("tp",   False),
}


def get_rank_list_for_comm_group(
    rank_generator: RankGenerator,
    comm_group: CommGroup,
    comm_group_size: int = None,
    ref_rank: int = 0,
) -> List[int]:
    if comm_group is None:
        return []
    if comm_group == CommGroup.all:
        return list(range(rank_generator.world_size))
    if comm_group not in _COMM_GROUP_TOKEN_MAP:
        raise ValueError(f"Unknown CommGroup value: {comm_group}")
    token, independent_ep = _COMM_GROUP_TOKEN_MAP[comm_group]
    groups = rank_generator.get_ranks(token, independent_ep=independent_ep)
    for group in groups:
        if ref_rank in group:
            return group
    raise ValueError(
        f"Rank {ref_rank} not found in any {comm_group} group. Groups: {groups[:3]}..."
    )


def build_rank_mapping_table(rank_generator: RankGenerator) -> List[Dict]:
    rows = []
    for comm_group, (token, independent_ep) in sorted(
        _COMM_GROUP_TOKEN_MAP.items(), key=lambda x: x[0].value
    ):
        try:
            groups = rank_generator.get_ranks(token, independent_ep=independent_ep)
            group_size = len(groups[0]) if groups else 0
            rank_groups_str = " ".join(
                "[" + ",".join(str(r) for r in g) + "]" for g in groups
            )
            rows.append({
                "group": comm_group.value,
                "size": group_size,
                "rank_groups": rank_groups_str,
            })
        except Exception as e:
            logger.warning(f"Skipping {comm_group.value} in mapping table: {e}")
    rows.append({
        "group": CommGroup.all.value,
        "size": rank_generator.world_size,
        "rank_groups": "[" + ",".join(str(r) for r in range(rank_generator.world_size)) + "]",
    })
    return rows
