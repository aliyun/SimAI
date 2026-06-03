"""
RankTable Generator — server-side version.
Generates ranktable.json files for SimAI simulation.
"""

import json
import os
from typing import Dict, List, Tuple


def generate_ranktable(
    rank_count: int,
    gpus_per_rack: int,
    num_racks: int = None,
    network_type: str = "OXC",
    superpod_prefix: str = "rack",
) -> Tuple[Dict, Dict[str, str]]:
    """Generate RankTable and RankRackMap for SimAI."""
    if num_racks is None:
        num_racks = (rank_count + gpus_per_rack - 1) // gpus_per_rack

    ranktable: Dict = {
        "version": "2.0",
        "status": "completed",
        "rank_count": rank_count,
        "rank_list": [],
    }
    rank_rack_map: Dict[str, str] = {}

    eid_base = 0x000000000000002000100000DF001001
    for rank_id in range(rank_count):
        local_id = rank_id % gpus_per_rack
        rack_name = f"{superpod_prefix}_{rank_id // gpus_per_rack}"
        eid_str = f"{eid_base + rank_id:032x}"

        ranktable["rank_list"].append({
            "rank_id": rank_id,
            "device_id": rank_id,
            "local_id": local_id,
            "level_list": [{
                "net_layer": 0,
                "net_instance_id": rack_name,
                "net_type": "TOPO_FILE_DESC",
                "net_attr": "",
                "rank_addr_list": [{
                    "addr_type": "EID",
                    "addr": eid_str,
                    "ports": ["0/0"],
                    "plane_id": "plane0",
                }],
            }],
        })
        rank_rack_map[str(rank_id)] = rack_name

    return ranktable, rank_rack_map


def generate_ranktable_with_topology(
    rank_count: int,
    topology: List[Dict],
) -> Tuple[Dict, Dict[str, str]]:
    """Generate RankTable with custom topology configuration."""
    ranktable: Dict = {
        "version": "2.0",
        "status": "completed",
        "rank_count": rank_count,
        "rank_list": [],
    }
    rank_rack_map: Dict[str, str] = {}
    rank_to_rack: Dict[int, Dict] = {}

    for rack_config in topology:
        for rank in rack_config.get("ranks", []):
            rank_to_rack[rank] = rack_config

    eid_base = 0x000000000000002000100000DF001001
    for rank_id in range(rank_count):
        rack_config = rank_to_rack.get(rank_id, {"rack_id": 0, "superpod": "rack_0"})
        rack_name = rack_config.get("superpod", f"rack_{rack_config.get('rack_id', 0)}")
        eid_str = f"{eid_base + rank_id:032x}"

        ranktable["rank_list"].append({
            "rank_id": rank_id,
            "device_id": rank_id,
            "local_id": rank_id % 8,
            "level_list": [{
                "net_layer": 0,
                "net_instance_id": rack_name,
                "net_type": "TOPO_FILE_DESC",
                "net_attr": "",
                "rank_addr_list": [{
                    "addr_type": "EID",
                    "addr": eid_str,
                    "ports": ["0/0"],
                    "plane_id": "plane0",
                }],
            }],
        })
        rank_rack_map[str(rank_id)] = rack_name

    return ranktable, rank_rack_map


def validate_ranktable(ranktable: Dict) -> Tuple[bool, List[str]]:
    """Validate ranktable structure."""
    errors: List[str] = []

    if "version" not in ranktable:
        errors.append("Missing 'version' field")
    if "rank_count" not in ranktable:
        errors.append("Missing 'rank_count' field")
    if "rank_list" not in ranktable:
        errors.append("Missing 'rank_list' field")
    else:
        rank_list = ranktable["rank_list"]
        expected = ranktable.get("rank_count", 0)
        if len(rank_list) != expected:
            errors.append(f"rank_list length ({len(rank_list)}) != rank_count ({expected})")
        seen_ids = set()
        for i, entry in enumerate(rank_list):
            if "rank_id" not in entry:
                errors.append(f"Rank entry {i} missing 'rank_id'")
            else:
                rid = entry["rank_id"]
                if rid in seen_ids:
                    errors.append(f"Duplicate rank_id: {rid}")
                seen_ids.add(rid)
            if "level_list" not in entry:
                errors.append(f"Rank entry {i} missing 'level_list'")

    return len(errors) == 0, errors
