"""
Chakra Execution Trace writer for AICB workload output.

Converts AICB Workload (list of LogItem) to the Chakra execution trace schema
(https://github.com/mlcommons/chakra), enabling interop with:
  - ASTRA-sim native Chakra feeder
  - MLCommons Chakra ecosystem tools
  - Chakra visualizer and analysis tools

Chakra schema overview:
  - Each ET is a directed acyclic graph (DAG) of nodes
  - Node types: COMP_ONLY (computation), COMM_COLL (collective comm),
    COMM_SEND (P2P send), COMM_RECV (P2P receive)
  - Nodes have: id, name, type, dependencies (list of parent node ids),
    attributes (tensor sizes, comm size, comm group, etc.)

Output format: Chakra-compatible JSON (no protobuf dependency required).
The JSON format follows the Chakra schema v0.1 as used by ASTRA-sim.

Usage:
    from utils.chakra_writer import ChakraWriter

    workloads = model.forward()
    writer = ChakraWriter(workloads, args)
    writer.write("output_workload.chakra.json")

File: chakra_writer.py
License: Apache 2.0
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from utils.utils import CommType


# ---------------------------------------------------------------------------
# AICB CommType -> Chakra node type mapping
# ---------------------------------------------------------------------------
_COMM_TYPE_TO_CHAKRA = {
    CommType.computation: "COMP_ONLY",
    CommType.all_reduce: "COMM_COLL",
    CommType.all_gather: "COMM_COLL",
    CommType.reduce_scatter: "COMM_COLL",
    CommType.all_to_all: "COMM_COLL",
    CommType.broadcast: "COMM_COLL",
    CommType.reduce: "COMM_COLL",
    CommType.barrier: "COMM_COLL",
    CommType.isend: "COMM_SEND",
    CommType.irecv: "COMM_RECV",
    CommType.epoch_end: "COMP_ONLY",  # treated as barrier/computation boundary
}


def _format_msg_size(msg_size) -> int:
    """Convert LogItem msg_size (int or tuple of tuples) to a byte count.

    For computation: msg_size is ((M,K), (K,N)) tensor shape pairs.
    For communication: msg_size is a byte count (int).

    Returns the total bytes for the operation.
    """
    if isinstance(msg_size, (int, float)):
        return int(msg_size)
    if isinstance(msg_size, (tuple, list)):
        total = 0
        for elem in msg_size:
            if isinstance(elem, (tuple, list)):
                prod = 1
                for d in elem:
                    if isinstance(d, (int, float)):
                        prod *= int(d)
                total += prod
            elif isinstance(elem, (int, float)):
                total += int(elem)
        return total
    return 0


class ChakraWriter:
    """Converts AICB Workload to Chakra execution trace format (JSON)."""

    def __init__(self, workloads: List[Any], args: Any = None):
        self._workloads = workloads
        self._args = args
        self._nodes: List[Dict[str, Any]] = []
        self._next_id = 0

    def _mk_node(self, name: str, node_type: str, dependencies: List[int],
                 attrs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        node = {
            "id": self._next_id,
            "name": name,
            "type": node_type,
            "dependencies": list(dependencies),
            "attrs": attrs or {},
        }
        self._next_id += 1
        return node

    def convert(self) -> List[Dict[str, Any]]:
        """Convert all LogItem entries to Chakra nodes.

        Builds a sequential dependency chain: each node depends on the
        previous one unless they are independent (future enhancement).
        """
        nodes = []
        prev_node_id: Optional[int] = None
        last_comm_send_id: Optional[int] = None

        for item in self._workloads:
            if not hasattr(item, "comm_type"):
                continue

            chakra_type = _COMM_TYPE_TO_CHAKRA.get(item.comm_type, "COMP_ONLY")
            deps = [prev_node_id] if prev_node_id is not None else []

            msg_size = getattr(item, "msg_size", 0)
            stage = getattr(item, "stage", "unknown")
            comm_group = getattr(item, "comm_group", None)
            comm_group_size = getattr(item, "comm_group_size", 1)

            attrs = {
                "stage": stage,
                "msg_size_bytes": _format_msg_size(msg_size),
            }

            # Include raw tensor shapes for computation nodes
            if item.comm_type == CommType.computation and isinstance(msg_size, (tuple, list)):
                attrs["tensor_shapes"] = [
                    list(s) if isinstance(s, (tuple, list)) else s
                    for s in msg_size
                ]

            # Include comm metadata for collective communication
            if chakra_type in ("COMM_COLL", "COMM_SEND", "COMM_RECV"):
                if comm_group is not None:
                    attrs["comm_group"] = str(comm_group)
                attrs["comm_group_size"] = comm_group_size

            # Wire send/recv pairs: a recv depends on the matching send
            if chakra_type == "COMM_RECV" and last_comm_send_id is not None:
                deps.append(last_comm_send_id)

            node = self._mk_node(stage, chakra_type, deps, attrs)
            nodes.append(node)
            prev_node_id = node["id"]

            if chakra_type == "COMM_SEND":
                last_comm_send_id = node["id"]

        self._nodes = nodes
        return nodes

    def write(self, filepath: str) -> None:
        """Convert and write Chakra ET to a JSON file."""
        nodes = self.convert()

        # Chakra schema metadata
        output: Dict[str, Any] = {
            "schema": "https://github.com/mlcommons/chakra",
            "schema_version": "0.1",
            "num_nodes": len(nodes),
            "nodes": nodes,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Chakra ET written to {filepath} ({len(nodes)} nodes)")

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        if not self._nodes:
            self.convert()
        return self._nodes
