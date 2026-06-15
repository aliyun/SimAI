"""
Declarative Tensor Graph for AICB workload generation.

Inspired by STAGE (arXiv:2511.10480), this module provides a data-driven
approach to defining model architectures. Instead of writing imperative
Python LogItem construction for each model, users define the operator
graph in CSV files and compose components declaratively.

Core classes:
  - TensorGraph: loads an operator-level tensor graph from CSV
  - ReplicateGraph: creates N replicas of a subgraph (for layer stacking)
  - ConnectGraph: connects subgraphs by matching named tensor ports

CSV format (columns):
  op_id, op_type, inputs, output, attrs

Where:
  - op_id: unique operation identifier (e.g., "einsum_0")
  - op_type: "einsum" | "placeholder" | "output" | "activation"
  - inputs: comma-separated names of input tensors
  - output: name of the output tensor
  - attrs: semicolon-separated key=value pairs

This is a proof-of-concept implementation. Phase 1 demonstrates that
a SwiGLU FFN can be defined entirely in CSV and produce the same
workload as the hand-coded LlamaMLP.forward().

Usage:
    from workload_generator.tensor_graph.graph import TensorGraph, ReplicateGraph, ConnectGraph

    swiglu = TensorGraph.load("templates/swiglu_ffn.csv")
    layer = ReplicateGraph.apply(swiglu, "layer_0.%s")
    workload = swiglu.to_workload(args)  # generates LogItem list

File: graph.py
License: Apache 2.0
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from utils.utils import CommType, CommGroup, divide
from log_analyzer.log import Workload, LogItem


# ---------------------------------------------------------------------------
# TensorGraph -- core graph representation
# ---------------------------------------------------------------------------

class TensorGraph:
    """A directed acyclic graph of tensor operations loaded from CSV.

    Each node in the graph represents one operation (einsum, placeholder, etc.).
    Edges are data-flow: output tensor of node A is input to node B.
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.nodes: List[Dict[str, Any]] = []       # list of operation dicts
        self.input_ports: List[str] = []            # external input tensor names
        self.output_ports: List[str] = []           # external output tensor names

    def add_node(self, op_id: str, op_type: str, inputs: List[str],
                 output: str, attrs: Optional[Dict[str, Any]] = None):
        self.nodes.append({
            "op_id": op_id,
            "op_type": op_type,
            "inputs": list(inputs),
            "output": output,
            "attrs": attrs or {},
        })

    def set_ports(self, input_ports: List[str], output_ports: List[str]):
        self.input_ports = list(input_ports)
        self.output_ports = list(output_ports)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, csv_path: str, name: Optional[str] = None) -> "TensorGraph":
        """Load a tensor graph from a CSV file.

        CSV columns: op_id, op_type, inputs, output, attrs
        """
        graph = cls(name=name or os.path.basename(csv_path))
        input_ports = []
        output_ports = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_id = row.get("op_id", "").strip()
                op_type = row.get("op_type", "").strip()
                inputs_str = row.get("inputs", "").strip()
                output = row.get("output", "").strip()
                attrs_str = row.get("attrs", "").strip()

                inputs = [t.strip() for t in inputs_str.split(",") if t.strip()]

                # Parse attrs: "key1=value1;key2=value2"
                attrs = {}
                if attrs_str:
                    for kv in attrs_str.split(";"):
                        kv = kv.strip()
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            attrs[k.strip()] = v.strip()

                if op_type == "placeholder":
                    input_ports.append(output)
                elif op_type == "output":
                    output_ports.append(output)
                    # Output node copies its input to the output port
                    if inputs:
                        graph.add_node(op_id, "output", inputs, output, attrs)
                    continue

                graph.add_node(op_id, op_type, inputs, output, attrs)

        graph.set_ports(input_ports, output_ports)
        return graph

    def dump(self, csv_path: str) -> None:
        """Write the tensor graph to a CSV file."""
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["op_id", "op_type", "inputs", "output", "attrs"])
            writer.writeheader()
            for node in self.nodes:
                writer.writerow({
                    "op_id": node["op_id"],
                    "op_type": node["op_type"],
                    "inputs": ",".join(node["inputs"]),
                    "output": node["output"],
                    "attrs": ";".join(f"{k}={v}" for k, v in node["attrs"].items()),
                })

    # ------------------------------------------------------------------
    # Workload generation (Phase 1: computation-only, no TP/EP/CP)
    # ------------------------------------------------------------------

    def to_workload(self, symbol_map: Optional[Dict[str, int]] = None) -> Workload:
        """Convert graph nodes to AICB LogItem Workload.

        symbol_map: maps symbolic dimension names (B, S, H, I) to integer values.
                    If None, uses placeholder sizes.
        """
        workloads = Workload()
        if symbol_map is None:
            symbol_map = {}

        for node in self.nodes:
            op_type = node["op_type"]
            stage = f"{self.name}.{node['op_id']}"

            if op_type in ("placeholder", "output"):
                continue  # no compute/comm for I/O ports

            if op_type == "einsum":
                equation = node["attrs"].get("equation", "bmn,mk->bnk")
                # Estimate tensor shapes from equation and symbol_map
                # Simplified: parse equation for dimensions
                lhs_rhs = equation.split("->")
                lhs_parts = lhs_rhs[0].split(",")
                shape_a = _parse_einsum_shape(lhs_parts[0], symbol_map)
                shape_b = _parse_einsum_shape(lhs_parts[1], symbol_map)
                workloads.append(LogItem(
                    comm_type=CommType.computation,
                    msg_size=(shape_a, shape_b),
                    stage=f"forward.einsum.{stage}",
                ))

        return workloads


# ---------------------------------------------------------------------------
# ReplicateGraph -- stack N copies of a subgraph
# ---------------------------------------------------------------------------

class ReplicateGraph:
    """Creates N named replicas of a TensorGraph.

    Used for layer stacking: each decoder layer is a replica of the same
    subgraph, but with unique tensor names to avoid collisions.
    """

    @staticmethod
    def apply(graph: TensorGraph, name_template: str,
              num_replicas: int = 1) -> List[TensorGraph]:
        """Create num_replicas copies with names from name_template.

        name_template should contain "%s" which is replaced by the replica index.
        Example: name_template="layer_%s" produces "layer_0", "layer_1", ...
        """
        replicas = []
        for i in range(num_replicas):
            replica = TensorGraph(name=name_template % i)
            for node in graph.nodes:
                replica.add_node(
                    op_id=f"{name_template % i}_{node['op_id']}",
                    op_type=node["op_type"],
                    inputs=[f"{name_template % i}_{inp}" if inp not in graph.input_ports else inp
                            for inp in node["inputs"]],
                    output=f"{name_template % i}_{node['output']}",
                    attrs=dict(node["attrs"]),
                )
            replica.set_ports(
                [f"{name_template % i}_{p}" if p not in graph.input_ports else p
                 for p in graph.input_ports],
                [f"{name_template % i}_{p}" for p in graph.output_ports],
            )
            replicas.append(replica)
        return replicas


# ---------------------------------------------------------------------------
# ConnectGraph -- connect subgraphs by matching named ports
# ---------------------------------------------------------------------------

class ConnectGraph:
    """Connects multiple TensorGraph subgraphs into one combined graph.

    Links are name-to-name mappings: {"subgraph_a.output_port": "subgraph_b.input_port"}.
    """

    @staticmethod
    def apply(subgraphs: List[TensorGraph], links: Dict[str, str]) -> TensorGraph:
        """Combine subgraphs into one graph, wiring links between named ports.

        links: maps source_port_name -> target_port_name.
        """
        combined = TensorGraph(name="connected")

        # Collect all nodes from subgraphs
        for sg in subgraphs:
            for node in sg.nodes:
                combined.add_node(
                    op_id=node["op_id"],
                    op_type=node["op_type"],
                    inputs=list(node["inputs"]),
                    output=node["output"],
                    attrs=dict(node["attrs"]),
                )

        # Resolve links: replace target's placeholder input with source's output
        for src_port, tgt_port in links.items():
            for node in combined.nodes:
                if tgt_port in node["inputs"]:
                    idx = node["inputs"].index(tgt_port)
                    node["inputs"][idx] = src_port

        # Collect external ports (those not satisfied by internal links)
        all_inputs = set()
        all_outputs = set()
        for sg in subgraphs:
            all_inputs.update(sg.input_ports)
            all_outputs.update(sg.output_ports)
        satisfied_ports = set(links.keys()) | set(links.values())
        ext_inputs = [p for p in all_inputs if p not in satisfied_ports]
        ext_outputs = [p for p in all_outputs if p not in satisfied_ports]
        combined.set_ports(ext_inputs, ext_outputs)

        return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_einsum_shape(einsum_term: str, symbol_map: Dict[str, int]) -> Tuple[int, ...]:
    """Parse an einsum term like 'bmn' into a shape tuple using symbol_map."""
    shape = []
    for c in einsum_term.strip():
        if c in symbol_map:
            shape.append(symbol_map[c])
        elif c.isalpha() and c.islower():
            shape.append(1)  # unknown symbol defaults to 1
    return tuple(shape) if shape else (1,)
