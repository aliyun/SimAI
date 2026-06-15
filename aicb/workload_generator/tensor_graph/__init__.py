"""
Declarative Tensor Graph package for AICB workload generation.

Provides a data-driven alternative to imperative MockedModel construction.
Tensor graphs are defined in CSV files and composed declaratively, following
the STAGE (arXiv:2511.10480) approach.

Main classes:
  - TensorGraph: load/save operator graphs from CSV
  - ReplicateGraph: create N replicas for layer stacking
  - ConnectGraph: wire subgraphs by named tensor ports
"""

from workload_generator.tensor_graph.graph import (
    TensorGraph,
    ReplicateGraph,
    ConnectGraph,
)

__all__ = ["TensorGraph", "ReplicateGraph", "ConnectGraph"]
