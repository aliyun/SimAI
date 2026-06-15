"""
Context Parallelism (CP) utilities for AICB workload generation.

Implements ring-attention communication patterns used by:
  - LLaMA 4 (long-sequence training with CP)
  - DeepSeek-V3 (sequence parallelism + CP combination)
  - Ring Attention (https://arxiv.org/abs/2310.01889)

CP splits the sequence dimension across GPUs. Each GPU computes attention
for its local sequence chunk, then passes K/V tensors to its neighbor in
a ring pattern. This is modeled as asynchronous isend/irecv operations.

Usage:
    from utils.context_parallel import ContextParallelRing

    cp_ring = ContextParallelRing(cp_size=4, cp_rank=rank,
                                   seq_len=seq_len, batch_size=batch_size,
                                   hidden_size=hidden_size)
    workloads = cp_ring.ring_attention_forward(stage_prefix="layer_0")
"""

from utils.utils import CommType, CommGroup
from log_analyzer.log import LogItem, Workload


class ContextParallelRing:
    """Models ring-attention P2P communication for Context Parallelism.

    In ring attention:
      - Sequence is split into cp_size chunks
      - Each step: send local K/V to next rank, receive K/V from previous rank
      - After cp_size steps, each rank has computed attention for all chunks
      - Total communication: (cp_size - 1) rounds of K/V exchange

    K/V size per chunk = seq_len/cp_size * batch_size * num_kv_heads * head_dim
    """

    def __init__(self, cp_size, seq_len, batch_size, hidden_size,
                 num_kv_heads=None, head_dim=None):
        self.cp_size = cp_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if cp_size <= 1:
            self._active = False
        else:
            self._active = True

        # K/V message size per exchange round (one direction)
        if num_kv_heads and head_dim:
            kv_size = num_kv_heads * head_dim
        else:
            kv_size = hidden_size
        self._chunk_kv_size = (seq_len // cp_size) * batch_size * kv_size

    @property
    def active(self):
        return self._active

    def ring_send_recv(self, stage_prefix, direction="forward"):
        """Generate a single round of ring P2P send/recv for K/V tensors.

        Each round: send to (rank+1)%cp_size, recv from (rank-1+cp_size)%cp_size.
        """
        workloads = Workload()
        if not self._active:
            return workloads

        workloads.append(LogItem(
            comm_type=CommType.isend,
            comm_group=CommGroup.cp_group,
            comm_group_size=self.cp_size,
            msg_size=self._chunk_kv_size,
            stage=f"{direction}.CP.ring_send.{stage_prefix}",
        ))
        workloads.append(LogItem(
            comm_type=CommType.irecv,
            comm_group=CommGroup.cp_group,
            comm_group_size=self.cp_size,
            msg_size=self._chunk_kv_size,
            stage=f"{direction}.CP.ring_recv.{stage_prefix}",
        ))
        return workloads

    def ring_attention_forward(self, stage_prefix=""):
        """Full ring attention forward pass: (cp_size - 1) exchange rounds."""
        workloads = Workload()
        if not self._active:
            return workloads

        for step in range(self.cp_size - 1):
            workloads.extend(
                self.ring_send_recv(f"{stage_prefix}.step{step}", "forward")
            )
        return workloads

    def ring_attention_backward(self, stage_prefix=""):
        """Full ring attention backward pass: reverse ring direction."""
        workloads = Workload()
        if not self._active:
            return workloads

        for step in range(self.cp_size - 1):
            workloads.extend(
                self.ring_send_recv(f"{stage_prefix}.step{step}", "backward")
            )
        return workloads
