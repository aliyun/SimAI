"""
Tests for PD (Prefill-Decode) separation configuration and cluster behavior.

Covers:
  - PD off (pd_node_ratio=1): config init and MIXED mode
  - PD on (pd_node_ratio=0.5): cluster creation with independent P/D clusters
  - PD params None fallback: prefill_*/decode_* fall back to shared values
  - Illegal pd_node_ratio (<=0, >1): must raise ValueError
  - num_prefill_replicas priority over pd_node_ratio
"""

import os
import sys
import pytest

# Ensure vidur package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidur.config import (
    ClusterConfig,
    MetricsConfig,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
)
from vidur.entities.cluster import Cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster(
    num_replicas: int = 4,
    pd_node_ratio: float = 1.0,
    tp: int = 1,
    pp: int = 1,
    prefill_tp=None,
    prefill_pp=None,
    decode_tp=None,
    decode_pp=None,
    num_prefill_replicas=None,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> Cluster:
    """Build a Cluster with the given PD separation params."""
    rc = ReplicaConfig(
        model_name=model_name,
        tensor_parallel_size=tp,
        num_pipeline_stages=pp,
        pd_node_ratio=pd_node_ratio,
        prefill_tensor_parallel_size=prefill_tp,
        prefill_num_pipeline_stages=prefill_pp,
        decode_tensor_parallel_size=decode_tp,
        decode_num_pipeline_stages=decode_pp,
        num_prefill_replicas=num_prefill_replicas,
    )
    cc = ClusterConfig(num_replicas=num_replicas, replica_config=rc)
    mc = MetricsConfig(write_metrics=False, write_json_trace=False)
    gc = SyntheticRequestGeneratorConfig()
    return Cluster(cc, mc, gc)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestPDOff:
    """pd_node_ratio == 1  →  MIXED mode, no PD separation."""

    def test_config_init_defaults(self):
        """ReplicaConfig defaults should have pd_node_ratio=1."""
        rc = ReplicaConfig()
        assert rc.pd_node_ratio == 1
        assert rc.prefill_tensor_parallel_size is None
        assert rc.decode_tensor_parallel_size is None
        assert rc.num_prefill_replicas is None

    def test_mixed_mode_cluster(self):
        """All replicas are created; prefill_ws == decode_ws."""
        cluster = _make_cluster(num_replicas=4, pd_node_ratio=1, tp=1, pp=1)
        rc = cluster._config.replica_config

        assert len(cluster.replicas) == 4
        assert rc.prefill_world_size == rc.decode_world_size
        assert rc._num_prefill_replicas == 4
        assert rc._num_decode_replicas == 4


class TestPDOn:
    """0 < pd_node_ratio < 1  →  PD separation enabled."""

    def test_pd_cluster_creation(self):
        """Replicas are created; P/D counts derived from ratio."""
        cluster = _make_cluster(num_replicas=4, pd_node_ratio=0.5, tp=1, pp=1)
        rc = cluster._config.replica_config

        assert len(cluster.replicas) == 4
        assert rc._num_prefill_replicas == 2
        assert rc._num_decode_replicas == 2

    def test_per_phase_world_size(self):
        """prefill_ws and decode_ws are computed independently."""
        cluster = _make_cluster(num_replicas=4, pd_node_ratio=0.5, tp=1, pp=1)
        rc = cluster._config.replica_config

        # ws = tp * pp * dp
        assert rc.prefill_world_size == 1 * 1 * 2  # tp=1, pp=1, dp=2
        assert rc.decode_world_size == 1 * 1 * 2


class TestPDParamsFallback:
    """PD-specific TP/PP params fall back to shared values when None."""

    def test_none_fallback(self):
        """When prefill_tp/decode_tp are None, use shared values."""
        cluster = _make_cluster(
            num_replicas=4, pd_node_ratio=0.5, tp=2, pp=1,
        )
        rc = cluster._config.replica_config

        assert rc._prefill_tp == 2  # fallback to tp=2
        assert rc._decode_tp == 2   # fallback to tp=2
        assert rc._prefill_pp == 1
        assert rc._decode_pp == 1

    def test_explicit_per_phase_params(self):
        """When prefill_tp/decode_tp are set explicitly, use them."""
        cluster = _make_cluster(
            num_replicas=4, pd_node_ratio=0.5,
            tp=2, pp=1,
            prefill_tp=4, decode_tp=1,
        )
        rc = cluster._config.replica_config

        assert rc._prefill_tp == 4
        assert rc._decode_tp == 1


class TestIllegalPdNodeRatio:
    """pd_node_ratio <= 0 or > 1 must raise ValueError."""

    def test_zero_ratio(self):
        with pytest.raises(ValueError, match="Invalid pd_node_ratio"):
            _make_cluster(num_replicas=4, pd_node_ratio=0)

    def test_negative_ratio(self):
        with pytest.raises(ValueError, match="Invalid pd_node_ratio"):
            _make_cluster(num_replicas=4, pd_node_ratio=-0.5)

    def test_ratio_greater_than_one(self):
        with pytest.raises(ValueError, match="Invalid pd_node_ratio"):
            _make_cluster(num_replicas=4, pd_node_ratio=1.5)


class TestNumPrefillReplicasPriority:
    """num_prefill_replicas takes priority over pd_node_ratio."""

    def test_explicit_prefill_replicas(self):
        """When num_prefill_replicas is set, it overrides pd_node_ratio calc."""
        cluster = _make_cluster(
            num_replicas=8, pd_node_ratio=0.5,
            num_prefill_replicas=3,
        )
        rc = cluster._config.replica_config

        # Should use explicit count, not 0.5 * 8 = 4
        assert rc._num_prefill_replicas == 3
        assert rc._num_decode_replicas == 5
