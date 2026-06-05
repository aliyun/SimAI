"""Unit tests for server.edg.merger — path resolution (v3 spine topology)."""

import json
import os

import pytest

from server.edg.merger import resolve_paths

LLD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "lld.json")


@pytest.fixture
def lld():
    with open(LLD_PATH) as f:
        return json.load(f)


def test_resolve_no_crosses(lld):
    graph = resolve_paths(lld, set(), participating_server_ips=["superpod#0_server#0"])
    assert len(graph["servers"]) == 1
    assert len(graph["leaf_leaf_edges"]) == 0
    assert len(graph["server_leaf_edges"]) >= 1


def test_resolve_dangling_port(lld):
    crosses = {("10.118.241.1(0)", "99", "999")}
    graph = resolve_paths(lld, crosses, participating_server_ips=["superpod#0_server#0"])
    assert len(graph["leaf_leaf_edges"]) == 0


def test_resolve_excludes_non_participating(lld):
    """Non-existent servers appear in output with empty fields (caller-side filtering)."""
    graph = resolve_paths(lld, set(), participating_server_ips=["no_such_server"])
    assert len(graph["servers"]) == 1  # entry created but empty
    assert graph["servers"][0]["leaf_ip"] == ""
    assert len(graph["leaves"]) == 0


def test_resolve_all_servers(lld):
    graph = resolve_paths(lld, set())
    assert len(graph["servers"]) == 1
    assert graph["servers"][0]["server_type"] == "A5"
    assert graph["servers"][0]["ip"] == "superpod#0_server#0"


def test_server_order_matches_input(lld):
    graph = resolve_paths(lld, set(), participating_server_ips=["superpod#0_server#0"])
    assert graph["servers"][0]["ip"] == "superpod#0_server#0"


def test_spine_topology_dual_homed(lld):
    """Server connects to 2 leaves via dual-homing — both leaves participate."""
    graph = resolve_paths(lld, set(), participating_server_ips=["superpod#0_server#0"])
    assert len(graph["servers"]) == 1
    assert len(graph["leaves"]) == 2
    assert len(graph["server_leaf_edges"]) == 2
    assert len(graph["servers"][0]["leaf_ips"]) == 2


def test_chassis_to_npu_type(lld):
    """Verify NPU type derived from chassis_topo."""
    graph = resolve_paths(lld, set())
    assert graph["servers"][0]["server_type"] == "A5"
