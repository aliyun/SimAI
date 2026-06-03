"""Unit tests for server.edg.merger — path resolution."""

import json
import os

import pytest

from server.edg.merger import resolve_paths

LLD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "lld.json")


@pytest.fixture
def lld():
    with open(LLD_PATH) as f:
        return json.load(f)


def test_resolve_single_cross(lld):
    # OXC port 1 → leaf_25:17, OXC port 16 → leaf_28:17
    crosses = {("10.118.241.50", "1", "16")}
    graph = resolve_paths(lld, crosses, participating_server_ips=["10.118.241.1", "10.118.241.15"])

    assert len(graph["servers"]) == 2
    assert graph["servers"][0]["ip"] == "10.118.241.1"
    assert graph["servers"][1]["ip"] == "10.118.241.15"
    assert len(graph["leaves"]) == 2
    assert len(graph["leaf_leaf_edges"]) == 1

    ll = graph["leaf_leaf_edges"][0]
    assert set([ll[0], ll[1]]) == {"10.118.241.25", "10.118.241.28"}


def test_resolve_no_crosses(lld):
    graph = resolve_paths(lld, set(), participating_server_ips=["10.118.241.1"])
    assert len(graph["servers"]) == 1
    assert len(graph["leaf_leaf_edges"]) == 0
    assert len(graph["server_leaf_edges"]) >= 1


def test_resolve_dangling_port(lld):
    # Port 999 doesn't exist in lld edges
    crosses = {("10.118.241.50", "1", "999")}
    graph = resolve_paths(lld, crosses, participating_server_ips=["10.118.241.1"])
    assert len(graph["leaf_leaf_edges"]) == 0


def test_resolve_excludes_non_participating(lld):
    crosses = {("10.118.241.50", "1", "16")}
    # Only server .1 participates, .15 does not → leaf_28 excluded → no leaf-leaf edge
    graph = resolve_paths(lld, crosses, participating_server_ips=["10.118.241.1"])
    assert len(graph["servers"]) == 1
    assert len(graph["leaf_leaf_edges"]) == 0


def test_resolve_all_servers(lld):
    crosses = {
        ("10.118.241.50", "1", "16"),
        ("10.118.241.50", "5", "9"),
    }
    graph = resolve_paths(lld, crosses)
    assert len(graph["servers"]) == 4
    assert len(graph["leaf_leaf_edges"]) >= 1


def test_server_order_matches_input(lld):
    crosses = {("10.118.241.50", "1", "16")}
    ips = ["10.118.241.15", "10.118.241.1"]
    graph = resolve_paths(lld, crosses, participating_server_ips=ips)
    assert graph["servers"][0]["ip"] == "10.118.241.15"
    assert graph["servers"][1]["ip"] == "10.118.241.1"
