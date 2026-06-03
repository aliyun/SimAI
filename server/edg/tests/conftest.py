"""Pytest fixtures for EDG e2e tests."""
import json
import os
import sys
import tempfile
import importlib.util
from collections import defaultdict

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

LLD_PATH = os.path.join(PROJECT_ROOT, "lld.json")
BIN_DIR = os.path.join(PROJECT_ROOT, "server", "bin")
NS3_OXC_BINARY = os.path.join(BIN_DIR, "SimAI_simulator_oxc")


def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_crosses_mod = _import_module("crosses", os.path.join(PROJECT_ROOT, "server", "edg", "crosses.py"))
_merger_mod = _import_module("merger", os.path.join(PROJECT_ROOT, "server", "edg", "merger.py"))

resolve_paths = _merger_mod.resolve_paths


@pytest.fixture(scope="session")
def lld():
    assert os.path.exists(LLD_PATH), f"LLD file not found: {LLD_PATH}"
    with open(LLD_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def crosses(lld):
    topo = lld["topology"]
    oxc_nodes = topo.get("oxc_nodes", [])
    oxc_ips = {n["node_ip"] for n in oxc_nodes}
    leaf_ips = {n["node_ip"] for n in topo.get("leaf_nodes", [])}
    edges = topo.get("edges", [])

    oxc_port_to_leaf = {}
    leaf_to_ports = defaultdict(list)

    for e in edges:
        a_ip, b_ip = e["a_node_ip"], e["b_node_ip"]
        a_port, b_port = str(e["a_node_port_id"]), str(e["b_node_port_id"])
        if a_ip in oxc_ips and b_ip in leaf_ips:
            oxc_port_to_leaf[(a_ip, a_port)] = b_ip
            leaf_to_ports[b_ip].append((a_ip, a_port))
        elif b_ip in oxc_ips and a_ip in leaf_ips:
            oxc_port_to_leaf[(b_ip, b_port)] = a_ip
            leaf_to_ports[a_ip].append((b_ip, b_port))

    base_crosses = set()
    leaf_list = sorted(leaf_to_ports.keys())
    for i in range(len(leaf_list)):
        for j in range(i + 1, len(leaf_list)):
            ports_i = list(leaf_to_ports[leaf_list[i]])
            ports_j = list(leaf_to_ports[leaf_list[j]])
            if ports_i and ports_j:
                oxc_ip_i, port_i = ports_i[0]
                oxc_ip_j, port_j = ports_j[0]
                pa, pb = sorted([port_i, port_j])
                base_crosses.add((oxc_ip_i, pa, pb))

    return base_crosses


@pytest.fixture(scope="session")
def graph(lld, crosses):
    all_server_ips = [n["node_ip"] for n in lld["topology"]["server_nodes"]]
    server_ips = all_server_ips[:2]
    return resolve_paths(lld, crosses, participating_server_ips=server_ips)


@pytest.fixture(scope="session")
def topo_path(graph, lld, tmp_path_factory):
    _emitter_mod = _import_module(
        "ns3_emitter", os.path.join(PROJECT_ROOT, "server", "edg", "ns3_emitter.py")
    )
    write_ns3_topology = _emitter_mod.write_ns3_topology
    out_dir = tmp_path_factory.mktemp("topo")
    out_path = str(out_dir / "edg_topo_test")
    write_ns3_topology(graph, out_path, npu_per_server=8, lld=lld)
    return out_path


@pytest.fixture(scope="session")
def binary_path():
    return NS3_OXC_BINARY
