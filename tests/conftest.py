"""
Shared pytest fixtures for SimAI-OXC test suite.

Provides common fixtures for project paths, sample data, and test utilities.
"""
import json
import os
import sys
import tempfile

import pytest

# Ensure GUI modules are importable
GUI_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "astra-sim-alibabacloud", "astra-sim", "network_frontend", "oxc", "gui",
)
GUI_DIR = os.path.normpath(GUI_DIR)
if GUI_DIR not in sys.path:
    sys.path.insert(0, GUI_DIR)


@pytest.fixture
def project_root():
    """Return the absolute path to the SimAI project root."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))


@pytest.fixture
def gui_dir():
    """Return the absolute path to the GUI directory."""
    return GUI_DIR


@pytest.fixture
def sample_ranktable():
    """Return a minimal valid RankTable dict for 16 GPUs, 2 racks."""
    rank_list = []
    for i in range(16):
        rack_id = i // 8
        rank_list.append({
            "rank_id": i,
            "device_id": i % 8,
            "local_id": i % 8,
            "level_list": [{
                "net_layer": 0,
                "net_instance_id": f"rack_{rack_id}",
                "net_type": "TOPO_FILE_DESC",
                "net_attr": "",
                "rank_addr_list": [{
                    "addr_type": "EID",
                    "addr": f"10.0.{rack_id}.{i % 8}",
                    "ports": ["0/0"],
                    "plane_id": "plane0",
                }],
            }],
        })
    return {
        "version": "2.0",
        "status": "completed",
        "rank_count": 16,
        "rank_list": rank_list,
    }


@pytest.fixture
def sample_rank_rack_map():
    """Return a minimal rank-to-rack mapping for 16 GPUs, 2 racks."""
    return {str(i): f"rack_{i // 8}" for i in range(16)}


@pytest.fixture
def sample_workload_params():
    """Return default workload generation parameters."""
    return {
        "model_type": "custom",
        "tp_size": 8,
        "dp_size": 2,
        "pp_size": 1,
        "ep_size": 1,
        "vpp": 1,
        "ga": 1,
        "all_gpus": 16,
        "num_layers": 4,
        "hidden_size": 4096,
        "seq_length": 2048,
        "micro_batch_size": 1,
        "vocab_size": 32000,
        "num_attention_heads": 32,
    }


@pytest.fixture
def temp_workspace(tmp_path):
    """Return a temporary workspace directory for test files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def ranktable_json_file(tmp_path, sample_ranktable):
    """Write sample RankTable to a temp JSON file and return the path."""
    path = tmp_path / "ranktable.json"
    path.write_text(json.dumps(sample_ranktable, indent=2))
    return str(path)


@pytest.fixture
def rank_rack_map_json_file(tmp_path, sample_rank_rack_map):
    """Write sample rank-rack map to a temp JSON file and return the path."""
    path = tmp_path / "rank_rack_map.json"
    path.write_text(json.dumps(sample_rank_rack_map, indent=2))
    return str(path)
