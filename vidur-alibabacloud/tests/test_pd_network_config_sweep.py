"""Tests for P/D network design-space generation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidur.config_optimizer.config_explorer.config import (
    JobConfig,
    PDNetworkConfig,
)


def _base_config():
    return {
        "models": [{"name": "toy", "identifier": "toy/model"}],
        "traces": [
            {
                "name": "chat",
                "trace_file": "chat.csv",
                "max_seq_len": 4096,
                "num_requests": 10,
                "start_qps": 1,
            }
        ],
        "clusters": [{"device": "h20", "num_gpus": 8, "gpus_per_node": 8}],
        "schedulers": [{"scheduler": "vllm"}],
        "tp_dimensions": [1],
        "pp_dimensions": [1],
        "batch_sizes": [32],
    }


def test_pd_network_dimension_expands_job_configs():
    config = _base_config()
    config["pd_networks"] = [
        {"name": "mixed"},
        {
            "name": "pd-100g",
            "pd_node_ratio": 0.5,
            "pd_p2p_comm_bandwidth": 100,
            "rdma_bandwidth": 100,
            "nvlink_bandwidth": 900,
        },
        {
            "name": "pd-400g",
            "pd_node_ratio": 0.5,
            "pd_p2p_comm_bandwidth": 400,
            "rdma_bandwidth": 400,
            "nvlink_bandwidth": 900,
        },
    ]

    jobs = JobConfig.generate_job_configs(config)

    assert len(jobs) == 3
    assert len({job.get_key() for job in jobs}) == 3
    assert {
        job.to_config_dict()["replica_config_pd_p2p_comm_bandwidth"]
        for job in jobs
    } == {100, 400, 800}


def test_legacy_config_keeps_single_mixed_mode_job():
    jobs = JobConfig.generate_job_configs(_base_config())

    assert len(jobs) == 1
    generated = jobs[0].to_config_dict()
    assert generated["replica_config_pd_node_ratio"] == 1.0
    assert jobs[0].pd_network_config.name == "mixed"
    assert "_pdr" not in jobs[0].get_key()


def test_pd_network_dtype_is_part_of_job_key():
    float16 = PDNetworkConfig(name="pd-100g", pd_p2p_comm_dtype="float16")
    fp8 = PDNetworkConfig(name="pd-100g", pd_p2p_comm_dtype="fp8")

    assert float16.get_key() != fp8.get_key()
    assert float16.to_config_dict()["replica_config_pd_p2p_comm_dtype"] == "float16"
    assert fp8.to_config_dict()["replica_config_pd_p2p_comm_dtype"] == "fp8"


def test_invalid_pd_split_or_bandwidth_is_rejected():
    assert not PDNetworkConfig(pd_node_ratio=0).is_valid(8)
    assert not PDNetworkConfig(pd_node_ratio=0.01).is_valid(8)
    assert not PDNetworkConfig(
        pd_node_ratio=0.5, pd_p2p_comm_bandwidth=0
    ).is_valid(8)
    assert PDNetworkConfig(pd_node_ratio=0.5).is_valid(8)
