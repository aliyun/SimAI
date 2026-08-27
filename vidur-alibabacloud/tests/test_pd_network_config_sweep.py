"""Tests for P/D network design-space generation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vidur.config_optimizer.config_explorer.config import JobConfig


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
        "clusters": [{"device": "h100", "num_gpus": 8, "gpus_per_node": 8}],
        "schedulers": [{"scheduler": "vllm"}],
        "tp_dimensions": [1],
        "pp_dimensions": [1],
        "batch_sizes": [32],
    }


def test_pd_network_dimension_expands_jobs_and_replica_config():
    config = _base_config()
    config["pd_networks"] = [
        {
            "name": "pd",
            "pd_node_ratio": 0.5,
            "pd_p2p_comm_bandwidth": 100,
            "rdma_bandwidth": 200,
            "nvlink_bandwidth": 900,
            "pd_p2p_comm_dtype": "float16",
        },
        {
            "name": "pd",
            "pd_node_ratio": 0.5,
            "pd_p2p_comm_bandwidth": 100,
            "rdma_bandwidth": 200,
            "nvlink_bandwidth": 900,
            "pd_p2p_comm_dtype": "fp8",
        },
    ]

    jobs = JobConfig.generate_job_configs(config)

    assert len(jobs) == 2
    assert len({job.get_key() for job in jobs}) == 2
    assert {
        job.to_config_dict()["replica_config_pd_p2p_comm_dtype"] for job in jobs
    } == {"float16", "fp8"}


def test_legacy_config_preserves_job_key():
    job = JobConfig.generate_job_configs(_base_config())[0]

    assert job.get_key() == "toy_chat_tk4096_rq10_h100_vllm_tp1_pp1_bsz32"
    assert job.to_config_dict()["replica_config_pd_node_ratio"] == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"pd_node_ratio": 0},
        {"pd_node_ratio": 1.1},
        {"pd_p2p_comm_bandwidth": 0},
        {"rdma_bandwidth": -1},
        {"nvlink_bandwidth": 0},
        {"pd_p2p_comm_dtype": "int8"},
    ],
)
def test_invalid_pd_network_point_is_rejected(overrides):
    config = _base_config()
    network = {
        "name": "invalid",
        "pd_node_ratio": 0.5,
        "pd_p2p_comm_bandwidth": 100,
        "rdma_bandwidth": 200,
        "nvlink_bandwidth": 900,
        "pd_p2p_comm_dtype": "float16",
    }
    network.update(overrides)
    config["pd_networks"] = [network]

    with pytest.raises(ValueError):
        JobConfig.generate_job_configs(config)


def test_pd_split_with_zero_phase_replicas_is_skipped():
    config = _base_config()
    config["pd_networks"] = [{"name": "invalid-split", "pd_node_ratio": 0.01}]

    assert JobConfig.generate_job_configs(config) == []


def test_empty_or_duplicate_pd_network_dimension_is_rejected():
    config = _base_config()
    config["pd_networks"] = []
    with pytest.raises(ValueError):
        JobConfig.generate_job_configs(config)

    point = {"name": "duplicate", "pd_node_ratio": 1}
    config["pd_networks"] = [point, point]
    with pytest.raises(ValueError):
        JobConfig.generate_job_configs(config)

    config["pd_networks"] = [{"pd_node_ratio": 0.5}]
    with pytest.raises(ValueError):
        JobConfig.generate_job_configs(config)
