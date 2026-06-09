import os
import platform
import socket
import time
from typing import Optional

import ray


def get_ip() -> str:
    # special handling for macos
    if platform.system() == "Darwin":
        return "127.0.0.1"

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_nodes() -> list[str]:
    cluster_resources_keys = list(ray.available_resources().keys())
    ip_addresses = [
        x
        for x in cluster_resources_keys
        if x.startswith("node:") and x != "node:__internal_head__"
    ]

    # special handling for macos, ensure that we only have one node
    if platform.system() == "Darwin":
        assert len(ip_addresses) == 1

    return ip_addresses


def run_on_each_node(func, *args, **kwargs):
    ip_addresses = get_nodes()
    remote_func = ray.remote(func)
    return ray.get(
        [
            remote_func.options(resources={ip_address: 0.1}).remote(*args, **kwargs)
            for ip_address in ip_addresses
        ]
    )


@ray.remote
class CpuAssignmentManager:
    def __init__(self):
        self._nodes = get_nodes()
        # remove "node:" prefix
        self._nodes = [node[5:] for node in self._nodes]
        self._num_cores = os.cpu_count() - 2
        self._core_mapping = {node: [False] * self._num_cores for node in self._nodes}

    def get_cpu_core_id(self) -> Optional[int]:
        for node in self._nodes:
            for i, is_core_assigned in enumerate(self._core_mapping[node]):
                if not is_core_assigned:
                    self._core_mapping[node][i] = True
                    return node, i
        return None, None

    def release_cpu_core_id(self, node: str, cpu_core_id: int) -> None:
        self._core_mapping[node][cpu_core_id] = False


class RayParallelRunner:
    def __init__(self):
        self._cpu_assignment_manager = CpuAssignmentManager.remote()

    def map(self, func, collection):
        # try to assign a core to each task
        promises = []

        remote_func = ray.remote(func)

        for item in collection:
            node = None
            cpu_core_id = None
            while node is None:
                node, cpu_core_id = ray.get(
                    self._cpu_assignment_manager.get_cpu_core_id.remote()
                )
                time.sleep(0.1)
            # launch the task
            promise = remote_func.options(resources={f"node:{node}": 0.001}).remote(
                self._cpu_assignment_manager, cpu_core_id, item
            )
            promises.append(promise)

        return ray.get(promises)
