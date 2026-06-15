from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType, NodeSKUType

logger = init_logger(__name__)


# num_devices_per_node is used in sklearn_execution_time_predictor.py:
#   - L72: devices_per_node = self._replica_config.node_config.num_devices_per_node
#   - L73-75: Validates num_workers vs devices_per_node (legality assertion)
#   - L77: Determines multi-node: self._is_multi_node = num_workers > devices_per_node
# It controls whether TP/EP communication uses NVLink (intra-node) or RDMA (inter-node).
# num_devices_per_node 在 sklearn_execution_time_predictor.py 中被引用:
#   - L72: 获取 devices_per_node
#   - L73-75: 校验 num_workers 与 devices_per_node 的合法性
#   - L77: 判断是否跨节点通信 (NVLink vs RDMA)
@dataclass
class BaseNodeSKUConfig(BaseFixedConfig):
    num_devices_per_node: int


@dataclass
class A40PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A40
    num_devices_per_node: int = 8

    @staticmethod
    def get_type():
        return NodeSKUType.A40_PAIRWISE_NVLINK


@dataclass
class A100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 4

    @staticmethod
    def get_type():
        return NodeSKUType.A100_PAIRWISE_NVLINK


@dataclass
class H100PairwiseNvlinkNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H100
    num_devices_per_node: int = 4

    @staticmethod
    def get_type():
        return NodeSKUType.H100_PAIRWISE_NVLINK


@dataclass
class A100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.A100
    num_devices_per_node: int = 8

    @staticmethod
    def get_type():
        return NodeSKUType.A100_DGX


@dataclass
class H100DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H100
    num_devices_per_node: int = 8

    @staticmethod
    def get_type():
        return NodeSKUType.H100_DGX

@dataclass
class H800DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H800
    num_devices_per_node: int = 8

    @staticmethod
    def get_type():
        return NodeSKUType.H800_DGX
    
@dataclass
class H20DgxNodeSKUConfig(BaseNodeSKUConfig):
    device_sku_type: DeviceSKUType = DeviceSKUType.H20
    num_devices_per_node: int = 8

    @staticmethod
    def get_type():
        return NodeSKUType.H20_DGX