from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType, NodeSKUType

logger = init_logger(__name__)


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