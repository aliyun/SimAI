from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100

@dataclass
class H20DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 148
    fp8_tflops: int = 296
    total_memory_gb: int = 141

    @staticmethod
    def get_type():
        return DeviceSKUType.H20


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H100

@dataclass
class H800DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 989
    fp8_tflops: int = 1979
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H800
