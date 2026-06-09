from vidur.request_generator.fixed_request_length_generator import (
    FixedRequestLengthGenerator,
)
from vidur.request_generator.trace_request_length_generator import (
    TraceRequestLengthGenerator,
)
from vidur.request_generator.uniform_request_length_generator import (
    UniformRequestLengthGenerator,
)
from vidur.request_generator.zipf_request_length_generator import (
    ZipfRequestLengthGenerator,
)
from vidur.types import RequestLengthGeneratorType
from vidur.utils.base_registry import BaseRegistry


class RequestLengthGeneratorRegistry(BaseRegistry):
    pass


RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.ZIPF, ZipfRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.UNIFORM, UniformRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.TRACE, TraceRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.FIXED, FixedRequestLengthGenerator
)
