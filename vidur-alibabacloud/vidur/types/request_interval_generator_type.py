from vidur.types.base_int_enum import BaseIntEnum


class RequestIntervalGeneratorType(BaseIntEnum):
    POISSON = 1
    GAMMA = 2
    STATIC = 3
    TRACE = 4
