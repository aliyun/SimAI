from vidur.types.base_int_enum import BaseIntEnum


class ActivationType(BaseIntEnum):
    GELU = 0
    SILU = 1

    # > DPSK v3
    DYNAMIC = 3