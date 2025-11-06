from abc import ABC
from dataclasses import dataclass
from typing import Any

from vidur.config.utils import get_all_subclasses


@dataclass
class BasePolyConfig(ABC):

    @classmethod
    def create_from_type(cls, type_: Any) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_type() == type_:
                return subclass()
        raise ValueError(f"Invalid type: {type_}")
