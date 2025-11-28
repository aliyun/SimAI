from abc import ABC
from dataclasses import dataclass
from typing import Any

from vidur.config.utils import get_all_subclasses


@dataclass
class BaseFixedConfig(ABC):

    @classmethod
    def create_from_type(cls, type_: Any) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_type() == type_:
                return subclass()
        raise ValueError(f"[{cls.__name__}] Invalid type: {type_}")

    @classmethod
    def create_from_name(cls, name: str) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass()
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")

    @classmethod
    def create_from_type_string(cls, type_str: str) -> Any:
        for subclass in get_all_subclasses(cls):
            if str(subclass.get_type()) == type_str:
                return subclass()
        raise ValueError(f"[{cls.__name__}] Invalid type string: {type_str}")
