from abc import ABC, abstractmethod
from typing import Any

from vidur.types import BaseIntEnum


class BaseRegistry(ABC):
    _key_class = BaseIntEnum

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, key: BaseIntEnum, implementation_class: Any) -> None:
        if key in cls._registry:
            return

        cls._registry[key] = implementation_class

    @classmethod
    def unregister(cls, key: BaseIntEnum) -> None:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        del cls._registry[key]

    @classmethod
    def get(cls, key: BaseIntEnum, *args, **kwargs) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        return cls._registry[key](*args, **kwargs)

    @classmethod
    def get_class(cls, key: BaseIntEnum) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        return cls._registry[key]

    @classmethod
    @abstractmethod
    def get_key_from_str(cls, key_str: str) -> BaseIntEnum:
        pass
    
    # @classmethod is a decorator used to mark a method as a class method
    # The first parameter of a class method is cls, representing the class itself (not an instance)
    # It can be called directly through the class name without creating an instance
    
    # 2. Parameter Description
    # cls: The class itself (in this example it is GlobalSchedulerRegistry)
    # key_str: str: A string parameter used to identify the type of registered item to retrieve, such as random, round_robin, split_wise, lor
    # *args: Variable positional arguments, allowing passing of any number of positional arguments
    # **kwargs: Variable keyword arguments, allowing passing of any number of keyword arguments
    # -> Any: Function return type hint, indicating that any type can be returned
    
    @classmethod
    def get_from_str(cls, key_str: str, *args, **kwargs) -> Any:
        # import pdb; pdb.set_trace() # >
        return cls.get(cls.get_key_from_str(key_str), *args, **kwargs)
