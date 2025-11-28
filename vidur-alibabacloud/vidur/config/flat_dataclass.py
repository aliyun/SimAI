import json
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
from collections import defaultdict, deque
from dataclasses import MISSING, fields, make_dataclass
from typing import Any, get_args

from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.utils import (
    get_all_subclasses,
    get_inner_type,
    is_bool,
    is_composed_of_primitives,
    is_dict,
    is_list,
    is_optional,
    is_primitive_type,
    is_subclass,
    to_snake_case,
)


def topological_sort(dataclass_dependencies: dict) -> list:
    in_degree = defaultdict(int)
    for cls, dependencies in dataclass_dependencies.items():
        for dep in dependencies:
            in_degree[dep] += 1

    zero_in_degree_classes = deque(
        [cls for cls in dataclass_dependencies if in_degree[cls] == 0]
    )
    sorted_classes = []

    while zero_in_degree_classes:
        cls = zero_in_degree_classes.popleft()
        sorted_classes.append(cls)
        for dep in dataclass_dependencies[cls]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                zero_in_degree_classes.append(dep)

    return sorted_classes


def reconstruct_original_dataclass(self) -> Any:
    """
    This function is dynamically mapped to FlatClass as an instance method.
    """
    sorted_classes = topological_sort(self.dataclass_dependencies)
    instances = {}

    for _cls in reversed(sorted_classes):
        args = {}

        for prefixed_field_name, original_field_name, field_type in self.dataclass_args[
            _cls
        ]:
            if is_subclass(field_type, BasePolyConfig):
                config_type = getattr(self, f"{original_field_name}_type")
                # find all subclasses of field_type and check which one matches the config_type
                config_type_matched = False
                for subclass in get_all_subclasses(field_type):
                    if str(subclass.get_type()) == config_type:
                        config_type_matched = True
                        args[original_field_name] = instances[subclass]
                        break
                assert (
                    config_type_matched
                ), f"Invalid type {config_type} for {prefixed_field_name}_type. Valid types: {[str(subclass.get_type()) for subclass in get_all_subclasses(field_type)]}"
            elif hasattr(field_type, "__dataclass_fields__"):
                args[original_field_name] = instances[field_type]
            else:
                value = getattr(self, prefixed_field_name)
                if callable(value):
                    # to handle default factory values
                    value = value()
                args[original_field_name] = value

        instances[_cls] = _cls(**args)

    return instances[sorted_classes[0]]


@classmethod
def create_from_cli_args(cls) -> Any:
    """
    This function is dynamically mapped to FlatClass as a class method.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    for field in fields(cls):
        nargs = None
        action = None
        field_type = field.type
        help_text = cls.metadata_mapping[field.name].get("help", None)

        if is_list(field.type):
            assert is_composed_of_primitives(field.type)
            field_type = get_args(field.type)[0]
            if is_primitive_type(field_type):
                nargs = "+"
            else:
                field_type = json.loads
        elif is_dict(field.type):
            assert is_composed_of_primitives(field.type)
            field_type = json.loads
        elif is_bool(field.type):
            action = BooleanOptionalAction

        arg_params = {
            "type": field_type,
            "action": action,
            "help": help_text,
        }

        # handle cases with default and default factory args
        if field.default is not MISSING:
            value = field.default
            if callable(value):
                value = value()
            arg_params["default"] = value
        elif field.default_factory is not MISSING:
            arg_params["default"] = field.default_factory()
        else:
            arg_params["required"] = True

        if nargs:
            arg_params["nargs"] = nargs
        parser.add_argument(f"--{field.name}", **arg_params)

    args = parser.parse_args()

    return cls(**vars(args))


def create_flat_dataclass(input_dataclass: Any) -> Any:
    """
    Creates a new FlatClass type by recursively flattening the input dataclass.
    This allows for easy parsing of command line arguments along with storing/loading the configuration to/from a file.
    """
    meta_fields_with_defaults = []
    meta_fields_without_defaults = []
    processed_classes = set()
    dataclass_args = defaultdict(list)
    dataclass_dependencies = defaultdict(set)
    metadata_mapping = {}

    def process_dataclass(_input_dataclass, prefix=""):
        if _input_dataclass in processed_classes:
            return

        processed_classes.add(_input_dataclass)

        for field in fields(_input_dataclass):
            prefixed_name = f"{prefix}{field.name}"

            if is_optional(field.type):
                field_type = get_inner_type(field.type)
            else:
                field_type = field.type

            # # if field is a BasePolyConfig, add a type argument and process it as a dataclass
            if is_subclass(field_type, BasePolyConfig):
                dataclass_args[_input_dataclass].append(
                    (field.name, field.name, field_type)
                )

                type_field_name = f"{field.name}_type"
                default_value = str(field.default_factory().get_type())
                meta_fields_with_defaults.append(
                    (type_field_name, type(default_value), default_value)
                )
                metadata_mapping[type_field_name] = field.metadata

                assert hasattr(field_type, "__dataclass_fields__")
                for subclass in get_all_subclasses(field_type):
                    dataclass_dependencies[_input_dataclass].add(subclass)
                    process_dataclass(subclass, f"{to_snake_case(subclass.__name__)}_")
                continue

            # if field is a dataclass, recursively process it
            if hasattr(field_type, "__dataclass_fields__"):
                dataclass_dependencies[_input_dataclass].add(field_type)
                dataclass_args[_input_dataclass].append(
                    (field.name, field.name, field_type)
                )
                process_dataclass(field_type, f"{to_snake_case(field_type.__name__)}_")
                continue

            field_default = field.default if field.default is not MISSING else MISSING
            field_default_factory = (
                field.default_factory
                if field.default_factory is not MISSING
                else MISSING
            )

            if field_default is not MISSING:
                meta_fields_with_defaults.append(
                    (prefixed_name, field_type, field_default)
                )
            elif field_default_factory is not MISSING:
                meta_fields_with_defaults.append(
                    (prefixed_name, field_type, field_default_factory)
                )
            else:
                meta_fields_without_defaults.append((prefixed_name, field_type))

            dataclass_args[_input_dataclass].append(
                (prefixed_name, field.name, field_type)
            )
            metadata_mapping[prefixed_name] = field.metadata

    process_dataclass(input_dataclass)

    meta_fields = meta_fields_without_defaults + meta_fields_with_defaults
    FlatClass = make_dataclass("FlatClass", meta_fields)

    # Metadata fields
    FlatClass.dataclass_args = dataclass_args
    FlatClass.dataclass_dependencies = dataclass_dependencies
    FlatClass.metadata_mapping = metadata_mapping

    # Helper methods
    FlatClass.reconstruct_original_dataclass = reconstruct_original_dataclass
    FlatClass.create_from_cli_args = create_from_cli_args

    return FlatClass
