"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from typing import List, Tuple
from enum import Enum

class InferencePhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"

class MockedParam:
    def __init__(self, shape: Tuple, elem_size=2, name=None) -> None:
        self.shape = shape
        self._numel = math.prod(shape)
        self._elem_size = elem_size
        self.name = name if name is not None else "Unknown"

    def numel(self):
        return self._numel

    def elem_size(self):
        return self._elem_size

    def msg_size(self):
        return self._numel * self._elem_size

    def get_shape(self):
        return self.shape

    # def name(self):
    #     return self.param_name


def _unpack_params(value: object) -> List[MockedParam]:
    if isinstance(value, MockedParam):
        return [value]
    elif isinstance(value, MockedModel):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["MockedModel"]:
    if isinstance(value, MockedModel):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    elif isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class MockedParamsBase:
    def __init__(self, model_name: str, frame: str, config_file=None, args=None):
        self.model_name = model_name
        self.frame = frame

        # Load from config file if provided
        if config_file:
            self.load_from_config(config_file)
        
        # Override with command line args if provided
        if args:
            self.load_from_args(args)

        if self.world_size < self.tensor_model_parallel_size or self.world_size < self.expert_model_parallel_size or self.world_size < 1:
            raise ValueError(f"Invalid world size: world_size={self.world_size}, tensor_model_parallel_size={self.tensor_model_parallel_size}, expert_model_parallel_size={self.expert_model_parallel_size}")

    def load_from_config(self, config_file):
        import json
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update attributes with values from config file
            for key, value in config_data.items():
                setattr(self, key, value)
                
        except FileNotFoundError:
            #error, quit
            print(f"[ERRPR] Config file '{config_file}' not found.")
            exit(1)
        except json.JSONDecodeError:
            print(f"[ERRPR] File '{config_file}' is not valid JSON.")
            exit(1)

    def load_from_args(self, args):
        """Load parameters from command line arguments"""
        # List of parameters that can be overridden by command line args
        arg_params = [
            'aiob_enable',
            'seq_length',
            'micro_batch',
            'world_size',
            'tensor_model_parallel_size',
            'expert_model_parallel_size',
            'pipeline_model_parallel',
            'moe_enable',
            'result_dir',
            'phase',
            'aiob_forward_loops'
        ]
        
        # Override parameters with values from args if they exist
        for param in arg_params:
            if hasattr(args, param):
                setattr(self, param, getattr(args, param))


class MockedModel:
    def __init__(self) -> None:
        self._pre_forward_hook = []
        self._post_forward_hook = []
        self._pre_backward_hook = []
        self._post_backward_hook = []

    def parameters(self) -> List[MockedParam]:
        return _unpack_params(self.__dict__)

    def child_modules(self) -> List["MockedModel"]:
        return _child_modules(self.__dict__)

    def register_forward_pre_hook(self, fn):
        self._pre_forward_hook.append(fn)

    def register_backward_pre_hook(self, fn):
        self._pre_backward_hook.append(fn)

    def register_forward_post_hook(self, fn):
        self._post_forward_hook.append(fn)

    def register_backward_post_hook(self, fn):
        self._post_backward_hook.append(fn)


class Linear(MockedModel):  # alias for LlamaRMSNorm, Embedding, LlamaRotaryEmbedding
    def __init__(self, in_feature, out_feature):
        self.weight = MockedParam((in_feature, out_feature))
