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

from typing import List
from workload_generator.mocked_model.MockedModel import MockedModel, Linear


class DeepspeedMLP(MockedModel):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        self.gate_proj = Linear(hidden_size, ffn_hidden_size)
        self.down_proj = Linear(ffn_hidden_size, hidden_size)
        self.up_proj = Linear(hidden_size, ffn_hidden_size)


class DeepspeedAttention(MockedModel):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size)
        # self.rotary_emb = Linear(self.head_dim, self.max_position_embeddings)


class DeepspeedDecoderLayer(MockedModel):
    def __init__(self, config):
        self.input_layernorm = Linear(config.hidden_size, 1)
        self.self_attn = DeepspeedAttention(config=config)
        self.post_attention_layernorm = Linear(config.hidden_size, 1)
        self.mlp = DeepspeedMLP(config.hidden_size, config.ffn_hidden_size)


class DeepspeedModel(MockedModel):
    def __init__(self, config):
        self.embed_tokens = Linear(config.vocab_size, config.hidden_size)
        self.layers = [DeepspeedDecoderLayer(config) for _ in range(config.num_layers)]
        self.norm = Linear(config.hidden_size, 1)


class DeepspeedForCausalLM(MockedModel):
    def __init__(self, config):
        super().__init__()
        self.model = DeepspeedModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)
