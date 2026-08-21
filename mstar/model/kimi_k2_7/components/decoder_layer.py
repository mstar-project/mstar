"""Kimi-K2.7 decoder layer with MLA position ids threaded through attention."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.kimi_k2_7.components.attention import KimiMLAAttention
from mstar.model.kimi_k2_7.components.language_model import (
    build_mlp_for_layer,
    build_rmsnorm,
)
from mstar.model.kimi_k2_7.config import KimiK2Config


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = KimiMLAAttention(config, comm_group=comm_group)
        self.mlp = build_mlp_for_layer(config, layer_idx, comm_group=comm_group)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cache_handle, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
