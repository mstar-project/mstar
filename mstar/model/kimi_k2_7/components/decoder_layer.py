"""Kimi-K2.7 decoder layer."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.flashinfer_allreduce import all_reduce_add_rmsnorm
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
        self.comm_group = comm_group
        self.self_attn = KimiMLAAttention(config, comm_group=comm_group)
        self.mlp = build_mlp_for_layer(config, layer_idx, comm_group=comm_group)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # hidden_states is the previous layer's unreduced MLP partial:
            # fuse its all-reduce with the residual add and this layer's
            # input norm into one kernel.
            residual, hidden_states = all_reduce_add_rmsnorm(
                self.comm_group, hidden_states, residual, self.input_layernorm,
            )
        attn_partial = self.self_attn(hidden_states)
        residual, hidden_states = all_reduce_add_rmsnorm(
            self.comm_group, attn_partial, residual, self.post_attention_layernorm,
        )
        mlp_partial = self.mlp(hidden_states)
        return mlp_partial, residual
