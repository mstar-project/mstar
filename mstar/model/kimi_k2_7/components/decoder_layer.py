"""Kimi-K2.7 / DeepSeek-V3 decoder layer (M4 assembly).

One pre-norm transformer block: MLA self-attention then a feed-forward that is
either the dense SwiGLU MLP (the ``first_k_dense_replace`` early layers) or the
fine-grained sigmoid-routed MoE block. Both feed-forwards expose the same
``(x) -> x`` interface, so the residual wiring here is agnostic to which it holds
(``build_mlp_for_layer`` picks per ``layer_idx``).

This is a Kimi-specific decoder layer rather than the shared
``mstar.model.components.DecoderLayer`` because MLA attention needs
``position_ids`` threaded through its forward (the shared layer's
``self_attn(x, cache_handle=...)`` signature has no position channel — YARN RoPE
is applied inside the attention over the decoupled ``qk_rope`` slice).

Residual structure mirrors vLLM ``DeepseekV2DecoderLayer.forward``:
    residual = h
    h = input_layernorm(h); h = self_attn(h, cache, pos); h = residual + h
    residual = h
    h = post_attention_layernorm(h); h = mlp(h); h = residual + h
"""
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
    """Pre-norm MLA + (dense-or-MoE) feed-forward block.

    Args:
        config: model config.
        layer_idx: index into the stack; selects the dense MLP (``layer_idx <
            first_k_dense_replace``) or the MoE block (``build_mlp_for_layer``).
        comm_group: TP comm group (trivial single-rank if ``None``).
    """

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
