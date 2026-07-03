"""Pre-norm transformer decoder layers.

``DecoderLayer``: the standard pre-norm block (norm → attn → residual,
norm → mlp → residual). Composes any ``nn.Module`` for the attention,
MLP, and norms — so models pick the variants they want and pass them in.

``GatedDecoderLayer``: variant for adaRMS conditioning (pi05 action
expert). The norms are ``AdaRMSNorm`` and return ``(normed, gate)``; the
residual is ``x + gate * y`` instead of ``x + y``.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.engine.cache_manager import BatchedCacheManager


def _gated_residual(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None,
) -> torch.Tensor:
    """``x + gate * y`` with a None-gate fallback to plain addition."""
    if gate is None:
        return x + y
    return x + y * gate


class DecoderLayer(nn.Module):
    """Standard pre-norm transformer decoder layer.

    Computes:
        residual = x
        x = input_layernorm(x); x = self_attn(x); x = residual + x
        residual = x
        x = post_attention_layernorm(x); x = mlp(x); x = residual + x

    Args:
        self_attn: attention module taking ``(hidden_states, cache_handle)``
            and returning a tensor.
        mlp: feedforward module taking and returning a tensor.
        input_layernorm: pre-attention norm.
        post_attention_layernorm: pre-FFN norm.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        mlp: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cache_handle=cache_handle)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class GatedDecoderLayer(nn.Module):
    """Pre-norm decoder layer with adaRMS gated residuals.

    The norms must be ``AdaRMSNorm``-shaped: ``forward(x, cond)`` returns
    ``(normed, gate)``. The residual becomes ``x + gate * y``.

    Used by pi05's action expert; ``adarms_cond`` is the shared condition
    vector consumed by both norms.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        mlp: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normed, gate = self.input_layernorm(hidden_states, adarms_cond)
        attn_out = self.self_attn(normed, cache_handle=cache_handle)
        hidden_states = _gated_residual(residual, attn_out, gate)

        residual = hidden_states
        normed, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        mlp_out = self.mlp(normed)
        return _gated_residual(residual, mlp_out, gate)
