"""Pre-norm transformer decoder layers.

``DecoderLayer``: the standard pre-norm block (norm → attn → residual,
norm → mlp → residual). Composes any ``nn.Module`` for the attention,
MLP, and norms — so models pick the variants they want and pass them in.

``GatedDecoderLayer``: variant for adaRMS conditioning (pi05 action
expert). The norms are ``AdaRMSNorm`` and return ``(normed, gate)``; the
residual is ``x + gate * y`` instead of ``x + y``.

``run_layers``: a stack of ``DecoderLayer``s with each residual add deferred
into the next norm (``RMSNorm.forward_residual``).
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


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
        self_attn: attention module taking ``(hidden_states, label=,
            layer_idx=)`` and returning a tensor.
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_mlp_residual(
        self, hidden_states: torch.Tensor, residual: torch.Tensor,
        next_norm: nn.Module, comm_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer after attention, through ``next_norm`` (the next layer's
        input norm, or the stack's final norm): add+norm, MLP, add+norm. Returns
        ``next_norm``'s output and the residual stream. One function, holding no
        layer index, so a compiled stack runs one frame per layer here."""
        h, residual = self.post_attention_layernorm.forward_residual(hidden_states, residual, comm_group)
        return next_norm.forward_residual(self.mlp(h), residual, comm_group)


def run_layers(
    layers: nn.ModuleList, hidden_states: torch.Tensor, norm: nn.Module,
    set_cursor: Callable[[int, nn.Module], None] | None = None,
    comm_group=None,
) -> torch.Tensor:
    """``layers`` then the final ``norm``, each residual add fused into the next
    norm (one add+norm kernel instead of two launches; under TP, with
    ``comm_group``, all-reduce+add+norm). The stream starts at ``hidden_states``,
    a full tensor, and later norms update it in place, so pass a tensor the
    caller may overwrite.

    Layer ``i`` moves its attention's cursor to ``i`` (``self_attn.attend``'s,
    or ``set_cursor(i, self_attn)`` in its place), then attends. The cursor
    calls are eager, so this loop runs eager and no compiled frame specializes
    on ``i``; ``set_cursor`` must be ``torch.compiler.disable``'d whole and do
    no tensor work. Not ``recursive=False``: Dynamo calls back into such a frame
    on every call, and the capture stance (``fail_on_recompile``) raises there.
    """
    first_norm = layers[0].input_layernorm if len(layers) else norm
    h, residual = first_norm.forward_residual(hidden_states, None, comm_group)
    for i, layer in enumerate(layers):
        if set_cursor is None:
            layer.self_attn.attend.set_layer_idx(i)
        else:
            set_cursor(i, layer.self_attn)
        h = layer.self_attn(h)
        next_norm = layers[i + 1].input_layernorm if i + 1 < len(layers) else norm
        h, residual = layer.forward_mlp_residual(h, residual, next_norm, comm_group)
    return h


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
        self, hidden_states: torch.Tensor, adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normed, gate = self.input_layernorm(hidden_states, adarms_cond)
        attn_out = self.self_attn(normed)
        hidden_states = _gated_residual(residual, attn_out, gate)

        residual = hidden_states
        normed, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        mlp_out = self.mlp(normed)
        return _gated_residual(residual, mlp_out, gate)
