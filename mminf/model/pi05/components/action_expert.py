"""Action expert transformer with adaRMS timestep conditioning.

The action expert is a Gemma-style transformer that processes action suffix
tokens during the flow-matching loop. It shares KV-cache dimensions with the
PaliGemma expert (same num_kv_heads and head_dim) so it can attend to the
prefix KV cache that PaliGemma wrote during the prefill walk.

adaRMS conditioning (matches openpi's modeling_gemma.GemmaRMSNorm with
``cond_dim`` set): each RMSNorm contains an ``nn.Linear(cond_dim, dim*3)``
that maps the shared ``adarms_cond`` vector to ``(scale, shift, gate)``. The
normalization becomes ``rmsnorm(x) * (1 + scale) + shift`` and the residual
connection becomes ``x + gate * y`` via :func:`_gated_residual`. The same
``adarms_cond`` is fed into all norms within the action expert for a given
Euler step.
"""

from __future__ import annotations

import torch
from torch import nn

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.pi05.components.paligemma import (
    Pi05GemmaMLP,
    Pi05PaliGemmaAttention,
)
from mminf.model.pi05.config import Pi05Config


class Pi05AdaRMSNorm(nn.Module):
    """RMSNorm with adaRMS conditioning.

    When ``cond`` is provided, a per-norm ``nn.Linear(cond_dim, dim*3)`` maps
    it to ``(scale, shift, gate)``. This mirrors the openpi reference norm
    exactly: in the conditional path, the learned ``weight`` parameter is
    intentionally unused — the modulation fully replaces it. The weight is
    kept as a parameter only for checkpoint compatibility with the reference.
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.variance_epsilon = eps
        # Kept for checkpoint compatibility; not used in the cond forward path.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.dense = nn.Linear(cond_dim, hidden_size * 3, bias=True)
        # Zero-init so the norm starts as the identity (matches openpi).
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS normalization in float32 (matches openpi's _norm).
        orig_dtype = x.dtype
        var = torch.mean(x.to(torch.float32).square(), dim=-1, keepdim=True)
        normed = x.to(torch.float32) * torch.rsqrt(var + self.variance_epsilon)
        return normed.to(orig_dtype)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self._rms_normalize(x)
        modulation = self.dense(cond)
        if modulation.dim() == 1:
            # cond was a single vector (e.g., [hidden]); broadcast to the seq.
            scale, shift, gate = modulation.chunk(3, dim=-1)
        else:
            # cond was [B, hidden] -> modulation [B, 3*hidden]; add seq dim.
            modulation = modulation.unsqueeze(-2)  # [B, 1, 3*hidden]
            scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale) + shift
        return normed, gate


def _gated_residual(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None
) -> torch.Tensor:
    """``x + gate * y`` with a None-gate fallback to plain addition."""
    if gate is None:
        return x + y
    return x + y * gate


class Pi05TimeMLP(nn.Module):
    """Two-layer SiLU MLP applied to the sincos timestep embedding.

    The openpi reference uses ``silu(Linear(silu(Linear(sincos(t)))))`` to
    produce the ``adarms_cond`` vector that feeds every norm in the action
    expert.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        h = nn.functional.silu(self.linear_in(time_emb))
        h = self.linear_out(h)
        return nn.functional.silu(h)


class Pi05ActionExpertLayer(nn.Module):
    def __init__(self, config: Pi05Config):
        super().__init__()
        self.self_attn = Pi05PaliGemmaAttention(config)
        self.mlp = Pi05GemmaMLP(config.hidden_size, config.action_intermediate_size)
        self.input_layernorm = Pi05AdaRMSNorm(
            config.hidden_size, cond_dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Pi05AdaRMSNorm(
            config.hidden_size, cond_dim=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = query_sequence
        normed, gate = self.input_layernorm(query_sequence, adarms_cond)
        attn_out = self.self_attn(query_sequence=normed, cache_handle=cache_handle)
        query_sequence = _gated_residual(residual, attn_out, gate)

        residual = query_sequence
        normed, gate = self.post_attention_layernorm(query_sequence, adarms_cond)
        mlp_out = self.mlp(normed)
        return _gated_residual(residual, mlp_out, gate)


class Pi05ActionExpert(nn.Module):
    """Stack of action expert layers plus a final adaRMS norm."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Pi05ActionExpertLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = Pi05AdaRMSNorm(
            config.hidden_size, cond_dim=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = layer(
                query_sequence=query_sequence,
                cache_handle=cache_handle,
                adarms_cond=adarms_cond,
            )
        out, _ = self.norm(query_sequence, adarms_cond)
        return out
