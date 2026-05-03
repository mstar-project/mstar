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
from mminf.model.pi05.kernels.adarms_norm import adarms_norm_fused


class Pi05AdaRMSNorm(nn.Module):
    """RMSNorm with adaRMS conditioning.

    A per-norm ``nn.Linear(cond_dim, hidden_size*3)`` maps the shared
    ``adarms_cond`` vector to ``(scale, shift, gate)``. This mirrors lerobot's
    ``PiGemmaRMSNorm`` (and openpi's ``GemmaRMSNorm``): the conditional path
    has only ``dense.weight``/``dense.bias``; the plain learned RMS ``weight``
    parameter is intentionally NOT created (so checkpoint state dicts that
    only contain ``dense.*`` keys load cleanly).
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.variance_epsilon = eps
        self.dense = nn.Linear(cond_dim, hidden_size * 3, bias=True)
        # Zero-init so the norm starts as the identity (matches lerobot/openpi).
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        BS = cond.shape[0]
        x_flat = x.view(BS * (x.shape[0] // BS), -1).contiguous()

        modulation = self.dense(cond)  # [BS, 3*H]
        H = self.hidden_size
        scale = modulation[:, :H]
        shift = modulation[:, H:2 * H]
        gate_mod = modulation[:, 2 * H:]

        return adarms_norm_fused(x_flat, scale, shift, gate_mod, self.variance_epsilon)


def _gated_residual(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None
) -> torch.Tensor:
    """``x + gate * y`` with a None-gate fallback to plain addition."""
    if gate is None:
        return x + y
    return x + y * gate


class Pi05TimeMLP(nn.Module):
    """Two-layer SiLU MLP that maps the sincos timestep embedding to the
    ``adarms_cond`` vector consumed by every norm in the action expert.

    Both layers operate in the action expert's hidden dimension (which may
    differ from PaliGemma's). Mirrors lerobot's ``time_mlp_in`` /
    ``time_mlp_out`` chain (Linear -> silu -> Linear -> silu).
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
    """One action-expert decoder layer.

    Operates in ``config.action_hidden_size`` (1024 for gemma_300m, 2048 for
    gemma_2b). The attention's K/V dims still match PaliGemma's so the action
    expert can attend to the prefix KV cache PaliGemma wrote.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        h = config.action_hidden_size
        self.self_attn = Pi05PaliGemmaAttention(config, input_hidden_size=h)
        self.mlp = Pi05GemmaMLP(h, config.action_intermediate_size)
        self.input_layernorm = Pi05AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Pi05AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps)

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
    """Stack of action expert layers plus a final adaRMS norm.

    Operates entirely in ``config.action_hidden_size``.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        h = config.action_hidden_size
        self.layers = nn.ModuleList(
            [Pi05ActionExpertLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = Pi05AdaRMSNorm(h, cond_dim=h, eps=config.rms_norm_eps)

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
