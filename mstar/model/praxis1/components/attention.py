"""The trunk's grouped-query attention, with its per-head sigmoid gate.

The order is the part worth stating, because every step of it is a place a
port silently diverges:

    project -> rotate q and k -> RMS-norm q and k per head -> attend ->
    multiply by the gate -> project out

Two of those are backwards from the common convention. The rotation happens
**before** the q/k norm (``rope_after_norm: false`` in the checkpoint), and
the norms are per *head dimension* rather than over the model width, scaled
by ``1 + weight`` in float32. Swapping the rotate and norm leaves every
shape intact and changes every number.

The gate is a plain ``sigmoid`` of a bias-free projection to one scalar per
head, applied to the attention output before the out projection. It is not
a GLU and not part of the MLP.

Q, K and V come from three separate bias-free projections rather than one
fused matrix, which matters for the weight loader: the checkpoint stores
``q_weight``, ``k_weight`` and ``v_weight`` apart, and fusing them here
would need a split on load that the reference never does.

Attention itself is left to the caller. The served path wants the engine's
kernel under the chunk-causal mask, which is the resource M* does not have
yet (see :mod:`mstar.model.praxis1.components.chunk_mask`); this module
takes a callable so the parity tests can drive it with plain SDPA and the
engine can drive it with something better.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from mstar.model.praxis1.components.norm_time import Float32RMSNorm
from mstar.model.praxis1.components.rope import apply_rope


class QKVProjection(nn.Module):
    """Three bias-free projections, kept separate as the checkpoint has them."""

    def __init__(self, dim: int, head_dim: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        self.q_weight = nn.Parameter(torch.empty(num_heads * head_dim, dim))
        self.k_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))
        self.v_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))

    def forward(self, x: torch.Tensor):
        return (
            torch.nn.functional.linear(x, self.q_weight),
            torch.nn.functional.linear(x, self.k_weight),
            torch.nn.functional.linear(x, self.v_weight),
        )


def sdpa_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reference-shaped attention over a dense boolean mask.

    This is the parity driver, not the served path: it materializes the
    whole ``(S, S)`` mask. ``enable_gqa`` lets the 8 K/V heads serve the 24
    query heads without an explicit repeat.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=k.shape[1] != q.shape[1]
    )


class Attention(nn.Module):
    """Grouped-query self-attention with a per-head sigmoid gate."""

    def __init__(
        self,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.num_kv_heads = num_kv_heads
        self.in_proj = QKVProjection(dim, head_dim, self.num_heads, num_kv_heads)
        self.gate_proj = nn.Linear(dim, self.num_heads, bias=False)
        # Per head dimension, not per model width.
        self.q_norm = Float32RMSNorm(head_dim)
        self.k_norm = Float32RMSNorm(head_dim)
        self.out_proj = nn.Linear(self.num_heads * head_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        attend: Callable = sdpa_attend,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        d = self.head_dim
        q, k, v = self.in_proj(x)
        g = self.gate_proj(x).sigmoid()
        g = g.permute(0, 2, 1).reshape(b, self.num_heads, s, 1)
        q = q.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        k = k.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        v = v.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        # Rotate first, then norm. This order is the checkpoint's.
        if rope is not None:
            q, k = apply_rope(q, k, rope)
        q, k = self.q_norm(q), self.k_norm(k)
        out = attend(q, k, v, mask)
        out = out * g
        out = out.permute(0, 2, 1, 3).reshape(b, s, self.num_heads * d)
        return self.out_proj(out)
