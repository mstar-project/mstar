"""Layers shared by the LTX-2.5 DiT and its text connectors.

Ports of ``diffusers.models.transformers.transformer_ltx2`` (``LTX2Attention`` and
its processor, the split rotary embedding) with the op order and dtypes of the
reference kept, so the bf16 results match it. Linear layers are the tensor-parallel
ones from ``mstar.model.components.distributed``; with a trivial comm group they are
plain linears.

Attention itself goes through an ``Attend`` callable the caller supplies (the
scaffold's ``joint_attention`` over a ragged attention resource, or
``sdpa_attention``). Layers never pick a kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

# ``(q, k, v) -> out`` over ``[B, L, H, D]`` tensors (``k``/``v`` may have their own
# ``L``); see ``mstar.model.components.diffusion.attention``.
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class RotaryTable(NamedTuple):
    """``cos`` / ``sin`` of a split rotary embedding, ``[B, T, H, D // 2]`` fp32."""

    cos: torch.Tensor
    sin: torch.Tensor

    def heads(self, start: int, count: int) -> "RotaryTable":
        """This rank's slice of the heads (tensor parallelism shards them)."""
        return RotaryTable(self.cos[:, :, start:start + count], self.sin[:, :, start:start + count])


def rms_norm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``diffusers.models.normalization.RMSNorm(elementwise_affine=False)``: the
    variance in fp32, the product promoted to fp32, then back to the input dtype."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype)


def apply_split_rope(x: torch.Tensor, rope: RotaryTable) -> torch.Tensor:
    """``apply_split_rotary_emb`` for ``[B, T, H, D]``: each head's dims are two halves
    ``(x1, x2)`` rotated to ``(x1 cos - x2 sin, x2 cos + x1 sin)``, in fp32.

    The reference's op sequence (``x * cos``, then ``addcmul_`` per half) is kept on
    purpose: ``addcmul_`` fuses the multiply-add, so ``x1 * cos - sin * x2`` rounds
    differently in a few elements per layer.
    """
    split = x.float().unflatten(-1, (2, -1))                  # [..., 2, D // 2]
    cos, sin = rope.cos.unsqueeze(-2), rope.sin.unsqueeze(-2)
    out = split * cos
    out[..., :1, :].addcmul_(-sin, split[..., 1:, :])
    out[..., 1:, :].addcmul_(sin, split[..., :1, :])
    return out.flatten(-2).to(x.dtype)


class AcrossHeadsRMSNorm(nn.Module):
    """``torch.nn.RMSNorm`` over all heads of a projected q or k (``qk_norm =
    "rms_norm_across_heads"``).

    Under tensor parallelism each rank holds a slice of the heads, so the mean square
    is summed across the group before the rsqrt; the weight is sharded like the
    projection's output.
    """

    def __init__(self, dim: int, eps: float, comm_group: CommGroup):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.comm_group = comm_group
        self.local_dim = dim // comm_group.world_size
        self.weight = nn.Parameter(torch.ones(self.local_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded: torch.Tensor, shard_id=None) -> None:
        start = self.comm_group.rank * self.local_dim
        param.data.copy_(loaded.narrow(0, start, self.local_dim))

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.weight.weight_loader = self.weight_loader
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.comm_group.world_size == 1:
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        square_sum = x.float().pow(2).sum(-1, keepdim=True)
        square_sum = self.comm_group.all_reduce(square_sum)
        normed = x.float() * torch.rsqrt(square_sum / self.dim + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class GatedAttention(nn.Module):
    """``LTX2Attention`` + ``LTX2AudioVideoAttnProcessor``, heads sharded across ``comm_group``.

    q / k are RMS-normalized across heads, then rotated (split RoPE) when a table is
    given; the output is gated per head by ``2 * sigmoid(to_gate_logits(x))``.
    Self-attention fuses q/k/v into one projection; cross-attention fuses k/v.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        head_dim: int,
        comm_group: CommGroup,
        *,
        kv_dim: int | None = None,
        gated: bool = True,
        bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.is_self = kv_dim is None
        self.heads = heads
        self.head_dim = head_dim
        self.local_heads = heads // comm_group.world_size
        self.head_offset = comm_group.rank * self.local_heads
        inner = heads * head_dim
        if self.is_self:
            self.to_qkv = QKVParallelLinear(comm_group, query_dim, head_dim, heads, heads, bias=bias)
        else:
            self.to_q = ColumnParallelLinear(comm_group, query_dim, inner, bias=bias)
            self.to_kv = MergedColumnParallelLinear(comm_group, kv_dim, [inner, inner], bias=bias)
        self.norm_q = AcrossHeadsRMSNorm(inner, eps, comm_group)
        self.norm_k = AcrossHeadsRMSNorm(inner, eps, comm_group)
        self.to_gate_logits = ColumnParallelLinear(comm_group, query_dim, heads, bias=True) if gated else None
        self.to_out = RowParallelLinear(comm_group, inner, query_dim, bias=out_bias)

    def forward(
        self,
        x: torch.Tensor,
        attend: Attend,
        context: torch.Tensor | None = None,
        q_rope: RotaryTable | None = None,
        k_rope: RotaryTable | None = None,
    ) -> torch.Tensor:
        local = self.local_heads * self.head_dim
        if self.is_self:
            q, k, v = self.to_qkv(x).split([local, local, local], dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(context).split([local, local], dim=-1)
        q = self.norm_q(q).unflatten(-1, (self.local_heads, self.head_dim))
        k = self.norm_k(k).unflatten(-1, (self.local_heads, self.head_dim))
        v = v.unflatten(-1, (self.local_heads, self.head_dim))
        if q_rope is not None:
            q = apply_split_rope(q, q_rope.heads(self.head_offset, self.local_heads))
            k_rope = q_rope if k_rope is None else k_rope
            k = apply_split_rope(k, k_rope.heads(self.head_offset, self.local_heads))
        out = attend(q, k, v).to(q.dtype)
        if self.to_gate_logits is not None:
            out = out * (2.0 * torch.sigmoid(self.to_gate_logits(x))).unsqueeze(-1)
        return self.to_out(out.flatten(-2))


class GeluFeedForward(nn.Module):
    """``diffusers.models.attention.FeedForward(activation_fn="gelu-approximate")``."""

    def __init__(self, dim: int, comm_group: CommGroup, *, mult: int = 4, bias: bool = True):
        super().__init__()
        self.up = ColumnParallelLinear(comm_group, dim, dim * mult, bias=bias)
        self.down = RowParallelLinear(comm_group, dim * mult, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x), approximate="tanh"))


def split_rope_table(
    positions: torch.Tensor, dim: int, heads: int, theta: float, max_positions: tuple[float, ...],
) -> RotaryTable:
    """The split-variant ``(cos, sin)`` of ``LTX2AudioVideoRotaryPosEmbed.forward`` /
    ``LTX2RotaryPosEmbed1d``, over fractional positions.

    ``positions``: ``[B, N, num_axes]`` fp32, already the patch midpoints. Each axis is
    scaled to ``[-1, 1]`` by its ``max_positions`` entry and multiplied by a geometric
    frequency grid (built in fp64, as the checkpoint saw); the frequencies of all axes
    interleave, are front-padded with ``cos = 1, sin = 0`` to ``dim // 2``, and split
    across ``heads``.
    """
    batch, tokens, num_axes = positions.shape
    grid = torch.stack([positions[..., i] / max_positions[i] for i in range(num_axes)], dim=-1)
    steps = dim // (2 * num_axes)
    pow_indices = torch.pow(
        theta, torch.linspace(0.0, 1.0, steps, dtype=torch.float64, device=positions.device),
    )
    freqs = (pow_indices * torch.pi / 2.0).to(torch.float32)
    freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs                # [B, N, axes, steps]
    freqs = freqs.transpose(-1, -2).flatten(2)                  # [B, N, steps * axes]
    cos, sin = freqs.cos(), freqs.sin()
    pad = dim // 2 - freqs.shape[-1]
    if pad:
        cos = torch.cat([torch.ones_like(cos[..., :pad]), cos], dim=-1)
        sin = torch.cat([torch.zeros_like(sin[..., :pad]), sin], dim=-1)
    return RotaryTable(cos.reshape(batch, tokens, heads, -1), sin.reshape(batch, tokens, heads, -1))
