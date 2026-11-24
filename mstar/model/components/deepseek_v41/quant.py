"""Microscaling (MX) FP8/FP4 rounding, fp32-multiply RMSNorm, and rounded linears for DeepSeek-V4.1."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide

FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

COLUMN = "column"
ROW = "row"


def _e2m1_grid(device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        return torch.tensor(_E2M1, dtype=torch.float32, device=device)


@functools.cache
def _e2m1_bytes(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with torch.inference_mode(False):
        values = torch.tensor(_E2M1 + tuple(-x for x in _E2M1), dtype=dtype)
        byte = torch.arange(256)
        return torch.stack([values[byte & 0x0F], values[byte >> 4]], dim=-1).to(device)


def pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2(v))`` for positive fp32 ``v``, exact at powers of two."""
    mantissa, exponent = torch.frexp(v)
    return torch.ldexp(torch.ones_like(v), torch.where(mantissa == 0.5, exponent - 1, exponent))


def round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 ``v`` in ``[-6, 6]`` to the nearest E2M1 value (ties to even code)."""
    grid = _e2m1_grid(v.device)
    a = v.abs()
    hi = torch.bucketize(a, grid).clamp(max=len(_E2M1) - 1)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    up = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return torch.copysign(torch.where(up, grid[hi], grid[lo]), v)


def pack_e2m1(q: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 values ``[..., K]`` into uint8 ``[..., K // 2]`` (low nibble first)."""
    grid = _e2m1_grid(q.device)
    code = torch.bucketize(q.abs(), grid).to(torch.uint8) | (torch.signbit(q).to(torch.uint8) << 3)
    code = code.unflatten(-1, (-1, 2))
    return (code[..., 0] | (code[..., 1] << 4)).contiguous()


def unpack_e2m1(packed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unpack ``pack_e2m1`` bytes into ``dtype`` values ``[..., K]``."""
    byte = packed.contiguous().view(torch.uint8)
    return F.embedding(byte.int(), _e2m1_bytes(packed.device, dtype)).flatten(-2)


def quant_fp8(x: torch.Tensor, block: int = 32, pow2_scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x [..., N]`` in ``block``-wide groups to E4M3 values + fp32 scales."""
    groups = x.float().unflatten(-1, (-1, block))
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) * (1.0 / FP8_MAX)
    if pow2_scale:
        scale = pow2_ceil(scale)
    q = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(-2), scale.squeeze(-1)


def round_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Round ``x`` to the MX FP8 grid (power-of-two scale per ``block`` elements)."""
    q, scale = quant_fp8(x, block)
    return (q.float().unflatten(-1, (-1, block)) * scale.unsqueeze(-1)).flatten(-2).to(x.dtype)


def quant_fp4(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x [..., N]`` in ``block``-wide groups to E2M1 values + fp32 scales."""
    groups = x.float().unflatten(-1, (-1, block))
    amax = groups.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = (amax.clamp_min(FP4_MAX * 2.0**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        scale = pow2_ceil(amax.clamp_min(FP4_MAX * 2.0**-126) * (1.0 / FP4_MAX))
    q = round_e2m1((groups / scale).clamp(-FP4_MAX, FP4_MAX))
    return q.flatten(-2), scale.squeeze(-1)


def round_fp4(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False) -> torch.Tensor:
    """Round ``x`` to the MX FP4 grid (E8M0 or E4M3 scale per ``block`` elements)."""
    q, scale = quant_fp4(x, block, e4m3_scale)
    return (q.unflatten(-1, (-1, block)) * scale.unsqueeze(-1)).flatten(-2).to(x.dtype)


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize block-scaled E4M3 ``weight [out, in]`` with ``scale`` into ``dtype``."""
    rows, cols = weight.shape[0] // scale.shape[0], weight.shape[1] // scale.shape[1]
    wide = scale.float().repeat_interleave(rows, dim=0).repeat_interleave(cols, dim=1)
    return (weight.float() * wide).to(dtype)


class ReferenceRMSNorm(nn.Module):
    """RMSNorm matching DeepSeek-V4.1's reference: fp32 multiply before casting back."""

    def __init__(self, width: int, eps: float = 1e-20) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * h).to(x.dtype)


def fp32_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` accumulated in fp32 and returned in ``x.dtype``."""
    return F.linear(x.float(), weight.float()).to(x.dtype)


class RoundedLinear(nn.Module):
    """Bias-free TP-aware linear with optional dynamic 32-wide FP8 activation rounding."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        comm_group: CommGroup | None = None,
        shard: str | None = None,
        *,
        rounded: bool = True,
        block: int = 32,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.shard = shard if self.comm_group.world_size > 1 else None
        tp = self.comm_group.world_size if self.shard else 1
        self.in_features = divide(in_features, tp) if self.shard == ROW else in_features
        self.out_features = divide(out_features, tp) if self.shard == COLUMN else out_features
        self.rounded = rounded
        self.block = block
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=dtype))

    def local(self, full: torch.Tensor) -> torch.Tensor:
        """Return this rank's slice of a full ``[out, in]`` checkpoint tensor."""
        rank = self.comm_group.rank
        if self.shard == COLUMN:
            return full.narrow(0, rank * self.out_features, self.out_features)
        if self.shard == ROW:
            return full.narrow(1, rank * self.in_features, self.in_features)
        return full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rounded:
            x = round_fp8(x, self.block)
        out = fp32_linear(x, self.weight)
        if self.shard == ROW:
            out = self.comm_group.all_reduce(out.float()).to(x.dtype)
        return out
