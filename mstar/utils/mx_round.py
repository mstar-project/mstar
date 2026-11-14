"""Microscaling (MX) rounding in torch: values rounded to a block-scaled FP8 or FP4 grid.

DeepSeek-V4.1's reference keeps activations and cache rows in bf16 but rounds them to these
grids first: E4M3 values with a power-of-two scale per block, or E2M1 values with a power-of-two
(E8M0) or an E4M3 scale per block. bf16 holds every rounded value exactly, so a packed FP8/FP4
store or kernel reproduces them bit for bit. This module is that rounding op by op, on any
device; ``pack_e2m1`` / ``unpack_e2m1`` are the checkpoint's packing of E2M1 weights, and
``pack_fp4_rows`` / ``unpack_fp4_rows`` a store of FP4-rounded rows in those bytes.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0
# E2M1 magnitudes by code (sign in bit 3)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# The constants below are made once a device: a tensor built from host values inside a forward is
# a copy to the device that waits for its stream, and one a captured graph cannot record. They are
# made outside inference mode, whatever the caller runs in: an inference tensor kept here would
# fail the first later call that autograd tracks.

@functools.cache
def _e2m1_grid(device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        return torch.tensor(_E2M1, dtype=torch.float32, device=device)


@functools.cache
def _e2m1_bytes(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``[256, 2]``: the values of a packed byte's low and high nibble."""
    with torch.inference_mode(False):
        values = torch.tensor(_E2M1 + tuple(-x for x in _E2M1), dtype=dtype)
        byte = torch.arange(256)
        return torch.stack([values[byte & 0x0F], values[byte >> 4]], dim=-1).to(device)


@functools.cache
def _scalar(value: float, device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        return torch.tensor(value, dtype=torch.float32, device=device)


def pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2(v))`` of positive fp32 ``v``, exact at powers of two."""
    mantissa, exponent = torch.frexp(v)  # v = mantissa * 2 ** exponent, mantissa in [0.5, 1)
    return torch.ldexp(torch.ones_like(v), torch.where(mantissa == 0.5, exponent - 1, exponent))


def round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """fp32 ``v`` in [-6, 6] to the nearest E2M1 value, a tie to the even code."""
    grid = _e2m1_grid(v.device)
    a = v.abs()
    hi = torch.bucketize(a, grid).clamp(max=len(_E2M1) - 1)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    up = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return torch.copysign(torch.where(up, grid[hi], grid[lo]), v)


def pack_e2m1(q: torch.Tensor) -> torch.Tensor:
    """E2M1 values ``[..., K]`` as uint8 ``[..., K // 2]``, element ``2i`` in the low nibble."""
    grid = _e2m1_grid(q.device)
    code = torch.bucketize(q.abs(), grid).to(torch.uint8) | (torch.signbit(q).to(torch.uint8) << 3)
    code = code.unflatten(-1, (-1, 2))
    return (code[..., 0] | (code[..., 1] << 4)).contiguous()


def unpack_e2m1(packed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``pack_e2m1``'s bytes (any one-byte dtype) as ``dtype`` values ``[..., K]`` (every E2M1
    value is exact in bf16 and up)."""
    byte = packed.contiguous().view(torch.uint8)
    return F.embedding(byte.int(), _e2m1_bytes(packed.device, dtype)).flatten(-2)


def quant_fp8(x: torch.Tensor, block: int = 32, pow2_scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [..., N]`` in ``block``-wide groups: E4M3 values ``[..., N]`` and each group's fp32
    scale ``[..., N // block]`` (``amax / 448``, at least ``1e-4 / 448``, rounded up to a power of
    two with ``pow2_scale``)."""
    groups = x.float().unflatten(-1, (-1, block))
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) * _scalar(1 / FP8_MAX, x.device)
    if pow2_scale:
        scale = pow2_ceil(scale)
    q = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(-2), scale.squeeze(-1)


def round_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """``x`` rounded to the FP8 grid of ``quant_fp8`` (power-of-two scales), in ``x``'s dtype."""
    q, scale = quant_fp8(x, block)
    return (q.float().unflatten(-1, (-1, block)) * scale.unsqueeze(-1)).flatten(-2).to(x.dtype)


def quant_fp4(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [..., N]`` in ``block``-wide groups: E2M1 values (fp32) and each group's fp32 scale,
    ``amax / 6`` rounded up to a power of two (E8M0), or with ``e4m3_scale`` rounded to E4M3."""
    groups = x.float().unflatten(-1, (-1, block))
    amax = groups.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = (amax.clamp_min(FP4_MAX * 2.0 ** -9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        scale = pow2_ceil(amax.clamp_min(FP4_MAX * 2.0 ** -126) * _scalar(1 / FP4_MAX, x.device))
    q = round_e2m1((groups / scale).clamp(-FP4_MAX, FP4_MAX))
    return q.flatten(-2), scale.squeeze(-1)


def round_fp4(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False) -> torch.Tensor:
    """``x`` rounded to the FP4 grid of ``quant_fp4``, in ``x``'s dtype."""
    q, scale = quant_fp4(x, block, e4m3_scale)
    return (q.unflatten(-1, (-1, block)) * scale.unsqueeze(-1)).flatten(-2).to(x.dtype)


def pack_fp4_rows(x: torch.Tensor, block: int = 32, e4m3_scale: bool = False) -> torch.Tensor:
    """``x [..., N]`` quantized as ``quant_fp4`` does, as bytes ``[..., N // 2 + N // block]``: the
    E2M1 values two to a byte, then each block's scale (E4M3, or the power of two as E8M0).
    ``unpack_fp4_rows`` of them is ``round_fp4(x)`` bit for bit. Takes ``x`` before rounding: a
    rounded block does not always quantize back to itself (an E4M3 scale below ``2 ** -6`` is
    too coarse to)."""
    q, scale = quant_fp4(x, block, e4m3_scale)
    scale = scale.to(torch.float8_e4m3fn if e4m3_scale else torch.float8_e8m0fnu)
    return torch.cat([pack_e2m1(q), scale.view(torch.uint8)], dim=-1)


def unpack_fp4_rows(rows: torch.Tensor, width: int, block: int = 32, e4m3_scale: bool = False,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``pack_fp4_rows``' bytes (the first ``width // 2 + width // block`` of each row) as
    ``dtype`` values ``[..., width]``: fp32, or bf16, which holds every such value exactly."""
    values = unpack_e2m1(rows[..., :width // 2], dtype)
    scale = rows[..., width // 2:width // 2 + width // block].contiguous()
    scale = scale.view(torch.float8_e4m3fn if e4m3_scale else torch.float8_e8m0fnu).to(dtype)
    return (values.unflatten(-1, (-1, block)) * scale.unsqueeze(-1)).flatten(-2)


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """An E4M3 ``[out, in]`` weight with one scale per ``[out / scale.rows, in / scale.cols]``
    block, as ``dtype`` (exact in bf16 for power-of-two scales)."""
    rows, cols = weight.shape[0] // scale.shape[0], weight.shape[1] // scale.shape[1]
    wide = scale.float().repeat_interleave(rows, dim=0).repeat_interleave(cols, dim=1)
    return (weight.float() * wide).to(dtype)


def dequant_row_fp4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A packed E2M1 ``[out, in // 2]`` weight with one scale per row and ``in / scale.cols``
    inputs, as ``dtype`` ``[out, in]``."""
    values = unpack_e2m1(packed)
    return (values * scale.float().repeat_interleave(values.shape[1] // scale.shape[1], dim=1)).to(dtype)
