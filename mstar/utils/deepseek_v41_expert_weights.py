"""Decode the published packed expert weights for the existing FP8 MoE path."""

from __future__ import annotations

import torch

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_FP4_BLOCK = 32
_FP8_BLOCK = 32


def _unpack(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Unpack low nibble first to float32, before applying per-row scales."""
    if packed.ndim != 2 or packed.dtype != torch.int8:
        raise ValueError("expert weight must be a two-dimensional packed int8 tensor")
    if scales.ndim != 2 or scales.dtype != torch.float8_e8m0fnu:
        raise ValueError("expert scales must be two-dimensional E8M0")
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    if not rows or not cols or cols % _FP4_BLOCK or tuple(scales.shape) != (rows, cols // _FP4_BLOCK):
        raise ValueError("expert scale shape must be [rows, unpacked_cols // 32]")
    if scales.device != packed.device or not bool(torch.isfinite(scales.float()).all()):
        raise ValueError("expert scales must be finite and on the weight's device")
    bytes_ = packed.contiguous().view(torch.uint8)
    nibbles = torch.stack((bytes_ & 15, bytes_ >> 4), dim=-1).reshape(rows, cols)
    table = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    return table[nibbles.long()]


def dequantize_expert_fp4(
    packed: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode E2M1 plus per-row/per-32 E8M0 scales for the correctness path."""
    values = _unpack(packed, scales)
    return (values * scales.float().repeat_interleave(_FP4_BLOCK, dim=1)).to(dtype)


def convert_expert_fp4_to_fp8(
    packed: torch.Tensor, scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 weights and float32 32x32 scales for ``fused_experts_fp8``.

    A block's largest E8M0 scale is mapped to FP8 magnitude 64. Refuse a block
    if the other row scales cannot be represented exactly with that one scale.
    The caller can use :func:`dequantize_expert_fp4` for that checkpoint instead.
    """
    values = _unpack(packed, scales)
    rows, cols = values.shape
    if rows % _FP8_BLOCK or cols % _FP8_BLOCK:
        raise ValueError("FP8 expert weights need rows and columns divisible by 32")
    row_blocks, col_blocks = rows // _FP8_BLOCK, cols // _FP8_BLOCK
    scale_rows = scales.float().reshape(row_blocks, _FP8_BLOCK, col_blocks)
    block_scales = scale_rows.amax(dim=1) / 64
    if not bool(torch.isfinite(block_scales).all()) or bool((block_scales == 0).any()):
        raise ValueError("expert scale cannot be represented as a finite FP8 block scale")

    row_offsets = scale_rows / block_scales[:, None, :]
    offsets = row_offsets.repeat_interleave(_FP4_BLOCK, dim=2).reshape(rows, cols)
    fp8_weight = (values * offsets).to(torch.float8_e4m3fn)
    expanded = block_scales.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
    source = values * scales.float().repeat_interleave(_FP4_BLOCK, dim=1)
    restored = fp8_weight.float() * expanded
    if not bool(torch.isfinite(source).all()) or not torch.equal(source, restored):
        raise ValueError("expert FP4 block has row scales not exactly representable in FP8")
    return fp8_weight, block_scales
