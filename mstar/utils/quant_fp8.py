"""Activation quantization for block-scale GEMMs.

Per-token-group FP8: the e4m3 activation quant that feeds block-scale W8A8
GEMMs (the fused MoE experts today).  Adapted from sglang's
``fused_moe_triton_kernels.py`` (Apache-2.0).

Block-scaled, for cuBLASLt's MXFP8 and NVFP4 GEMMs (``quant_mxfp8``,
``quant_nvfp4``): the same idea with the scales in the GEMM's format and
swizzled layout. ``model.components.block_scaled`` holds the formats.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.utils.fp8_block import (
    BLOCKS_PER_TILE,
    E8M0_DTYPE,
    FP4_DTYPE,
    FP8_DTYPE,
    BlockQuant,
    swizzled_scale_elements,
)

# Groups quantized per program.  One program per group (sglang's shape) is
# 393k programs for an 8k x 6144 prefill; 16 matches vLLM's CUDA quantizer.
GROUPS_PER_PROGRAM = 16


@triton.jit
def per_token_group_quant_fp8_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    num_groups,
    eps,
    fp8_min,
    fp8_max,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Quantize ``GROUPS`` contiguous ``group_size`` slices to e4m3, one fp32 scale each.

    Groups tile the rows of a contiguous 2-D tensor: group ``g`` is
    ``y.view(-1)[g*group_size:(g+1)*group_size]`` and writes scale slot ``g``
    of the row-major ``(M, K // group_size)`` scales.  Program ``p`` takes
    groups ``[p*GROUPS, (p+1)*GROUPS)``; the tail past ``num_groups`` is masked.
    """
    p = tl.program_id(0).to(tl.int64)
    groups = p * GROUPS + tl.arange(0, GROUPS)
    cols = tl.arange(0, BLOCK)
    g_mask = groups < num_groups
    mask = g_mask[:, None] & (cols < group_size)[None, :]
    offs = groups[:, None] * group_size + cols[None, :]

    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # amax / e4m3-max with an eps floor so all-zero groups get a finite scale.
    y_s = tl.maximum(tl.max(tl.abs(y), axis=1), eps) / fp8_max
    y_q = tl.minimum(tl.maximum(y / y_s[:, None], fp8_min), fp8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + offs, y_q, mask=mask)
    tl.store(y_s_ptr + groups, y_s, mask=g_mask)


@torch.compiler.disable
def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to e4m3, one fp32 scale per ``group_size`` K-slice per row.

    Returns ``(x_q, x_s)`` with ``x_q`` e4m3 of ``x.shape`` and ``x_s`` fp32 of
    shape ``(M, K // group_size)``; dequant is ``x_q * x_s`` (the same
    multiply-back convention as the checkpoint's ``weight_scale_inv``).

    ``torch.compiler.disable``: Inductor's (re)compile of this kernel fails in
    Triton's make_llir ("PassManager::run failed") while the kernel's own JIT
    path is fine.  The graph break keeps Inductor out of the launch while
    stream capture still records it.  Remove when the toolchain bug is fixed.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.shape[-1] % group_size == 0, (
        f"last dim {x.shape[-1]} must be a multiple of group_size {group_size}"
    )

    M, K = x.shape
    finfo = torch.finfo(FP8_DTYPE)
    x_q = torch.empty_like(x, dtype=FP8_DTYPE)
    x_s = torch.empty((M, K // group_size), dtype=torch.float32, device=x.device)

    num_groups = M * (K // group_size)
    grid = (triton.cdiv(num_groups, GROUPS_PER_PROGRAM),)
    per_token_group_quant_fp8_kernel[grid](
        x,
        x_q,
        x_s,
        group_size,
        num_groups,
        eps,
        finfo.min,
        finfo.max,
        BLOCK=triton.next_power_of_2(group_size),
        GROUPS=GROUPS_PER_PROGRAM,
    )
    return x_q, x_s


# Rows per program of the block quantizers: a scale tile's rows come in fours of these.
_BLOCK_QUANT_ROWS = 32


@triton.jit
def _scale_tile_offsets(g, col_tile, col_tiles, ROWS: tl.constexpr, BLOCKS: tl.constexpr):
    """Flat offsets of a ``[ROWS, BLOCKS]`` piece of the swizzled scales, row group ``g``, column tile ``col_tile``."""
    return (
        512 * ((g // 4) * col_tiles + col_tile) + tl.arange(0, ROWS)[:, None] * 16 + (g % 4) * BLOCKS
        + tl.arange(0, BLOCKS)[None, :]
    )


@triton.jit
def _mxfp8_quant_kernel(
    y_ptr, q_ptr, s_ptr, rows, k, col_tiles, ROWS: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
):
    """One scale tile's ``[ROWS, BLOCKS * BLOCK]`` values: e4m3 data, biased e8m0 scales. One read per value."""
    g = tl.program_id(0).to(tl.int64)
    col_tile = tl.program_id(1).to(tl.int64)
    row = g * ROWS + tl.arange(0, ROWS)
    live = (row < rows)[:, None]
    offs = row[:, None] * k + col_tile * (BLOCKS * BLOCK) + tl.arange(0, BLOCKS * BLOCK)[None, :]
    y = tl.reshape(tl.load(y_ptr + offs, mask=live, other=0.0).to(tl.float32), [ROWS, BLOCKS, BLOCK])
    # ceil(log2(amax / 448)) from the float's own bits, biased like e8m0; the multiply-back is built the same way
    bits = (tl.max(tl.abs(y), axis=2) / 448.0).to(tl.int32, bitcast=True)
    biased = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    biased = tl.minimum(tl.maximum(biased, 1), 253)
    inverse = ((254 - biased) << 23).to(tl.float32, bitcast=True)
    tl.store(q_ptr + offs, tl.reshape(y * inverse[:, :, None], [ROWS, BLOCKS * BLOCK]).to(tl.float8e4nv), mask=live)
    tl.store(s_ptr + _scale_tile_offsets(g, col_tile, col_tiles, ROWS, BLOCKS), biased.to(tl.uint8), mask=live)


@triton.jit
def _nvfp4_quant_kernel(
    y_ptr, q_ptr, s_ptr, ts_ptr, rows, k, col_tiles, ROWS: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
):
    """One scale tile's ``[ROWS, BLOCKS * BLOCK]`` values: packed e2m1 data, e4m3 scales in units of the tensor scale
    at ``ts_ptr``, rounded up. One read per value."""
    g = tl.program_id(0).to(tl.int64)
    col_tile = tl.program_id(1).to(tl.int64)
    row = g * ROWS + tl.arange(0, ROWS)
    live = (row < rows)[:, None]
    offs = row[:, None] * k + col_tile * (BLOCKS * BLOCK) + tl.arange(0, BLOCKS * BLOCK)[None, :]
    y = tl.reshape(tl.load(y_ptr + offs, mask=live, other=0.0).to(tl.float32), [ROWS, BLOCKS, BLOCK])
    tensor_scale = tl.load(ts_ptr)
    want = tl.max(tl.abs(y), axis=2) / 6.0 / tensor_scale
    code = want.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.int32)
    code = code + (code.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32) < want).to(tl.int32)
    code = tl.minimum(tl.maximum(code, 1), 0x7E)
    scale = code.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32) * tensor_scale
    u = y / scale[:, :, None]
    a = tl.abs(u)
    nibble = (
        (a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32) + (u < 0).to(tl.int32) * 8
    )
    # e2m1x2 stores the first value of a pair in the low nibble
    nibble = tl.reshape(nibble, [ROWS, BLOCKS * BLOCK // 2, 2]) << (tl.arange(0, 2) * 4)[None, None, :]
    q_offs = row[:, None] * (k // 2) + col_tile * (BLOCKS * BLOCK // 2) + tl.arange(0, BLOCKS * BLOCK // 2)[None, :]
    tl.store(q_ptr + q_offs, tl.sum(nibble, axis=2).to(tl.uint8), mask=live)
    tl.store(s_ptr + _scale_tile_offsets(g, col_tile, col_tiles, ROWS, BLOCKS), code.to(tl.uint8), mask=live)


def _block_grid(x: torch.Tensor, block: int) -> tuple[tuple[int, int], int, torch.Tensor]:
    """The quantizers' grid, column tiles and swizzled scale buffer for ``x``. Triton has no e8m0 or e2m1x2
    pointers; both are written as bytes and viewed. Rows need not fill a scale tile: the GEMM doesn't read scales
    past its M."""
    rows, k = x.shape
    k_blocks = k // block
    col_tiles = k_blocks // BLOCKS_PER_TILE
    scale = torch.empty(swizzled_scale_elements(rows, k_blocks), dtype=torch.uint8, device=x.device)
    return (triton.cdiv(rows, _BLOCK_QUANT_ROWS), col_tiles), col_tiles, scale


def quant_mxfp8(x: torch.Tensor) -> BlockQuant:
    """A contiguous bf16 ``[rows, K]`` activation as MXFP8: e4m3 data, one power-of-two e8m0 scale per 32 values.
    ``block_scaled.Mxfp8Format.quantize`` is the host-side reference."""
    rows, k = x.shape
    grid, col_tiles, scale = _block_grid(x, 32)
    data = torch.empty_like(x, dtype=FP8_DTYPE)
    _mxfp8_quant_kernel[grid](
        x, data, scale, rows, k, col_tiles, ROWS=_BLOCK_QUANT_ROWS, BLOCKS=BLOCKS_PER_TILE, BLOCK=32,
    )
    return BlockQuant(data, scale.view(E8M0_DTYPE))


def quant_nvfp4(x: torch.Tensor, tensor_scale: torch.Tensor) -> BlockQuant:
    """A contiguous bf16 ``[rows, K]`` activation as NVFP4: packed e2m1 data, one e4m3 scale per 16 values in units
    of ``tensor_scale`` (a ``[1]`` fp32 on the device, the linear's: reading the activation twice for its own
    maximum costs as much as quantizing it). ``block_scaled.Nvfp4Format.quantize`` is the host-side reference."""
    rows, k = x.shape
    grid, col_tiles, scale = _block_grid(x, 16)
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    _nvfp4_quant_kernel[grid](
        x, packed, scale, tensor_scale, rows, k, col_tiles, ROWS=_BLOCK_QUANT_ROWS, BLOCKS=BLOCKS_PER_TILE, BLOCK=16,
    )
    return BlockQuant(packed.view(FP4_DTYPE), scale.view(FP8_DTYPE), tensor_scale)
