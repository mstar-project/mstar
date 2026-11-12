"""A batched copy of row blocks between strided views, for the Ulysses exchange's packing.

The exchange moves one rank's heads of every token: a block of rows, each a few KB contiguous at a pitch of the
whole token's width. torch's copy of such a view runs element by element over its five-dim index (~0.7 TB/s on an
H200 at a Wan2.2 shape); this kernel moves whole rows in vector loads, every block of a ``[B0, B1]`` grid in one
launch, near the copy bandwidth."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _copy_blocks_kernel(SRC, DST, rows, n1, s0, s1, sr, d0, d1, dr,
                            WIDTH: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr):
        r = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)[:, None]
        b = tl.program_id(1).to(tl.int64)
        b0, b1 = b // n1, b % n1
        src = SRC + b0 * s0 + b1 * s1 + r * sr
        dst = DST + b0 * d0 + b1 * d1 + r * dr
        for w0 in tl.static_range(0, WIDTH, BLOCK_W):
            w = w0 + tl.arange(0, BLOCK_W)[None, :]
            mask = (r < rows) & (w < WIDTH)
            tl.store(dst + w, tl.load(src + w, mask=mask), mask=mask)


def copy_blocks(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst[a, b, i] = src[a, b, i]`` for ``[B0, B1, rows, width]`` views of one dtype whose rows are contiguous
    (any strides on the other dims), on the current stream. A plain ``copy_`` off CUDA."""
    if dst.shape != src.shape or dst.dtype != src.dtype or dst.dim() != 4:
        raise ValueError(f"copy_blocks: {tuple(dst.shape)}/{dst.dtype} from {tuple(src.shape)}/{src.dtype}")
    n0, n1, rows, width = src.shape
    if triton is None or src.device.type != "cuda" or src.stride(3) != 1 or dst.stride(3) != 1:
        dst.copy_(src)
        return
    if not src.numel():
        return
    esz = src.element_size()
    # 16-byte vectors over a row; a program takes ~32 KB
    block_w = min(triton.next_power_of_2(width), max(1, 2048 // esz))
    block_r = max(1, min(64, (32768 // esz) // block_w))
    _copy_blocks_kernel[(triton.cdiv(rows, block_r), n0 * n1)](
        src, dst, rows, n1, src.stride(0), src.stride(1), src.stride(2), dst.stride(0), dst.stride(1), dst.stride(2),
        WIDTH=width, BLOCK_R=block_r, BLOCK_W=block_w, num_warps=4,
    )
