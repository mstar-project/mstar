"""A batched copy of row blocks between strided views, for the Ulysses pipeline's same-device copies.

Those move a few hundred contiguous bytes a row at a pitch of several KB (one head group of each row). torch's
copy runs such a view element by element; this kernel moves whole rows in vector loads, every block of a batch in
one launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_kernel(SRC, DST, rows, src_batch, src_row, dst_batch, dst_row,
                      WIDTH: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr):
    r = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)[:, None]
    b = tl.program_id(1).to(tl.int64)
    w = tl.arange(0, BLOCK_W)[None, :]
    mask = (r < rows) & (w < WIDTH)
    x = tl.load(SRC + b * src_batch + r * src_row + w, mask=mask)
    tl.store(DST + b * dst_batch + r * dst_row + w, x, mask=mask)


def copy_rows(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst[b, i] = src[b, i]`` for ``[batch, rows, width]`` views of one dtype whose rows are contiguous, on
    the current stream."""
    if dst.shape != src.shape or dst.dtype != src.dtype or dst.stride(2) != 1 or src.stride(2) != 1:
        raise ValueError(f"copy_rows: {tuple(dst.shape)}/{dst.dtype} from {tuple(src.shape)}/{src.dtype}")
    batch, rows, width = src.shape
    block_w = triton.next_power_of_2(width)
    block_r = max(1, min(64, 8192 // block_w))
    _copy_rows_kernel[(triton.cdiv(rows, block_r), batch)](
        src, dst, rows, src.stride(0), src.stride(1), dst.stride(0), dst.stride(1),
        WIDTH=width, BLOCK_R=block_r, BLOCK_W=block_w, num_warps=4,
    )
