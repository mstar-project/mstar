"""Many small byte copies in one launch: how a persistent batch's diff reaches its rows.

A segment is ``(src, dst, nbytes)`` with raw addresses. One kernel copies
every segment of a phase; segments of one phase must not overlap one
another's destinations, or read what another writes.
"""
from __future__ import annotations

import ctypes

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs; the host fallback below serves them
    triton = None

_CHUNK = 1024


if triton is not None:
    @triton.jit
    def _copy_segments_kernel(table_ptr, CHUNK: tl.constexpr):
        seg = tl.program_id(0).to(tl.int64)
        start = tl.program_id(1).to(tl.int64) * CHUNK
        nbytes = tl.load(table_ptr + seg * 3 + 2)
        if start < nbytes:
            src = tl.load(table_ptr + seg * 3).to(tl.pointer_type(tl.uint8))
            dst = tl.load(table_ptr + seg * 3 + 1).to(tl.pointer_type(tl.uint8))
            offs = start + tl.arange(0, CHUNK)
            mask = offs < nbytes
            tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def copy_segments(block: torch.Tensor, host_view: np.ndarray, at: int, count: int, longest: int) -> None:
    """Run the ``count`` segments whose table starts ``at`` bytes into
    ``block`` (``host_view`` is the same bytes on the host), on the current
    stream."""
    if block.device.type != "cpu" and triton is not None:
        table = block[at:at + count * 24].view(torch.int64)
        _copy_segments_kernel[(count, triton.cdiv(longest, _CHUNK))](table, CHUNK=_CHUNK)
        return
    rows = host_view[at:at + count * 24].view(np.int64).reshape(count, 3)
    if block.device.type == "cpu":
        for src, dst, n in rows.tolist():
            ctypes.memmove(dst, src, n)
        return
    raise RuntimeError(f"persistent rows on {block.device.type} need Triton")
