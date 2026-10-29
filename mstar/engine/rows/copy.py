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


class SegmentTable:
    """Segments as ``int64`` columns. A source may be an offset into the
    step's uploaded payload, resolved to an address once the block it lands
    in is known."""

    def __init__(self):
        self._src: list[np.ndarray] = []
        self._dst: list[np.ndarray] = []
        self._len: list[np.ndarray] = []
        self._rel: list[np.ndarray] = []

    def _add(self, src, dst, nbytes, payload: bool) -> None:
        src = np.asarray(src, dtype=np.int64)
        self._src.append(src)
        self._dst.append(np.asarray(dst, dtype=np.int64))
        self._len.append(np.broadcast_to(np.int64(nbytes), src.shape))
        self._rel.append(np.full(src.shape, payload))

    def add_rows(self, src_ptr: int, src_rows, dst_ptr: int, dst_rows, row_bytes: int) -> None:
        """``dst[dst_rows[i]] = src[src_rows[i]]``, rows of ``row_bytes`` each."""
        src_rows, dst_rows = np.asarray(src_rows, np.int64), np.asarray(dst_rows, np.int64)
        self._add(src_ptr + src_rows * row_bytes, dst_ptr + dst_rows * row_bytes, row_bytes, False)

    def add_fill(self, payload_at: int, dst_ptr: int, dst_rows, row_bytes: int) -> None:
        """Every ``dst_rows`` row set to one payload row."""
        dst_rows = np.asarray(dst_rows, np.int64)
        self._add(np.full(dst_rows.shape, payload_at), dst_ptr + dst_rows * row_bytes, row_bytes, True)

    def add_payload_rows(self, payload_at: int, dst_ptr: int, dst_rows, row_bytes: int) -> None:
        """Consecutive payload rows into ``dst_rows``."""
        dst_rows = np.asarray(dst_rows, np.int64)
        src = payload_at + np.arange(len(dst_rows), dtype=np.int64) * row_bytes
        self._add(src, dst_ptr + dst_rows * row_bytes, row_bytes, True)

    def add_payload_elements(self, payload_at: int, dst_ptr: int, flat_index, item: int) -> None:
        """Consecutive payload elements into elements ``flat_index`` of ``dst``."""
        flat_index = np.asarray(flat_index, np.int64)
        src = payload_at + np.arange(len(flat_index), dtype=np.int64) * item
        self._add(src, dst_ptr + flat_index * item, item, True)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        """``[n, 3]`` rows of (src, dst, nbytes), and which sources are payload offsets."""
        if not self._src:
            return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=bool)
        table = np.stack(
            [np.concatenate(self._src), np.concatenate(self._dst), np.concatenate(self._len)], axis=1,
        )
        return table, np.concatenate(self._rel)


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
