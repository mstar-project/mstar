"""Many small byte copies in one launch: how a persistent batch's diff reaches
its rows, and how a step's rows move between their slots and the dense view
its graph reads.

A segment is ``(src, dst, nbytes)`` with raw addresses (``stage.py`` builds
a step's). One kernel copies every segment of a phase; segments of one phase
must not overlap one another's destinations, or read what another writes.
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
    from triton import knobs

    @triton.jit(do_not_specialize=["table_at"])
    def _copy_segments_kernel(block_ptr, table_at, CHUNK: tl.constexpr):
        seg = tl.program_id(0).to(tl.int64)
        start = tl.program_id(1).to(tl.int64) * CHUNK
        row = block_ptr + table_at + seg * 3
        nbytes = tl.load(row + 2)
        if start < nbytes:
            src = tl.load(row).to(tl.pointer_type(tl.uint8))
            dst = tl.load(row + 1).to(tl.pointer_type(tl.uint8))
            offs = start + tl.arange(0, CHUNK)
            mask = offs < nbytes
            tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


# (kernel, device, constexprs, pointer alignments) -> the kernel compiled for
# them: launched directly, a launch costs a fraction of the jit's dispatch.
# The alignments are the jit's own specialization (the gather and scatter
# slots sit 4 bytes into the step header)
_COMPILED: dict[tuple, object] = {}


def _launch(jit, grid: tuple[int, int, int], stream: int | None, args: tuple, **constexprs) -> None:
    device = args[0].device
    aligned = tuple(a.data_ptr() % 16 == 0 for a in args if isinstance(a, torch.Tensor))
    key = (jit, device, aligned, *sorted(constexprs.items()))
    kernel = _COMPILED.get(key)
    if kernel is None:
        kernel = _COMPILED[key] = jit.warmup(*args, **constexprs, grid=(1, 1, 1))
    if stream is None:
        stream = torch._C._cuda_getCurrentRawStream(device.index)
    kernel.run(
        *grid, stream, kernel.function, kernel.packed_metadata,
        kernel.launch_metadata(grid, stream, *args),
        knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
        *args,
    )


def copy_segments(
    block: torch.Tensor, host_view: np.ndarray, at: int, count: int, longest: int,
    stream: int | None = None,
) -> None:
    """Run the ``count`` segments whose table starts ``at`` bytes into
    ``block`` (``int64``; ``host_view`` is the same bytes on the host), on
    ``stream`` (a raw handle; the current stream if None)."""
    if block.device.type == "cuda" and triton is not None:
        _launch(_copy_segments_kernel, (count, -(-longest // _CHUNK), 1), stream, (block, at // 8), CHUNK=_CHUNK)
        return
    if block.device.type != "cpu" and triton is not None:
        _copy_segments_kernel[(count, triton.cdiv(longest, _CHUNK))](block, at // 8, CHUNK=_CHUNK)
        return
    rows = host_view[at:at + count * 24].view(np.int64).reshape(count, 3)
    if block.device.type == "cpu":
        for src, dst, n in rows.tolist():
            ctypes.memmove(dst, src, n)
        return
    raise RuntimeError(f"persistent rows on {block.device.type} need Triton")


class RowTable:
    """Buffers whose rows move between their resident slots and their dense
    step view as one launch (``move_rows``): ``(resident, dense, row bytes)``
    per buffer, on the device for the kernel, beside the tensors for the
    fallback."""

    def __init__(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device):
        self.pairs = pairs
        rows = [(r.data_ptr(), d.data_ptr(), r[0].numel() * r.element_size()) for r, d in pairs]
        self.table = torch.tensor(rows, dtype=torch.int64).reshape(-1, 3).to(device)
        self.longest = max((nbytes for _, _, nbytes in rows), default=0)

    def __len__(self) -> int:
        return len(self.pairs)


if triton is not None:
    @triton.jit
    def _move_rows_kernel(table_ptr, idx_ptr, GATHER: tl.constexpr, CHUNK: tl.constexpr):
        buf = tl.program_id(0).to(tl.int64)
        i = tl.program_id(1).to(tl.int64)
        start = tl.program_id(2).to(tl.int64) * CHUNK
        row = table_ptr + buf * 3
        nbytes = tl.load(row + 2)
        if start < nbytes:
            slot = tl.load(idx_ptr + i).to(tl.int64)
            resident = tl.load(row).to(tl.pointer_type(tl.uint8)) + slot * nbytes
            dense = tl.load(row + 1).to(tl.pointer_type(tl.uint8)) + i * nbytes
            offs = start + tl.arange(0, CHUNK)
            mask = offs < nbytes
            if GATHER:
                tl.store(dense + offs, tl.load(resident + offs, mask=mask), mask=mask)
            else:
                tl.store(resident + offs, tl.load(dense + offs, mask=mask), mask=mask)


def move_rows(table: RowTable, idx: torch.Tensor, n: int, gather: bool) -> None:
    """For the first ``n`` dense rows of every buffer of ``table``: copy row
    ``i`` from (``gather``) or to its resident slot ``idx[i]`` (``int32`` on
    the device). Device-only, so it can be captured."""
    if not len(table) or not n:
        return
    if idx.device.type == "cuda" and triton is not None:
        grid = (len(table), n, -(-table.longest // _CHUNK))
        _launch(_move_rows_kernel, grid, None, (table.table, idx), GATHER=gather, CHUNK=_CHUNK)
        return
    at = idx[:n].long()
    for resident, dense in table.pairs:
        if gather:
            torch.index_select(resident, 0, at, out=dense[:n])
        else:
            resident.index_copy_(0, at, dense[:n])
