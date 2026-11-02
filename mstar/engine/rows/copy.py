"""Many small byte copies in one launch: how a persistent batch's diff reaches
its rows, among them the rows that move between their slots and the dense
view its graph reads.

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


    @triton.jit
    def _apply_staged_kernel(block_ptr, which, PROGRAMS: tl.constexpr, CHUNK: tl.constexpr):
        # the table a step staged into a fixed block (``StagedTables``): its
        # segment count and word offset at words ``2 * which``, and a fixed
        # grid, so a captured graph runs whatever the step staged
        count = tl.load(block_ptr + 2 * which)
        at = tl.load(block_ptr + 2 * which + 1)
        for seg in range(tl.program_id(0).to(tl.int64), count, PROGRAMS):
            row = block_ptr + at + seg * 3
            src = tl.load(row).to(tl.pointer_type(tl.uint8))
            dst = tl.load(row + 1).to(tl.pointer_type(tl.uint8))
            nbytes = tl.load(row + 2)
            for start in range(0, nbytes, CHUNK):
                offs = start + tl.arange(0, CHUNK)
                mask = offs < nbytes
                tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


# (kernel, device, constexprs, pointer alignments) -> the kernel compiled for
# them: launched directly, a launch costs a fraction of the jit's dispatch.
# The alignments are the jit's own specialization
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


class StagedTables:
    """A fixed device block a step's upload lands in, whose tables a captured
    graph applies itself (``apply``): a header of ``PHASES`` (segment count,
    table word offset) pairs, then the tables and payload. Whoever applies it
    zeroes the counts, so a step that uploads nothing applies nothing."""

    PHASES = 3
    HEADER = 2 * PHASES * 8  # bytes
    _PROGRAMS = 64

    def __init__(self, nbytes: int, device: torch.device):
        self.words = torch.zeros(-(-nbytes // 8), dtype=torch.int64, device=device)
        self.nbytes = self.words.numel() * 8

    @property
    def payload_base(self) -> int:
        """The device address the tables and payload start at."""
        return self.words.data_ptr() + self.HEADER

    def header(self, placed: list[tuple[int, int, int]]) -> np.ndarray:
        """The header for tables ``pack`` placed at ``payload_base``."""
        if len(placed) > self.PHASES:
            raise ValueError(f"{len(placed)} phases for a block of {self.PHASES}")
        header = np.zeros(2 * self.PHASES, dtype=np.int64)
        for k, (at, count, _) in enumerate(placed):
            header[2 * k], header[2 * k + 1] = count, (self.HEADER + at) // 8
        return header

    def apply(self) -> None:
        """Run the staged phases in order and zero the counts; on the current
        stream, capturable."""
        if self.words.device.type == "cuda" and triton is not None:
            for which in range(self.PHASES):
                _apply_staged_kernel[(self._PROGRAMS,)](
                    self.words, which, PROGRAMS=self._PROGRAMS, CHUNK=_CHUNK,
                )
        else:
            raise RuntimeError(f"staged rows on {self.words.device.type} need Triton on CUDA")
        self.words[:2 * self.PHASES].zero_()

    def clear(self) -> None:
        """Drop whatever is staged (a step that never ran its graph)."""
        self.words[:2 * self.PHASES].zero_()
