"""Ulysses attention with the head exchange pipelined against the attention over the copy engines.

``ulysses_attention`` brackets one attention call with two NCCL all-to-alls. NCCL's all-to-all runs on the SMs,
and beside a full-occupancy attention it slows that attention by about what it moves, so it never hides behind
compute. Here a rank's heads for each destination split into ``groups`` head groups. Group g's q/k/v rows go by
pitched copy-engine copies (``cuMemcpy2DAsync``) straight into each peer's receive slot, a CUDA IPC mapping, and a
``cuStreamWriteValue32`` queued behind them bumps the peer's flag. The peer's attention of group g waits on that
flag with ``cuStreamWaitValue32``, which holds no SM, while group g+1 is still on the copy engines; each group's
output rows go back the same way. A head attends as it does in the full call, so the result is the serving
exchange's bit for bit: the first call of each shape checks that on every rank and keeps the serving exchange
for that shape otherwise.

Adapted from SGLang's ``ipc_a2a_multi.py`` (Apache-2.0), cut to what a DiT's Ulysses attention needs: one
sequence, equal shares, q, k and v with one head count.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.distributed as dist

from mstar.distributed.communication import CommGroup

logger = logging.getLogger(__name__)

# (q, k, v) [rows, heads, head_dim] -> out [rows, heads, head_dim], for one head group
AttendGroup = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

_CU_MEMORYTYPE_DEVICE = 2


class _Memcpy2D(ctypes.Structure):  # CUDA_MEMCPY2D
    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t), ("srcMemoryType", ctypes.c_int),
        ("srcHost", ctypes.c_void_p), ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t), ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_int), ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_uint64),
        ("dstArray", ctypes.c_void_p), ("dstPitch", ctypes.c_size_t), ("WidthInBytes", ctypes.c_size_t),
        ("Height", ctypes.c_size_t),
    ]


class _StreamOps:
    """The driver's stream memory operations and pitched async copy, through ``libcuda``: a wait or a signal
    queued on a stream holds no SM, and a signal behind a copy lands when the copy retires."""

    def __init__(self):
        cuda = ctypes.CDLL("libcuda.so.1")
        args = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self._write, self._wait = cuda.cuStreamWriteValue32_v2, cuda.cuStreamWaitValue32_v2
        for fn in (self._write, self._wait):
            fn.argtypes, fn.restype = args, ctypes.c_int
        self._copy2d = cuda.cuMemcpy2DAsync_v2
        self._copy2d.argtypes, self._copy2d.restype = [ctypes.POINTER(_Memcpy2D), ctypes.c_void_p], ctypes.c_int
        # a driver can disable stream memory operations: find out here, before any rank relies on them
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        self.write(stream, flag.data_ptr(), 1)
        self.wait(stream, flag.data_ptr(), 1)
        stream.synchronize()

    def write(self, stream: torch.cuda.Stream, ptr: int, value: int) -> None:
        # the default flags order the stream's earlier copies before the write
        if rc := self._write(stream.cuda_stream, ptr, value & 0xFFFFFFFF, 0):
            raise RuntimeError(f"cuStreamWriteValue32 failed: CUresult {rc}")

    def wait(self, stream: torch.cuda.Stream, ptr: int, value: int) -> None:
        # CU_STREAM_WAIT_VALUE_GEQ (0) compares cyclically, so the counters may wrap
        if rc := self._wait(stream.cuda_stream, ptr, value & 0xFFFFFFFF, 0):
            raise RuntimeError(f"cuStreamWaitValue32 failed: CUresult {rc}")

    def copy2d(self, stream: torch.cuda.Stream, dst: int, dst_pitch: int, src: int, src_pitch: int,
               width: int, height: int) -> None:
        desc = _Memcpy2D(srcMemoryType=_CU_MEMORYTYPE_DEVICE, srcDevice=src, srcPitch=src_pitch,
                         dstMemoryType=_CU_MEMORYTYPE_DEVICE, dstDevice=dst, dstPitch=dst_pitch,
                         WidthInBytes=width, Height=height)
        if rc := self._copy2d(desc, stream.cuda_stream):
            raise RuntimeError(f"cuMemcpy2DAsync failed: CUresult {rc}")


def _shareable_zeros(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """Zeros for a buffer exported over CUDA IPC: outside expandable segments, which IPC can't export."""
    get = getattr(torch._C, "_accelerator_getAllocatorSettings", None)  # torch 2.12+
    # older torch has only the setter; this process sets the allocator from its environment
    settings = get() if get is not None else ",".join(
        os.environ.get(name, "") for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"))
    if "expandable_segments:True" not in settings:
        return torch.zeros(numel, dtype=dtype, device="cuda")
    torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
    try:
        return torch.zeros(numel, dtype=dtype, device="cuda")
    finally:
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")


class _Shape(NamedTuple):
    rows: int       # this rank's rows (every rank's: shares are equal)
    heads: int      # q, k and v heads on each rank before the exchange
    head_dim: int
    dtype: torch.dtype


class _Pool:
    """Every rank's receive slots, output blocks and flags, each peer's opened in this rank's context; one pool for
    every call, grown to the largest (a pool per shape would grow with every new row bucket). For a call of
    ``rows``: per head group, q, k and v slots as attention reads them, ``[world * rows, group_heads, head_dim]``
    each, rank r's rows at ``r * rows``; per (source rank, group), an output block of that rank's output rows of
    this rank's queries, ``[rows, group_heads, head_dim]``. ``fin[p]`` counts the head groups peer p has delivered,
    ``fout[g * world + p]`` the calls whose group g it has."""

    def __init__(self, group: CommGroup, shape: _Shape, groups: int):
        world = group.world_size
        self.shape = shape
        self.group_heads = shape.heads // (world * groups)
        per_row = self.group_heads * shape.head_dim
        self.inb = _share(group, _shareable_zeros(groups * 3 * world * shape.rows * per_row, shape.dtype))
        self.outb = _share(group, _shareable_zeros(world * groups * shape.rows * per_row, shape.dtype))
        self.fin = _share(group, _shareable_zeros(world, torch.int32))
        self.fout = _share(group, _shareable_zeros(groups * world, torch.int32))
        self.groups_sent = 0
        self.calls = 0

    def serves(self, shape: _Shape) -> bool:
        return shape._replace(rows=0) == self.shape._replace(rows=0) and shape.rows <= self.shape.rows


def _share(group: CommGroup, t: torch.Tensor) -> list[torch.Tensor]:
    """Every member's ``t``, the peers' opened over CUDA IPC in this rank's context. A group collective."""
    from torch.multiprocessing.reductions import reduce_tensor

    handles: list = [None] * group.world_size
    dist.all_gather_object(handles, reduce_tensor(t), group=group.cpu_group or group.device_group)
    device = torch.cuda.current_device()
    out = []
    for member, (rebuild, args) in enumerate(handles):
        if member == group.rank:
            out.append(t)
            continue
        args = list(args)
        for i, value in enumerate(args):
            # open the peer's allocation from this rank's device, not the device it was made on
            if isinstance(value, torch.device):
                args[i] = torch.device("cuda", device)
            elif isinstance(value, int) and i == 6:  # rebuild_cuda_tensor's device index
                args[i] = device
        out.append(rebuild(*args))
    return out


class UlyssesPipeline:
    """One SP group's pipelined Ulysses exchange. ``__call__`` returns None, on every rank alike, when it can't
    serve a call (no stream memory operations, a shape it hasn't verified bit-identical, a capture), and the
    caller then runs the serving exchange."""

    def __init__(self, group: CommGroup, groups: int, group_streams: bool = False):
        self.group = group
        self.groups = int(groups)
        # each group attends on a stream of its own, so one group's last wave of tiles can run beside the next
        # group's first; off by default: on H3 at SP4 it measured the same as one stream (2 and 7 groups)
        self.group_streams = bool(group_streams)
        self._streams: list[torch.cuda.Stream] = []
        self._ops: _StreamOps | None = None
        self._usable: bool | None = None
        self._pool: _Pool | None = None
        self._verdict: dict[_Shape, bool] = {}
        self._copy_in = self._copy_out = None

    def group_heads(self, heads: int) -> int:
        return heads // (self.group.world_size * self.groups)

    def supports(self, heads: int) -> bool:
        return self.groups > 0 and heads % (self.group.world_size * self.groups) == 0

    def _ready(self) -> bool:
        """Whether every rank can use the transport; decided once, on every rank alike."""
        if self._usable is None:
            ok = 1
            try:
                self._ops = _StreamOps()
            except (OSError, AttributeError, RuntimeError) as error:
                logger.warning("Ulysses pipeline: stream memory operations unavailable (%s)", error)
                ok = 0
            self._usable = bool(self.group.agree([ok])[0])
            if self._usable:
                self._copy_in, self._copy_out = torch.cuda.Stream(), torch.cuda.Stream()
        return self._usable

    @torch.compiler.disable
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attend_group: AttendGroup,
                 sequential: Callable[[], torch.Tensor]) -> torch.Tensor | None:
        """``q``/``k``/``v``: this rank's ``[rows, heads, head_dim]``, each row's heads one contiguous run (any
        row stride), rows equal on every rank;
        ``attend_group`` attends one head group's slots; ``sequential()`` is the serving exchange's result,
        run once per shape to check the pipeline against it. Returns this rank's ``[rows, heads, head_dim]``."""
        if (q.device.type != "cuda" or torch.cuda.is_current_stream_capturing() or not self.supports(q.shape[1])
                or not all(t.shape == q.shape and t.dtype == q.dtype and t.stride(2) == 1 and t.stride(1) == q.shape[2]
                           for t in (q, k, v))
                or not self._ready()):
            return None
        shape = _Shape(q.shape[0], q.shape[1], q.shape[2], q.dtype)
        verdict = self._verdict.get(shape)
        if verdict is False:
            return None
        if (self._pool is None or not self._pool.serves(shape)) and not self._grow(shape):
            self._verdict[shape] = False
            return None
        out = self._run(q, k, v, attend_group, self._pool)
        if verdict is None:
            reference = sequential()
            same = self.group.agree([int(torch.equal(out, reference))])[0]
            self._verdict[shape] = bool(same)
            logger.info("Ulysses pipeline over %d head groups at %s: %s", self.groups, tuple(shape),
                        "bit-identical to the serving exchange; kept" if same else "differs; serving exchange kept")
            return out if same else reference
        return out

    def _grow(self, shape: _Shape) -> bool:
        """A pool for ``shape``, as a call outgrew the last one (or there was none); False, on every rank, when
        some rank can't spare it twice over, and the shape keeps the serving exchange. Every rank reaches the same
        call with the same shape, in lockstep; each waits out its queued copies into the old pool before it goes."""
        world = self.group.world_size
        need = (self.groups * 3 * world + world * self.groups) * shape.rows * self.group_heads(shape.heads) \
            * shape.head_dim * torch.empty((), dtype=shape.dtype).element_size()
        held = 0
        if self._pool is not None:
            torch.cuda.synchronize()
            held = sum(t.numel() * t.element_size() for t in (self._pool.inb[self.group.rank],
                                                              self._pool.outb[self.group.rank]))
        free, _ = torch.cuda.mem_get_info()
        spare = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated() + held
        if not self.group.agree([int(spare >= 2 * need)])[0]:
            logger.info("Ulysses pipeline: its %.2f GiB of buffers for %s don't fit on every rank; serving exchange",
                        need / 2**30, tuple(shape))
            return False
        self._pool = None
        self._pool = _Pool(self.group, shape, self.groups)
        return True

    def _run(self, q, k, v, attend_group: AttendGroup, buf: _Pool) -> torch.Tensor:
        group, ops, groups = self.group, self._ops, self.groups
        world, r = group.world_size, group.rank
        rows, heads, head_dim = q.shape
        hg, esz = buf.group_heads, q.element_size()
        slot_numel = world * rows * hg * head_dim
        out_numel = rows * hg * head_dim
        main = torch.cuda.current_stream()
        base, buf.groups_sent = buf.groups_sent, buf.groups_sent + groups
        buf.calls += 1
        call = buf.calls
        parts = (q, k, v)
        peers = [(r + i) % world for i in range(1, world)]
        row_bytes, out_pitch = hg * head_dim * esz, heads * head_dim * esz

        def head0(p: int, g: int) -> int:  # first head of destination p's group g
            return (p * groups + g) * hg

        def slot(member: int, g: int, part: int) -> torch.Tensor:
            start = (g * 3 + part) * slot_numel
            return buf.inb[member].narrow(0, start, slot_numel).view(world * rows, hg, head_dim)

        ready = main.record_event()
        # this rank's own rows on the SMs, before the first attention: a same-device copy queued behind a
        # running attention would wait for it, and hold up the peer copies behind it
        for g in range(groups):
            h0 = head0(r, g)
            for part, t in enumerate(parts):
                slot(r, g, part)[r * rows:(r + 1) * rows].copy_(t[:, h0:h0 + hg])
        self._copy_in.wait_event(ready)
        for g in range(groups):
            for p in peers:
                h0 = head0(p, g)
                for part, t in enumerate(parts):
                    ops.copy2d(self._copy_in, slot(p, g, part)[r * rows:].data_ptr(), row_bytes,
                               t.data_ptr() + h0 * head_dim * esz, t.stride(0) * esz, row_bytes, rows)
            for p in peers:
                ops.write(self._copy_in, buf.fin[p].data_ptr() + 4 * r, base + g + 1)
        merged = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        if self.group_streams:
            while len(self._streams) < groups:
                self._streams.append(torch.cuda.Stream())
            placed = main.record_event()
        for g in range(groups):
            stream = self._streams[g] if self.group_streams else main
            if stream is not main:
                stream.wait_event(placed)
            with torch.cuda.stream(stream):
                for p in peers:
                    ops.wait(stream, buf.fin[r].data_ptr() + 4 * p, base + g + 1)
                out = attend_group(slot(r, g, 0), slot(r, g, 1), slot(r, g, 2)).contiguous()
                # the copy stream reads it after the next group's attention may have reused its memory
                out.record_stream(self._copy_out)
                self._copy_out.wait_event(stream.record_event())
                for p in peers:
                    block = buf.outb[p].narrow(0, (r * groups + g) * out_numel, out_numel)
                    ops.copy2d(self._copy_out, block.data_ptr(), row_bytes, out[p * rows:].data_ptr(), row_bytes,
                               row_bytes, rows)
                for p in peers:
                    ops.write(self._copy_out, buf.fout[p].data_ptr() + 4 * (g * world + r), call)
                h0 = head0(r, g)
                merged[:, h0:h0 + hg].copy_(out[r * rows:(r + 1) * rows])
        for g in range(groups):
            for p in peers:
                ops.wait(self._copy_out, buf.fout[r].data_ptr() + 4 * (g * world + p), call)
                block = buf.outb[r].narrow(0, (p * groups + g) * out_numel, out_numel)
                ops.copy2d(self._copy_out, merged.data_ptr() + head0(p, g) * head_dim * esz, out_pitch,
                           block.data_ptr(), row_bytes, row_bytes, rows)
        # the next call rewrites the slots and the caller may free q / k / v, which the copy streams read
        main.wait_stream(self._copy_out)
        main.wait_stream(self._copy_in)
        if self.group_streams:
            for stream in self._streams[:groups]:
                main.wait_stream(stream)
        return merged
