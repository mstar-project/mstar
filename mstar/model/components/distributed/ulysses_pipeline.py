"""Ulysses attention with the head exchange pipelined against the attention over the copy engines.

``ulysses_attention`` brackets one attention call with two NCCL all-to-alls. NCCL's all-to-all runs on the SMs,
and beside a full-occupancy attention it slows that attention by about what it moves, so it never hides behind
compute. Here a rank's heads for each destination split into ``groups`` head groups. Group g's q/k/v rows go by
pitched copy-engine copies (``cuMemcpy2DAsync``) straight into each peer's receive slot, a CUDA IPC mapping, and a
``cuStreamWriteValue32`` queued behind them bumps the peer's flag. The peer's attention of group g waits on that
flag with ``cuStreamWaitValue32``, which holds no SM, while group g+1 is still on the copy engines. Each group's
output rows go back the same way, straight into the owner's output tensor at the group's heads, so nothing merges
them afterwards. A head attends as it does in the full call, so the result is the serving exchange's bit for bit:
the first call of each shape checks that on every rank and keeps the serving exchange for that shape otherwise.

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
    """Every rank's receive slots, output and flags, each peer's opened in this rank's context; one pool for every
    call, grown to the largest (a pool per shape would grow with every new row bucket). For a call of ``rows``: per
    head group, q, k and v slots as attention reads them, ``[world * rows, group_heads, head_dim]`` each, rank r's
    rows at ``r * rows``; the call's output, ``[rows, heads, head_dim]``, which every rank writes its heads of
    directly. ``fin[p]`` counts the head groups peer p has delivered to this rank's slots, ``fout[p]`` those it has
    written into this rank's output."""

    def __init__(self, group: CommGroup, shape: _Shape, groups: int):
        world = group.world_size
        self.shape = shape
        self.group_heads = shape.heads // (world * groups)
        per_row = self.group_heads * shape.head_dim
        self.inb = _share(group, _shareable_zeros(groups * 3 * world * shape.rows * per_row, shape.dtype))
        self.out = _share(group, _shareable_zeros(shape.rows * shape.heads * shape.head_dim, shape.dtype))
        self.fin = _share(group, _shareable_zeros(world, torch.int32))
        self.fout = _share(group, _shareable_zeros(world, torch.int32))
        self.groups_sent = 0

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
    caller then runs the serving exchange. What it returns is the pool's output buffer, valid until the next call
    on the current stream: the peers' next writes into it are ordered after everything queued before that call."""

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
        self._copy_rows = None

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
                from mstar.model.components.distributed.row_copy import copy_rows  # needs triton
                self._copy_rows = copy_rows
            except (OSError, AttributeError, RuntimeError, ImportError) as error:
                logger.warning("Ulysses pipeline: its transport is unavailable (%s)", error)
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
        need = (self.groups * 3 * world * self.group_heads(shape.heads) + shape.heads) * shape.rows \
            * shape.head_dim * torch.empty((), dtype=shape.dtype).element_size()
        held = 0
        if self._pool is not None:
            torch.cuda.synchronize()
            held = sum(t.numel() * t.element_size() for t in (self._pool.inb[self.group.rank],
                                                              self._pool.out[self.group.rank]))
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
        group, ops, groups, copy_rows = self.group, self._ops, self.groups, self._copy_rows
        world, r = group.world_size, group.rank
        rows, heads, head_dim = q.shape
        hg, esz = buf.group_heads, q.element_size()
        slot_numel = world * rows * hg * head_dim
        main = torch.cuda.current_stream()
        base, buf.groups_sent = buf.groups_sent, buf.groups_sent + groups
        parts = (q, k, v)
        peers = [(r + i) % world for i in range(1, world)]
        row_bytes, head_bytes = hg * head_dim * esz, head_dim * esz
        out_pitch = heads * head_bytes
        # addresses, not views: a tensor view per copy costs more host time than queueing the copy
        inb, outs = [t.data_ptr() for t in buf.inb], [t.data_ptr() for t in buf.out]
        fin, fout = buf.fin[r].data_ptr(), buf.fout[r].data_ptr()

        def head0(p: int, g: int) -> int:  # first head of destination p's group g
            return (p * groups + g) * hg

        def slot(g: int, part: int) -> torch.Tensor:  # this rank's
            return buf.inb[r].narrow(0, (g * 3 + part) * slot_numel, slot_numel).view(world * rows, hg, head_dim)

        ready = main.record_event()
        # this rank's own rows on the SMs, before the first attention: a same-device copy queued behind a
        # running attention would wait for it, and hold up the peer copies behind it. One pass a part, every group.
        width = hg * head_dim
        own = buf.inb[r]
        for part, t in enumerate(parts):
            copy_rows(own.as_strided((groups, rows, width), (3 * slot_numel, width, 1),
                                     own.storage_offset() + part * slot_numel + r * rows * width),
                      t.as_strided((groups, rows, width), (width, t.stride(0), 1),
                                   t.storage_offset() + head0(r, 0) * head_dim))
        self._copy_in.wait_event(ready)
        for g in range(groups):
            for p in peers:
                h0 = head0(p, g)
                for part, t in enumerate(parts):
                    dst = inb[p] + ((g * 3 + part) * slot_numel) * esz + r * rows * row_bytes
                    ops.copy2d(self._copy_in, dst, row_bytes, t.data_ptr() + h0 * head_bytes, t.stride(0) * esz,
                               row_bytes, rows)
            for p in peers:
                ops.write(self._copy_in, buf.fin[p].data_ptr() + 4 * r, base + g + 1)
        # Every rank writes its heads of this rank's rows straight in, and the caller reads it on `main` until the
        # next call. A peer's writes for that call come after its attention, which waits for this rank's q/k/v of
        # that call, sent after everything queued on `main` before the call: so after the caller has read this one.
        merged = buf.out[r].narrow(0, 0, rows * heads * head_dim).view(rows, heads, head_dim)
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
                    ops.wait(stream, fin + 4 * p, base + g + 1)
                out = attend_group(slot(g, 0), slot(g, 1), slot(g, 2)).contiguous()
                # the copy stream reads it after the next group's attention may have reused its memory
                out.record_stream(self._copy_out)
                self._copy_out.wait_event(stream.record_event())
                h0 = head0(r, g)
                for p in peers:
                    ops.copy2d(self._copy_out, outs[p] + h0 * head_bytes, out_pitch,
                               out.data_ptr() + p * rows * row_bytes, row_bytes, row_bytes, rows)
                for p in peers:
                    ops.write(self._copy_out, buf.fout[p].data_ptr() + 4 * r, base + g + 1)
                copy_rows(merged[:, h0:h0 + hg].flatten(1)[None], out[r * rows:(r + 1) * rows].flatten(1)[None])
        # A peer writes its groups from one stream, in order, so its count covers every group before it. Once
        # every peer has written, every peer has also attended over its slots, which the next call rewrites.
        for p in peers:
            ops.wait(main, fout + 4 * p, base + groups)
        # the caller may free q / k / v, which the copy-in stream reads; `main` needn't wait out this rank's own
        # sends, whose sources record_stream keeps alive
        main.wait_stream(self._copy_in)
        if self.group_streams:
            for stream in self._streams[:groups]:
                main.wait_stream(stream)
        return merged
