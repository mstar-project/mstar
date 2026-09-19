"""Lamport-style one-shot all-reduce over symmetric memory, as one Triton launch.

The small all-reduces of a tensor-parallel decode step (a few KB to a few hundred KB, hundreds
per step) are latency-bound: what matters is how many times the ranks have to wait for each
other, not the bytes. The classic one-shot kernel needs a barrier before the reads (so that
every rank's input is in place) and one after (so that nobody overwrites a buffer a peer is
still reading): two cross-GPU round trips. The Lamport protocol used by TensorRT-LLM has none:

* every rank *pushes* its partial into a slot of every peer's buffer (posted NVLink writes),
* the buffer was pre-filled with a sentinel that no real value uses (``-0.0``; the sender maps a
  real ``-0.0`` to ``+0.0``), so a reader knows an element has arrived when it is not the
  sentinel: it spins on its own local buffer until all ``world`` partials are there, sums them
  in fp32 and writes the result,
* it then re-fills what it read with the sentinel. Two buffers alternate: a peer writes buffer
  ``s`` again two rounds later, and it can only get there after receiving this rank's next
  round, which this rank sends only after the kernel that did the re-fill has finished.

The round counter lives on the device and is advanced by the last program of each launch, so
the kernel is CUDA-graph safe (replays keep alternating buffers). Every rank must issue the same
sequence of all-reduces on a workspace, which tensor parallelism guarantees.

Memory model: the pushes are plain stores over the peer mapping, the polls are volatile loads
(they bypass L1, the point of coherence for peer writes is L2), and each element is written
once, so a torn 16-byte read only shows some elements still at the sentinel and is retried.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

#: bf16 / fp16 bit pattern of -0.0, the "not written yet" marker (the constexpr twin is what the
#: kernel reads; Triton kernels see only constexpr globals).
SENTINEL_I16 = -32768
_SENT = tl.constexpr(SENTINEL_I16)
#: ``ctrl`` words: [round, programs finished in this launch, spin timeouts seen, unused]
CTRL_ROUND, CTRL_DONE, CTRL_STATUS = 0, 1, 2

_MAX_PROGRAMS = 2048  # keep every program of a launch co-resident: a spinning program must not starve a pusher
_MAX_SPIN = 1 << 26  # ~ seconds at a few tens of ns per poll; then flag the timeout and give up


@triton.jit
def _lamport_all_reduce_kernel(
    x_ptr, out_ptr, ptrs_ptr, ctrl_ptr,
    D, stride_xt, stride_ot,
    slot_stride, rank_stride, row_stride,
    rank, max_spin,
    WORLD: tl.constexpr, BLOCK: tl.constexpr, CONCAT: tl.constexpr,
):
    """One ``[row t, chunk c]`` program: push, poll, then the sum (all-reduce) or, with ``CONCAT``,
    the ``world`` shards side by side (all-gather: ``out[t, p * D + j] = shard_p[t, j]``)."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    rnd = tl.load(ctrl_ptr + 0, volatile=True)
    slot = (rnd % 2).to(tl.int64)
    # push: my partial, with -0.0 mapped to +0.0, into slot [slot][rank][t] of every rank
    x = tl.load(x_ptr + t * stride_xt + offs, mask=mask, other=0.0)
    xi = x.to(tl.int16, bitcast=True)
    xi = tl.where(xi == _SENT, 0, xi)
    dst = slot * slot_stride + rank * rank_stride + t * row_stride + offs  # rank: a plain (maybe specialized) int
    for p in tl.static_range(WORLD):
        peer = tl.load(ptrs_ptr + p).to(tl.pointer_type(tl.int16))
        tl.store(peer + dst, xi, mask=mask)
    # poll my own buffer until all WORLD partials of this chunk have arrived
    mine = tl.load(ptrs_ptr + rank).to(tl.pointer_type(tl.int16))
    peers = tl.arange(0, WORLD).to(tl.int64)
    src = mine + slot * slot_stride + peers[:, None] * rank_stride + t * row_stride + offs[None, :]
    src_mask = mask[None, :] & (peers[:, None] < WORLD)
    v = tl.load(src, mask=src_mask, other=0, volatile=True)
    spins = 0
    pending = tl.max((v == _SENT).to(tl.int32)) > 0
    while pending & (spins < max_spin):
        v = tl.load(src, mask=src_mask, other=0, volatile=True)
        spins += 1
        pending = tl.max((v == _SENT).to(tl.int32)) > 0
    tl.store(ctrl_ptr + 2, 1, mask=spins >= max_spin)
    # re-arm this chunk of the slot for its next use (two rounds from now)
    tl.store(src, tl.full([WORLD, BLOCK], _SENT, tl.int16), mask=src_mask)
    if CONCAT:
        out_off = t * stride_ot + peers[:, None] * D + offs[None, :]
        tl.store(out_ptr + out_off, v.to(out_ptr.dtype.element_ty, bitcast=True), mask=src_mask)
    else:
        acc = tl.sum(v.to(x_ptr.dtype.element_ty, bitcast=True).to(tl.float32), axis=0)
        tl.store(out_ptr + t * stride_ot + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)
    # the last program to finish advances the round for the next launch
    n_progs = tl.num_programs(0) * tl.num_programs(1)
    done = tl.atomic_add(ctrl_ptr + 1, 1, sem="acq_rel")
    is_last = done == n_progs - 1
    tl.store(ctrl_ptr + 1, 0, mask=is_last)
    tl.store(ctrl_ptr + 0, rnd + 1, mask=is_last)


def lamport_supported(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.dim() == 2 and x.stride(1) == 1


def _pick_block(rows: int, width: int) -> int:
    """Elements per program: 256 (one warp, the fastest poll measured: the spin's reduction stays
    inside a warp) unless that would exceed the co-residency cap."""
    block = 256
    while rows * triton.cdiv(width, block) > _MAX_PROGRAMS and block < 8192:
        block *= 2
    return block


class LamportAllReduce:
    """One workspace = one (group, dtype, width) channel: symmetric buffers ``[2, world, max_rows,
    width]`` on every rank, the table of their addresses and the round counter. Build it
    eagerly (the rendezvous is a collective that must not run under CUDA-graph capture), then
    :meth:`all_reduce` any ``[rows <= max_rows, width]`` tensor of the channel's dtype."""

    def __init__(
        self,
        group_name: str,
        rank: int,
        world_size: int,
        max_rows: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"Lamport all-reduce needs a 16-bit float dtype, got {dtype}")
        if world_size & (world_size - 1):
            raise ValueError(f"Lamport all-reduce needs a power-of-two group, got {world_size}")
        self.rank, self.world_size = rank, world_size
        self.max_rows, self.width, self.dtype, self.device = max_rows, width, dtype, device
        self.buf = symm_mem.empty(2, world_size, max_rows, width, dtype=dtype, device=device)
        self.buf.view(torch.int16).fill_(SENTINEL_I16)
        self._handle = symm_mem.rendezvous(self.buf, group_name)
        ptrs = list(self._handle.buffer_ptrs)
        if ptrs[rank] != self.buf.data_ptr():
            raise RuntimeError("symmetric buffer does not start at its allocation")
        self.ptrs = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self.ctrl = torch.zeros(4, dtype=torch.int32, device=device)
        # every rank's sentinel fill must be complete before any rank pushes into it
        self._handle.barrier(channel=0)

    def applies(self, x: torch.Tensor) -> bool:
        return (
            lamport_supported(x) and x.dtype == self.dtype and x.shape[1] == self.width
            and x.shape[0] <= self.max_rows and x.device == self.device
        )

    def _launch(self, x: torch.Tensor, out: torch.Tensor, concat: bool, max_spin: int) -> torch.Tensor:
        assert self.applies(x), (x.shape, x.dtype, x.device, x.stride())
        assert out.dtype == x.dtype and out.stride(1) == 1 and out.shape[0] == x.shape[0], (out.shape, out.dtype)
        rows, width = x.shape
        block = _pick_block(rows, width)
        grid = (rows, triton.cdiv(width, block))
        _lamport_all_reduce_kernel[grid](
            x, out, self.ptrs, self.ctrl,
            width, x.stride(0), out.stride(0),
            self.buf.stride(0), self.buf.stride(1), self.buf.stride(2),
            self.rank, max_spin,
            WORLD=self.world_size, BLOCK=block, CONCAT=concat,
            num_warps=1 if block <= 256 else 4 if block <= 1024 else 8,
        )
        return out

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor | None = None, max_spin: int = _MAX_SPIN) -> torch.Tensor:
        """Sum ``x [rows, width]`` across the group into ``out`` (a new tensor by default)."""
        return self._launch(x, torch.empty_like(x) if out is None else out, False, max_spin)

    def all_gather(self, x: torch.Tensor, out: torch.Tensor | None = None, max_spin: int = _MAX_SPIN) -> torch.Tensor:
        """The group's shards ``x [rows, width]`` side by side: ``out [rows, world * width]`` with
        rank ``p``'s shard at columns ``[p * width, (p + 1) * width)``."""
        if out is None:
            out = torch.empty(x.shape[0], self.world_size * x.shape[1], dtype=x.dtype, device=x.device)
        assert out.shape[1] == self.world_size * x.shape[1], out.shape
        return self._launch(x, out, True, max_spin)

    def timed_out(self) -> bool:
        """Whether any launch so far gave up waiting for a peer (a host sync)."""
        return bool(self.ctrl[CTRL_STATUS].item())


class FlashInferAllReduce:
    """The same channel contract on flashinfer's TensorRT-LLM Lamport one-shot kernel
    (``flashinfer.comm.trtllm_allreduce_fusion``, pattern ``kAllReduce``, fp32 accumulation, PDL
    launch): measured 2.5 us against this module's 5 us at TP2 for a [1, 3584] bf16 message. The
    workspace (IPC-mapped Lamport buffers with device-side flags) is per (group, dtype, width) and
    CUDA-graph safe like ours. Raises ImportError when flashinfer's comm module is missing."""

    def __init__(self, group, rank: int, world_size: int, max_rows: int, width: int, dtype: torch.dtype,
                 device: torch.device):
        from flashinfer import comm as fi

        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"flashinfer all-reduce needs a 16-bit float dtype, got {dtype}")
        self.fi = fi
        self.rank, self.world_size = rank, world_size
        self.max_rows, self.width, self.dtype, self.device = max_rows, width, dtype, device
        with torch.cuda.device(device):
            self._handles, self.workspace, self.metadata = fi.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, world_size, max_rows, width, group=group, use_fp32_lamport=False, create_metadata=True)

    def applies(self, x: torch.Tensor) -> bool:
        return (
            lamport_supported(x) and x.dtype == self.dtype and x.shape[1] == self.width
            and x.shape[0] <= self.max_rows and x.device == self.device
        )

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        assert self.applies(x), (x.shape, x.dtype, x.device, x.stride())
        if out is None:
            out = torch.empty_like(x)
        fi = self.fi
        fi.trtllm_allreduce_fusion(
            allreduce_in=x.contiguous(), world_size=self.world_size, world_rank=self.rank, token_num=x.shape[0],
            hidden_dim=self.width, workspace_ptrs=self.workspace, launch_with_pdl=True,
            trigger_completion_at_end=True, fp32_acc=True, pattern_code=fi.AllReduceFusionPattern.kAllReduce,
            use_oneshot=True, allreduce_out=out, residual_in=None, residual_out=None, norm_out=None,
            quant_out=None, scale_out=None, rms_gamma=None, rms_eps=None, scale_factor=None, layout_code=None,
            metadata=self.metadata)
        return out
