"""Two-GPU, two-layer-MLP latency: CTA-pipelining vs micro-batching vs TP2.

Replicates Fig. 5 of arXiv:2607.07862 on Wan2.2-TI2V-5B FFN shapes (hidden 3072,
ffn 14336, gelu-tanh, bias) instead of the paper's square 8192x8192 GEMMs::

    python -m benchmark.cta_pipelining.bench_mlp_2gpu                # Wan2.2 shapes
    python -m benchmark.cta_pipelining.bench_mlp_2gpu --paper        # 8192x8192, no act/bias
    python -m benchmark.cta_pipelining.bench_mlp_2gpu --tokens 8192 --modes single,ctapipe

Modes (all bf16, fp32 accumulate):

* ``single``      one GPU, cuBLAS: fc1 -> act -> fc2. The reference for correctness.
* ``microbatch``  two GPUs, cuBLAS: rows split into chunks; GPU 0 runs fc1 on chunk
                  i while GPU 1 runs fc2 on chunk i-1 (paper Fig. 3a). Chunk size
                  is swept and the best is reported, as the paper does.
* ``tp2``         two processes, Megatron-style TP over NCCL: fc1 column-sharded,
                  fc2 row-sharded, one all-reduce (paper Sec. IV-B baseline).
* ``ctapipe``     this repo's CTA-pipelined Triton kernels (paper Fig. 3b).
* ``ctapipe_cutlass``  the same protocol on the CUTLASS (CuTe DSL) warp-specialised
                  persistent WGMMA kernels (``mstar/utils/cta_pipelining/cute_gemm.py``).
* ``single_cutlass``   one GPU, the CuTe DSL kernel in its plain role: fc1 -> act -> fc2.
                  Separates kernel quality from protocol overhead.
* ``tp2_cute``    TP2's shards in one process on the plain CuTe kernels, then a peer copy of the
                  partial rows and an add (no overlap): isolates the overlap gain of ``tp2_overlap``
                  from kernel differences.
* ``tp2_overlap`` ``CuteTP2OverlapMLP``: TP2's shards with the partial exchange fused into the fc2
                  kernel (``ROLE_REDUCE``). X on both GPUs, like ``tp2``.

``--tp2-output sharded,replicated`` picks the output of ``tp2_cute`` / ``tp2_overlap``: sharded =
each GPU returns its own row blocks (a reduce-scatter), replicated = the full y on both (like
``tp2``'s all-reduce; columns ``-rep``).

``--producer-share f1,f2,..`` sweeps the CuTe CTA-pipe's fc1 column split (GPU 1 computes the
last ``1 - f`` of H itself; X is copied to GPU 1 inside the forward). ``--x-replicated`` also
times it with X already on GPU 1 (``x_consumer``), which is what TP2 assumes. ``--graph`` also
times it with ``use_graph=True`` (one CUDA-graph replay per forward; columns ``-g``).

Reports the median of ``--iters`` timed runs after ``--warmup`` and the paper's
two reductions: R-MB (vs best micro-batch) and R-TP (vs TP2).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cuda.bindings import runtime as cudart

from mstar.utils.cta_pipelining import CTAPipelinedMLP, mlp_reference, verify_peer_kernel_access
from mstar.utils.cta_pipelining.cute_gemm import ROLE_PLAIN, CuteGemmOp
from mstar.utils.cta_pipelining.cute_mlp import PLAIN_RASTER_G, CuteCTAPipelinedMLP, CutePlainMLP
from mstar.utils.cta_pipelining.cute_tp2 import CuteTP2OverlapMLP

DTYPE = torch.bfloat16


def _act(h: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu_tanh":
        return F.gelu(h, approximate="tanh")
    if activation == "silu":
        return F.silu(h)
    return h


def make_weights(K: int, N: int, bias: bool, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w1 = (torch.randn(N, K, generator=g) * K**-0.5).to(DTYPE)
    w2 = (torch.randn(K, N, generator=g) * N**-0.5).to(DTYPE)
    b1 = (torch.randn(N, generator=g) * 0.02).to(DTYPE) if bias else None
    b2 = (torch.randn(K, generator=g) * 0.02).to(DTYPE) if bias else None
    return w1.to(device), b1 if b1 is None else b1.to(device), w2.to(device), b2 if b2 is None else b2.to(device)


def time_ms(fn, *, device: torch.device, devices: list[torch.device], iters: int, warmup: int) -> float:
    """Median wall time of ``fn`` measured with events on ``device``'s current
    stream; every device is synchronized between runs so no work overlaps."""
    for _ in range(warmup):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    times = []
    stream = torch.cuda.current_stream(device)
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn()
        end.record(stream)
        for d in devices:
            torch.cuda.synchronize(d)
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def time_ms_b2b(fn, *, device: torch.device, devices: list[torch.device], iters: int, warmup: int,
                join=None) -> float:
    """Back-to-back: ``iters`` calls queued without a host sync, ``(end - start) / iters`` on ``device``'s
    current stream (the production condition: host launch cost hides behind the GPU work). ``join``
    runs once after the loop and must make ``device``'s stream wait for the other devices' work."""
    for _ in range(warmup):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    stream = torch.cuda.current_stream(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        fn()
    if join is not None:
        join()
    end.record(stream)
    for d in devices:
        torch.cuda.synchronize(d)
    return start.elapsed_time(end) / iters


# --------------------------------------------------------------------------- single GPU
def bench_single(x, w1, b1, w2, b2, activation, iters, warmup):
    d0 = x.device
    fn = lambda: mlp_reference(x, w1, b1, w2, b2, activation)  # noqa: E731
    return time_ms(fn, device=d0, devices=[d0], iters=iters, warmup=warmup)


# --------------------------------------------------------------------------- micro-batching
class MicroBatchMLP:
    """Static chunk pipelining across two GPUs with cuBLAS (paper Fig. 3a).

    Streams: ``s0`` computes fc1 on GPU 0 and issues the H chunk copy; ``s1_in``
    receives it on GPU 1 (so the copy never waits on GPU 1's compute); ``s1``
    runs fc2; ``s0_out`` receives Y on GPU 0. PyTorch's cross-device ``copy_``
    runs on the source's current stream and makes the destination's current
    stream wait, which is why receive streams are separate from compute.
    """

    def __init__(self, w1, b1, w2, b2, d0, d1, activation):
        self.w1, self.b1 = w1.to(d0), None if b1 is None else b1.to(d0)
        self.w2, self.b2 = w2.to(d1), None if b2 is None else b2.to(d1)
        self.d0, self.d1, self.activation = d0, d1, activation
        self.s0, self.s0_out = torch.cuda.Stream(device=d0), torch.cuda.Stream(device=d0)
        self.s1, self.s1_in = torch.cuda.Stream(device=d1), torch.cuda.Stream(device=d1)

    def __call__(self, x: torch.Tensor, chunk: int) -> torch.Tensor:
        M = x.shape[0]
        caller = torch.cuda.current_stream(self.d0)
        y = torch.empty((M, self.w2.shape[0]), dtype=x.dtype, device=self.d0)
        self.s0.wait_stream(caller)
        self.s0_out.wait_stream(caller)
        for i in range(0, M, chunk):
            with torch.cuda.stream(self.s0):
                h = _act(F.linear(x[i : i + chunk], self.w1, self.b1), self.activation)
                with torch.cuda.stream(self.s1_in):
                    h1 = h.to(self.d1, non_blocking=True)
            with torch.cuda.stream(self.s1):
                self.s1.wait_stream(self.s1_in)
                h1.record_stream(self.s1)
                yi = F.linear(h1, self.w2, self.b2)
                with torch.cuda.stream(self.s0_out):
                    y[i : i + chunk].copy_(yi, non_blocking=True)
        caller.wait_stream(self.s0_out)
        caller.wait_stream(self.s0)
        return y


def bench_microbatch(x, w1, b1, w2, b2, activation, d1, chunks, iters, warmup, ref):
    mb = MicroBatchMLP(w1, b1, w2, b2, x.device, d1, activation)
    results = {}
    for chunk in chunks:
        if chunk > x.shape[0]:
            continue
        y = mb(x, chunk)
        torch.cuda.synchronize(x.device)
        err = (y.float() - ref.float()).abs().max().item()
        if err > 0.1 * ref.float().abs().max().item():
            raise RuntimeError(f"microbatch chunk={chunk} mismatch, max abs err {err}")
        results[chunk] = time_ms(
            lambda c=chunk: mb(x, c), device=x.device, devices=[x.device, d1], iters=iters, warmup=warmup
        )
    return results


# --------------------------------------------------------------------------- CTA-pipelining
def _check(name, y, ref):
    err = (y.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    print(f"    {name} correctness: max |err| = {err:.4g} (max |ref| = {scale:.4g})")
    if err > 0.1 * scale:
        raise RuntimeError(f"{name} does not match the cuBLAS reference")


def bench_ctapipe(x, w1, b1, w2, b2, activation, d1, iters, warmup, ref, cls=CTAPipelinedMLP, name="ctapipe"):
    verify_peer_kernel_access(x.device, d1)
    verify_peer_kernel_access(d1, x.device)
    mlp = cls(
        w1, b1, w2, b2, producer_device=x.device, consumer_device=d1, activation=activation, max_tokens=x.shape[0]
    )
    y = mlp(x)
    torch.cuda.synchronize(x.device)
    torch.cuda.synchronize(d1)
    _check(name, y, ref)
    return time_ms(lambda: mlp(x), device=x.device, devices=[x.device, d1], iters=iters, warmup=warmup)


_CUTE_KERNELS: dict[str, tuple] = {}  # MLP attribute -> compiled kernel; one variant per process


def bench_ctapipe_cutlass(x, w1, b1, w2, b2, activation, d1, iters, warmup, ref, shares, x_replicated, graph=False):
    """CuTe CTA-pipe per producer share: {(f, x_replicated, use_graph): ms}. Every MLP in this
    process is the same kernel variant (shapes are dynamic), so the compiled kernels are shared."""
    verify_peer_kernel_access(x.device, d1)
    verify_peer_kernel_access(d1, x.device)
    xc = x.to(d1) if x_replicated else None  # placed once, outside the timed region
    res, tiles = {}, {}
    for f in shares:
        for g in (False, True) if graph else (False,):
            mlp = CuteCTAPipelinedMLP(
                w1, b1, w2, b2, producer_device=x.device, consumer_device=d1, activation=activation,
                max_tokens=x.shape[0], producer_share=f, use_graph=g,
            )
            ops = {k: getattr(mlp, k) for k in ("producer", "local", "consumer") if getattr(mlp, k) is not None}
            for k, op in ops.items():
                if k in _CUTE_KERNELS:
                    op._compiled, op._compiled_key = _CUTE_KERNELS[k]
            tiles[f] = f"{mlp.N1p // mlp.tile_n}/{mlp.N1 // mlp.tile_n}"
            for xr in (False, True) if x_replicated else (False,):
                fwd = (lambda m=mlp: m(x, xc)) if xr else (lambda m=mlp: m(x))
                y = fwd()
                torch.cuda.synchronize(x.device)
                torch.cuda.synchronize(d1)
                _check(f"ctapipe_cutlass f={tiles[f]}{' xr' if xr else ''}{' graph' if g else ''}", y, ref)
                for k, op in ops.items():
                    _CUTE_KERNELS.setdefault(k, (op._compiled, op._compiled_key))
                res[(f, xr, g)] = time_ms(fwd, device=x.device, devices=[x.device, d1], iters=iters, warmup=warmup)
                print(f"  CTA-pipe (CuTe DSL) : {res[(f, xr, g)]:9.3f} ms  share {tiles[f]} tiles"
                      f"{', x replicated' if xr else ''}{', graph' if g else ''}")
            del mlp
    return res, tiles


def bench_single_cutlass(x, w1, b1, w2, b2, activation, iters, warmup, ref):
    mlp = CutePlainMLP(w1, b1, w2, b2, activation=activation)
    y = mlp(x)
    torch.cuda.synchronize(x.device)
    _check("single_cutlass", y, ref)
    return time_ms(lambda: mlp(x), device=x.device, devices=[x.device], iters=iters, warmup=warmup)


# --------------------------------------------------------------------------- TP2 in one process (CuTe)
class CuteTP2MLP:
    """TP2 sharding in one process with the plain CuTe kernels and no overlap: per GPU fc1 shard
    (G = 32) and fc2 shard into a full-M partial (b2 in GPU 0's, as in ``tp2``), then each GPU
    pushes the rows the peer needs with a copy-engine ``cudaMemcpyAsync`` on its own stream (both
    directions at once) and the owner adds them. Everything runs on one non-blocking side stream
    per GPU (as in ``CuteTP2OverlapMLP``): a peer copy on the legacy default stream serializes
    with the other GPU's default stream. H, the partial and the receive buffer are persistent (as
    H / S in ``CuteTP2OverlapMLP``); per-call allocations thrash the caching allocator when many
    calls are queued back-to-back. sharded: the row-block split of
    ``CuteTP2OverlapMLP.split``; replicated: the full partials are exchanged (same bytes as
    reduce-scatter + all-gather)."""

    def __init__(self, w1, b1, w2, b2, d0, d1, activation, output):
        half = w1.shape[0] // 2
        self.devs, self.output = (d0, d1), output
        self.w1 = [w1[g * half : (g + 1) * half].to(d).contiguous() for g, d in enumerate(self.devs)]
        self.b1 = [None if b1 is None else b1[g * half : (g + 1) * half].to(d).contiguous()
                   for g, d in enumerate(self.devs)]
        self.w2 = [w2[:, g * half : (g + 1) * half].to(d).contiguous() for g, d in enumerate(self.devs)]
        self.b2 = [None if b2 is None else b2.to(d0).contiguous(), None]
        self.fc1 = CuteGemmOp(ROLE_PLAIN, activation, b1 is not None, raster_group=PLAIN_RASTER_G)  # both GPUs
        self.fc2 = [CuteGemmOp(ROLE_PLAIN, "none", b is not None) for b in self.b2]
        self.streams = [torch.cuda.Stream(device=d) for d in self.devs]
        self._buf: list[tuple[torch.Tensor, ...]] = []  # (h, p, recv) per GPU, grown to the largest M
        self._ev_add: list[torch.cuda.Event | None] = [None, None]  # previous call's add (reads recv)

    def _buffers(self, M, N2, dtype):
        if not self._buf or self._buf[0][0].shape[0] < M:
            for d in self.devs:
                torch.cuda.synchronize(d)
            self._buf = [tuple(torch.empty((M, n), dtype=dtype, device=d) for n in (self.w1[g].shape[0], N2, N2))
                         for g, d in enumerate(self.devs)]
            self._ev_add = [None, None]
            for op in (self.fc1, *self.fc2):
                op.clear_cache()  # cached conversions of the old buffers
        return self._buf

    def __call__(self, x0, x1):
        M, N2 = x0.shape[0], self.w2[0].shape[0]
        m_split = (-(-M // 128) + 1) // 2
        r0 = min(M, m_split * 128)
        rep = self.output == "replicated"
        rows = ((0, M), (0, M)) if rep else ((0, r0), (r0, M))  # rows GPU g ends up with
        cur, s = [torch.cuda.current_stream(d) for d in self.devs], self.streams
        h, p, recv = zip(*((b[0][:M], b[1][:M], b[2]) for b in self._buffers(M, N2, x0.dtype)), strict=True)
        xs, ys, ev_in = (x0, x1), [], []
        for g, d in enumerate(self.devs):  # outputs on the caller's streams, written on the side streams
            with torch.cuda.device(d):
                ys.append(torch.empty((rows[g][1] - rows[g][0], N2), dtype=x0.dtype, device=d))
                for t in (xs[g], ys[g]):
                    t.record_stream(s[g])
                e = torch.cuda.Event()
                e.record(cur[g])
                ev_in.append(e)
        for g, d in enumerate(self.devs):
            with torch.cuda.device(d), torch.cuda.stream(s[g]):
                s[g].wait_event(ev_in[g])
                self.fc1.launch(xs[g], self.w1[g], self.b1[g], h[g], None, device=d, stream=s[g],
                                static=("w", "bias", "out"))
        ev_copy = []
        for g, d in enumerate(self.devs):
            q = 1 - g
            with torch.cuda.device(d), torch.cuda.stream(s[g]):
                self.fc2[g].launch(h[g], self.w2[g], self.b2[g], p[g], None, device=d, stream=s[g],
                                   static=("a", "w", "bias", "out"))
                if self._ev_add[q] is not None:
                    s[g].wait_event(self._ev_add[q])  # the peer's previous add has read its receive buffer
                lo, hi = rows[q]
                if hi > lo:
                    (err,) = cudart.cudaMemcpyAsync(recv[q].data_ptr(), p[g][lo:hi].data_ptr(), (hi - lo) * N2 * 2,
                                                    cudart.cudaMemcpyKind.cudaMemcpyDefault, s[g].cuda_stream)
                    assert err == cudart.cudaError_t.cudaSuccess, err
                e = torch.cuda.Event()
                e.record(s[g])
                ev_copy.append(e)
        for g, d in enumerate(self.devs):
            with torch.cuda.device(d), torch.cuda.stream(s[g]):
                s[g].wait_event(ev_copy[1 - g])
                lo, hi = rows[g]
                torch.add(p[g][lo:hi], recv[g][: hi - lo], out=ys[g])
                e = torch.cuda.Event()
                e.record(s[g])
            cur[g].wait_event(e)
            self._ev_add[g] = e
        return tuple(ys)


# --------------------------------------------------------------------------- TP2 (two processes)
def _tp2_worker(rank, world, port, M, K, N, activation, bias, iters, warmup, queue, b2b=False):
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    w1, b1, w2, b2 = make_weights(K, N, bias, torch.device("cpu"))
    shard = N // world
    w1s = w1[rank * shard : (rank + 1) * shard].to(dev)
    b1s = None if b1 is None else b1[rank * shard : (rank + 1) * shard].to(dev)
    w2s = w2[:, rank * shard : (rank + 1) * shard].to(dev)
    b2s = b2.to(dev) if (b2 is not None and rank == 0) else None  # bias added once, on rank 0
    x = torch.randn(M, K, device=dev, dtype=DTYPE, generator=torch.Generator(device=dev).manual_seed(1))

    def step():
        h = _act(F.linear(x, w1s, b1s), activation)
        y = F.linear(h, w2s, b2s)
        dist.all_reduce(y)
        return y

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(dev)
    dist.barrier()
    times = []
    for _ in range(iters):
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize(dev)
        times.append(start.elapsed_time(end))
    t_b2b = 0.0
    if b2b:  # iters steps queued without a host sync (as in b5_gate_tp2_ar.py)
        dist.barrier()
        torch.cuda.synchronize(dev)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize(dev)
        t_b2b = start.elapsed_time(end) / iters
    t = torch.tensor([statistics.median(times), t_b2b], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        queue.put(tuple(t.tolist()) if b2b else t[0].item())
    dist.destroy_process_group()


def bench_tp2(M, K, N, activation, bias, iters, warmup, port=29577, b2b=False):
    ctx = torch.multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_tp2_worker, args=(r, 2, port, M, K, N, activation, bias, iters, warmup, queue, b2b))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    result = queue.get()
    for p in procs:
        p.join()
    return result


# --------------------------------------------------------------------------- main
def _check_tp2(name, ys, ref, output):
    y = torch.cat([ys[0], ys[1].to(ys[0].device)]) if output == "sharded" else ys[0]
    _check(name, y, ref)
    if output == "replicated":
        _check(name + " (GPU 1 copy)", ys[1].to(ref.device), ref)


def _joined(fn, d0, d1):
    """``fn`` then make ``d0``'s current stream wait for ``d1``'s (outputs are ready per device)."""
    def run():
        fn()
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(d1))
        torch.cuda.current_stream(d0).wait_event(ev)
    return run


def _share_kernels(ops: dict, tag: str):
    """Reuse this process's compiled kernels across MLP instances (one variant per key); returns a
    callback that records the ones compiled by the first call."""
    for k, op in ops.items():
        if (tag, k) in _CUTE_KERNELS:
            op._compiled, op._compiled_key = _CUTE_KERNELS[(tag, k)]

    def record():
        for k, op in ops.items():
            _CUTE_KERNELS.setdefault((tag, k), (op._compiled, op._compiled_key))
    return record


def _b2b(fn, d0, d1, iters, warmup):
    return time_ms_b2b(fn, device=d0, devices=[d0, d1], iters=iters, warmup=warmup, join=_joined(lambda: None, d0, d1))


def bench_tp2_cute(x, w1, b1, w2, b2, activation, d1, iters, warmup, ref, outputs, b2b=False):
    verify_peer_kernel_access(x.device, d1)
    verify_peer_kernel_access(d1, x.device)
    x1 = x.to(d1)
    res = {}
    for out in outputs:
        mlp = CuteTP2MLP(w1, b1, w2, b2, x.device, d1, activation, out)
        record = _share_kernels({"fc1": mlp.fc1, "fc2_0": mlp.fc2[0], "fc2_1": mlp.fc2[1]}, "tp2_cute")
        ys = mlp(x, x1)
        record()
        torch.cuda.synchronize(x.device)
        torch.cuda.synchronize(d1)
        _check_tp2(f"tp2_cute {out}", ys, ref, out)
        del ys
        res[out] = time_ms(_joined(lambda m=mlp: m(x, x1), x.device, d1), device=x.device, devices=[x.device, d1],
                           iters=iters, warmup=warmup)
        print(f"  TP2 CuTe, no overlap: {res[out]:9.3f} ms  {out}")
        if b2b:
            res[out + "_b2b"] = _b2b(lambda m=mlp: m(x, x1), x.device, d1, iters, warmup)
            print(f"  TP2 CuTe, no overlap: {res[out + '_b2b']:9.3f} ms  {out}, back-to-back")
        del mlp
    return res


def bench_tp2_overlap(x, w1, b1, w2, b2, activation, d1, iters, warmup, ref, outputs, b2b=False):
    verify_peer_kernel_access(x.device, d1)
    verify_peer_kernel_access(d1, x.device)
    x1 = x.to(d1)
    res = {}
    for out in outputs:
        mlp = CuteTP2OverlapMLP(w1, b1, w2, b2, devices=(x.device, d1), activation=activation,
                                max_tokens=x.shape[0], output=out)
        record = _share_kernels({"fc1": mlp.fc1, "fc2": mlp.fc2}, f"tp2_overlap {out}")
        ys = mlp(x, x1)
        record()
        torch.cuda.synchronize(x.device)
        torch.cuda.synchronize(d1)
        _check_tp2(f"tp2_overlap {out}", ys, ref, out)
        del ys
        res[out] = time_ms(_joined(lambda m=mlp: m(x, x1), x.device, d1), device=x.device, devices=[x.device, d1],
                           iters=iters, warmup=warmup)
        print(f"  TP2 overlap (CuTe)  : {res[out]:9.3f} ms  {out}")
        if b2b:
            res[out + "_b2b"] = _b2b(lambda m=mlp: m(x, x1), x.device, d1, iters, warmup)
            print(f"  TP2 overlap (CuTe)  : {res[out + '_b2b']:9.3f} ms  {out}, back-to-back")
        del mlp
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="4096,8192,16384,32768", help="comma-separated M values")
    ap.add_argument("--k", type=int, default=3072, help="hidden size (fc1 in / fc2 out)")
    ap.add_argument("--n", type=int, default=14336, help="ffn size (fc1 out / fc2 in)")
    ap.add_argument("--activation", default="gelu_tanh", choices=["gelu_tanh", "silu", "none"])
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--paper", action="store_true", help="paper setup: K=N=8192, no activation, no bias")
    ap.add_argument("--modes", default="single,single_cutlass,microbatch,tp2,ctapipe,ctapipe_cutlass")
    ap.add_argument("--chunks", default="512,1024,2048,4096,8192")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--producer-share", default=None,
                    help="comma list of CuTe CTA-pipe producer shares f (default: the module default)")
    ap.add_argument("--x-replicated", action="store_true",
                    help="also time the CuTe CTA-pipe with x pre-placed on GPU 1 (x_consumer)")
    ap.add_argument("--graph", action="store_true", help="also time the CuTe CTA-pipe with use_graph=True")
    ap.add_argument("--b2b", action="store_true",
                    help="also time tp2 / tp2_cute / tp2_overlap back-to-back (iters calls without a host sync)")
    ap.add_argument("--tp2-output", default="sharded",
                    help="comma list of tp2_cute / tp2_overlap outputs: sharded, replicated")
    args = ap.parse_args(argv)
    tp2_outputs = args.tp2_output.split(",")
    shares = [None] if args.producer_share is None else [float(f) for f in args.producer_share.split(",")]

    if args.paper:
        args.k, args.n, args.activation, args.no_bias = 8192, 8192, "none", True
    bias = not args.no_bias
    modes = args.modes.split(",")
    d0, d1 = (torch.device("cuda", int(i)) for i in args.devices.split(","))
    if torch.cuda.device_count() < 2:
        sys.exit("need two CUDA devices")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")

    print(f"K={args.k} N={args.n} act={args.activation} bias={bias} dtype=bf16  GPUs: "
          f"{torch.cuda.get_device_name(d0)} x2")
    rows = []
    for M in (int(t) for t in args.tokens.split(",")):
        print(f"\n== M={M} tokens ==")
        w1, b1, w2, b2 = make_weights(args.k, args.n, bias, d0)
        x = torch.randn(M, args.k, device=d0, dtype=DTYPE, generator=torch.Generator(device=d0).manual_seed(1))
        ref = mlp_reference(x, w1, b1, w2, b2, args.activation)
        torch.cuda.synchronize(d0)
        row = {"M": M}
        if "single" in modes:
            row["single"] = bench_single(x, w1, b1, w2, b2, args.activation, args.iters, args.warmup)
            print(f"  single GPU          : {row['single']:9.3f} ms")
        if "microbatch" in modes:
            chunks = [int(c) for c in args.chunks.split(",")]
            res = bench_microbatch(x, w1, b1, w2, b2, args.activation, d1, chunks, args.iters, args.warmup, ref)
            best = min(res, key=res.get)
            row["microbatch"], row["mb_chunk"] = res[best], best
            print("  micro-batch sweep   : " + "  ".join(f"c{c}={t:.3f}" for c, t in res.items()))
            print(f"  micro-batch best    : {res[best]:9.3f} ms  (chunk {best})")
        if "tp2" in modes:
            del ref  # free GPU 1 / GPU 0 memory before the child processes start
            torch.cuda.empty_cache()
            t = bench_tp2(M, args.k, args.n, args.activation, bias, args.iters, args.warmup, b2b=args.b2b)
            ref = mlp_reference(x, w1, b1, w2, b2, args.activation)
            if args.b2b:
                row["tp2"], row["tp2_b2b"] = t
                print(f"  TP2 (NCCL)          : {row['tp2']:9.3f} ms, back-to-back {row['tp2_b2b']:.3f} ms")
            else:
                row["tp2"] = t
                print(f"  TP2 (NCCL)          : {row['tp2']:9.3f} ms")
        if "ctapipe" in modes:
            row["ctapipe"] = bench_ctapipe(x, w1, b1, w2, b2, args.activation, d1, args.iters, args.warmup, ref)
            print(f"  CTA-pipe (Triton)   : {row['ctapipe']:9.3f} ms")
        if "single_cutlass" in modes:
            row["single_cutlass"] = bench_single_cutlass(
                x, w1, b1, w2, b2, args.activation, args.iters, args.warmup, ref
            )
            print(f"  single GPU (CuTe)   : {row['single_cutlass']:9.3f} ms")
        if "ctapipe_cutlass" in modes:
            res, tiles = bench_ctapipe_cutlass(
                x, w1, b1, w2, b2, args.activation, d1, args.iters, args.warmup, ref, shares, args.x_replicated,
                args.graph,
            )
            for xr in (False, True) if args.x_replicated else (False,):
                for g in (False, True) if args.graph else (False,):
                    row["ctapipe_cutlass" + ("_xr" if xr else "") + ("_g" if g else "")] = res[(shares[0], xr, g)]
            row["share_sweep"], row["share_tiles"] = res, tiles
        if "tp2_cute" in modes:
            res = bench_tp2_cute(x, w1, b1, w2, b2, args.activation, d1, args.iters, args.warmup, ref, tp2_outputs,
                                 args.b2b)
            for out, t in res.items():
                row["tp2_cute" + out.replace("sharded", "").replace("replicated", "_rep")] = t
        if "tp2_overlap" in modes:
            res = bench_tp2_overlap(x, w1, b1, w2, b2, args.activation, d1, args.iters, args.warmup, ref,
                                    tp2_outputs, args.b2b)
            for out, t in res.items():
                row["tp2_overlap" + out.replace("sharded", "").replace("replicated", "_rep")] = t
        rows.append(row)
        del x, ref, w1, w2
        torch.cuda.empty_cache()

    def cell(r, k):
        return f"{r[k]:9.3f}" if k in r else f"{'-':>9}"

    def reduction(r, variant, base):
        return f"{100 * (1 - r[variant] / r[base]):6.1f}%" if {variant, base} <= r.keys() else "      -"

    print("\n== latency (ms, lower is better; CTA-CuTe-xr = x already on GPU 1; -g = CUDA graph) ==")
    cols = ("single", "1GPU-CuTe", "MB best", "chunk", "TP2", "CTA-Trit", "CTA-CuTe", "CTA-CuTe-xr", "CTA-CuTe-g",
            "CTA-CuTe-xr-g")
    print(f"{'M':>7} " + " ".join(f"{c:>9}" if c != "chunk" else f"{c:>6}" for c in cols))
    for r in rows:
        chunk = f"{r['mb_chunk']:6d}" if "mb_chunk" in r else f"{'-':>6}"
        print(f"{r['M']:>7} {cell(r, 'single')} {cell(r, 'single_cutlass')} {cell(r, 'microbatch')} {chunk} "
              f"{cell(r, 'tp2')} {cell(r, 'ctapipe')} {cell(r, 'ctapipe_cutlass')} {cell(r, 'ctapipe_cutlass_xr'):>11} "
              f"{cell(r, 'ctapipe_cutlass_g'):>10} {cell(r, 'ctapipe_cutlass_xr_g'):>13}")
    if {"tp2_cute", "tp2_overlap"} & set(modes):
        print("\n== TP2 variants (ms; -rep = full y on both GPUs, like tp2; % vs tp2, negative = faster) ==")
        for sfx, title in (("", "per step (sync before each call)"), ("_b2b", "back-to-back")):
            keys = tuple(k + sfx for k in ("tp2", "tp2_cute", "tp2_cute_rep", "tp2_overlap", "tp2_overlap_rep"))
            if not any(k in r for r in rows for k in keys):
                continue
            print(f"-- {title}")
            print(f"{'M':>7} " + " ".join(f"{k:>16}" for k in keys))
            for r in rows:
                cells = []
                for k in keys:
                    if k not in r:
                        cells.append(f"{'-':>16}")
                    elif k != keys[0] and keys[0] in r:
                        cells.append(f"{r[k]:8.3f} {100 * (r[k] / r[keys[0]] - 1):+6.1f}%")
                    else:
                        cells.append(f"{r[k]:16.3f}")
                print(f"{r['M']:>7} " + " ".join(cells))
    if len(shares) > 1:
        print("\n== CuTe CTA-pipe producer-share sweep (ms; producer tiles / N1 tiles; xr = x on GPU 1) ==")
        for r in rows:
            if "share_sweep" in r:
                print(f"{r['M']:>7} " + "  ".join(
                    f"{r['share_tiles'][f]}{' xr' if xr else ''}{' g' if g else ''}={t:.3f}"
                    for (f, xr, g), t in r["share_sweep"].items()))
    print("\n== latency reduction of CTA-pipelining (R-MB vs best micro-batch, R-TP vs TP2; paper Fig. 5) ==")
    print(f"{'M':>7} {'Triton R-MB':>12} {'Triton R-TP':>12} {'CuTe R-MB':>12} {'CuTe R-TP':>12} {'CuTe vs 1GPU':>13}")
    for r in rows:
        print(f"{r['M']:>7} {reduction(r, 'ctapipe', 'microbatch'):>12} {reduction(r, 'ctapipe', 'tp2'):>12} "
              f"{reduction(r, 'ctapipe_cutlass', 'microbatch'):>12} {reduction(r, 'ctapipe_cutlass', 'tp2'):>12} "
              f"{reduction(r, 'ctapipe_cutlass', 'single'):>13}")


if __name__ == "__main__":
    main()
