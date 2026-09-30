"""TP2 MLP with the fc2 reduction hidden inside the fc2 kernel (batch 5).

Same sharding as Megatron-style TP2 (``bench_mlp_2gpu._tp2_worker``): GPU g holds
``W1[g*N1/2:(g+1)*N1/2]`` (+ that ``b1`` shard), ``W2[:, g*N1/2:(g+1)*N1/2]`` and the full
``b2``; X is replicated on both GPUs. Per GPU, on its own stream: the plain-role fc1 shard
(``H_g = act(X W1_g^T + b1_g)``, local), then the fc2 shard in ``ROLE_REDUCE``
(``cute_gemm.py``). Row blocks of 128 rows are owned by one GPU each (GPU 0 the first
``ceil(num_m / 2)``, GPU 1 the rest). fc2 on GPU g first computes the *peer-owned* row blocks
and TMA-stores the bf16 partials straight into the peer's output (then bumps the peer's row
counter: the producer release of ``CuteCTAPipelinedMLP``); then its own row blocks, whose epilogue
waits for the peer's 12 partial tiles of that row block and TMA reduce-adds ``acc + b2`` onto
them (the add happens in L2). So the only communication is TP2's reduce-scatter bytes,
overlapped with the second half of fc2. ``output="replicated"``: range A also stores the partial
into the local output and range B also reduce-adds into the peer's (an in-kernel all-reduce).

Single process, two devices (peer access), like :class:`CuteCTAPipelinedMLP`.
"""

from __future__ import annotations

import torch

from mstar.utils.cta_pipelining.cute_gemm import ROLE_PLAIN, ROLE_REDUCE, CuteGemmOp, _max_clusters
from mstar.utils.cta_pipelining.cute_mlp import DEFAULT_TILE_MN, FC2_CLUSTER_MN, PLAIN_RASTER_G
from mstar.utils.cta_pipelining.mlp import ensure_peer_access


class CuteTP2OverlapMLP:
    """``y = act(x @ W1^T + b1) @ W2^T + b2`` over two GPUs with TP2 sharding and the fc2
    reduce-scatter (``output="sharded"``) or all-reduce (``"replicated"``) fused into fc2.

    ``forward(x0, x1)``: X on ``devices[0]`` and on ``devices[1]`` (the same values). Returns
    ``(y0, y1)``: sharded, GPU 0's rows ``[0, r0)`` on ``devices[0]`` and rows ``[r0, M)`` on
    ``devices[1]`` with ``r0 = min(M, ceil(ceil(M / 128) / 2) * 128)`` (see :meth:`split`);
    replicated, the full ``[M, N2]`` output on each device. Each output is ready on its device's
    current stream. Requires ``N1 % (2 * tile_n) == 0``, ``K % 64 == 0``, ``N2 % tile_n == 0``.
    """

    def __init__(
        self,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        *,
        devices: tuple = (0, 1),
        activation: str = "gelu_tanh",
        max_tokens: int = 0,
        output: str = "sharded",
        tile_shape_mn: tuple[int, int] = DEFAULT_TILE_MN,
    ):
        if output not in ("sharded", "replicated"):
            raise ValueError(f"output must be 'sharded' or 'replicated', got {output!r}")
        self.devs = tuple(torch.device(d) if not isinstance(d, int) else torch.device("cuda", d) for d in devices)
        ensure_peer_access(*self.devs)
        if w2.shape[1] != w1.shape[0]:
            raise ValueError(f"W2 in-features {w2.shape[1]} != W1 out-features {w1.shape[0]}")
        self.N1, self.K = w1.shape
        self.N2 = w2.shape[0]
        self.tile_m, self.tile_n = tile_shape_mn
        if self.N1 % (2 * self.tile_n):
            raise ValueError(f"N1 ({self.N1}) must split into two multiples of tile_n ({self.tile_n})")
        self.output = output
        rep = output == "replicated"
        half = self.N1 // 2
        w1, w2 = w1.detach(), w2.detach()
        self.w1 = [w1[g * half : (g + 1) * half].to(d).contiguous() for g, d in enumerate(self.devs)]
        self.b1 = [None if b1 is None else b1.detach()[g * half : (g + 1) * half].to(d).contiguous()
                   for g, d in enumerate(self.devs)]
        self.w2 = [w2[:, g * half : (g + 1) * half].to(d).contiguous() for g, d in enumerate(self.devs)]
        # b2 is added once, by the GPU that finalizes a row block (range B), never into a partial.
        self.b2 = [None if b2 is None else b2.detach().to(d).contiguous() for d in self.devs]
        fc2_cluster = FC2_CLUSTER_MN if self.N2 % (self.tile_n * FC2_CLUSTER_MN[1]) == 0 else (1, 1)
        # One op (one compiled kernel) per GEMM, launched on both devices. The DSL loads a compiled
        # library into every device on its first call, and that load blocks behind a running kernel:
        # with one fc2 op per device, GPU 1's first fc2 launch waited for GPU 0's fc2, which spins
        # until GPU 1's partials arrive (deadlock). Shared ops load before the first fc2 launch.
        self.fc1 = CuteGemmOp(ROLE_PLAIN, activation, b1 is not None, tile_shape_mn=tile_shape_mn,
                              raster_group=PLAIN_RASTER_G)
        self.fc2 = CuteGemmOp(ROLE_REDUCE, "none", b2 is not None, tile_shape_mn=tile_shape_mn,
                              cluster_shape_mn=fc2_cluster, replicated=rep)
        for d in self.devs:
            _max_clusters(d, self.fc2.cluster_size)  # the occupancy query, also ahead of any spinning kernel
        # Partial tiles per row block == what an own row block waits for (one signal per tile).
        self.n_ready = self.N2 // self.tile_n
        self.streams = [torch.cuda.Stream(device=d) for d in self.devs]
        self._h: list[torch.Tensor] | None = None  # [cap, N1/2] per GPU
        self._counters: list[torch.Tensor] | None = None
        # Epoch counters as in CuteCTAPipelinedMLP: never zeroed, forward e waits for e * n_ready.
        self._epoch = 0
        self._ev_done: list[torch.cuda.Event | None] = [None, None]  # previous forward's fc2 per GPU
        if max_tokens:
            self._reserve(max_tokens)

    def split(self, M: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Row-block ownership ``((m0, count), (m0, count))`` of GPU 0 / GPU 1 for ``M`` rows."""
        num_m = -(-M // self.tile_m)
        m_split = (num_m + 1) // 2  # odd num_m: one extra block on GPU 0
        return (0, m_split), (m_split, num_m - m_split)

    def _reserve(self, M: int) -> None:
        if self._h is not None and self._h[0].shape[0] >= M:
            return
        for d in self.devs:
            torch.cuda.synchronize(d)  # only on (re)allocation
        num_m = -(-M // self.tile_m)
        self._h, self._counters = [], []
        for d, w1 in zip(self.devs, self.w1, strict=True):
            with torch.cuda.device(d):
                self._h.append(torch.empty((M, w1.shape[0]), dtype=w1.dtype, device=d))
                self._counters.append(torch.zeros(num_m, dtype=torch.int32, device=d))
        for d in self.devs:
            torch.cuda.synchronize(d)  # the zero-fill must land before the peer's first signal
        self._epoch = 0
        for op in (self.fc1, self.fc2):
            op.clear_cache()  # cached conversions of the old buffers

    def _bump(self, g: int, M: int) -> None:
        """Advance every counter of GPU g that no partial tile targets this forward (the rows GPU g
        does not own, and rows past M) by ``n_ready``, so all counters stay at ``epoch * n_ready``.
        Only those slices are written: the peer's ``red.add`` on the own rows may already be in
        flight, and a read-modify-write over them (even adding 0) loses its updates."""
        m0, cnt = self.split(M)[g]
        c = self._counters[g]
        for lo, hi in ((0, m0), (m0 + cnt, c.numel())):
            if hi > lo:
                c[lo:hi].add_(self.n_ready)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xs = (x0, x1)
        for g, (x, d) in enumerate(zip(xs, self.devs, strict=True)):
            if x.device != d or x.dim() != 2 or x.shape[1] != self.K or x.shape != x0.shape:
                raise ValueError(f"x{g} must be [M, {self.K}] on {d}, got {tuple(x.shape)} on {x.device}")
        M = x0.shape[0]
        self._reserve(M)
        own = self.split(M)
        rows = (min(M, own[0][1] * self.tile_m), M - min(M, own[0][1] * self.tile_m))
        rep = self.output == "replicated"
        if (self._epoch + 1) * self.n_ready >= 2**30:  # keep int32 counters far from overflow
            for d in self.devs:
                torch.cuda.synchronize(d)
            for d, c in zip(self.devs, self._counters, strict=True):
                with torch.cuda.device(d):
                    c.zero_()
            for d in self.devs:
                torch.cuda.synchronize(d)
            self._epoch = 0
        self._epoch += 1

        xs = tuple(x.contiguous() for x in xs)
        cur = [torch.cuda.current_stream(d) for d in self.devs]
        ev_in = []
        for c in cur:  # inputs ready; also orders the outputs allocated below after the caller's work
            e = torch.cuda.Event()
            e.record(c)
            ev_in.append(e)
        for g, (d, s) in enumerate(zip(self.devs, self.streams, strict=True)):
            with torch.cuda.device(d), torch.cuda.stream(s):
                s.wait_event(ev_in[g])
                xs[g].record_stream(s)
                # H_g is stream-ordered after the previous forward's fc2 on this stream.
                self.fc1.launch(xs[g], self.w1[g], self.b1[g], self._h[g][:M], None, device=d, stream=s,
                                   static=("w", "bias", "out"))
        ys = []
        for g, d in enumerate(self.devs):
            with torch.cuda.device(d):
                y = torch.empty((M if rep else max(rows[g], 1), self.N2), dtype=self.w2[g].dtype, device=d)
            y.record_stream(self.streams[g])
            y.record_stream(self.streams[1 - g])  # the peer's fc2 writes its partials into y
            ys.append(y)
        for g, (d, s) in enumerate(zip(self.devs, self.streams, strict=True)):
            with torch.cuda.device(d), torch.cuda.stream(s):
                # Before either fc2 launch: nothing but event ops sits between the two fc2 launches (GPU 0's
                # fc2 spins until GPU 1's fc2 runs, so a host call that waits on GPU 0 there would deadlock).
                self._bump(g, M)
        ev_done = []
        for g, (d, s) in enumerate(zip(self.devs, self.streams, strict=True)):
            p = 1 - g
            with torch.cuda.device(d), torch.cuda.stream(s):
                if self._ev_done[p] is not None:
                    # Range A signals the peer's counters: the peer's previous fc2 (and so its _bump,
                    # which may cover these rows when M changed) must have finished.
                    s.wait_event(self._ev_done[p])
                s.wait_event(ev_in[p])  # range A writes the peer's output (allocated on its stream)
                a_m0, a_rows = own[p]
                b_m0 = own[g][0]
                self.fc2.launch(
                    self._h[g][:M], self.w2[g], self.b2[g], ys[p], self._counters[g],
                    n_ready=self._epoch * self.n_ready, device=d, stream=s,
                    reduce=(ys[g], self._counters[p], a_m0, a_rows, b_m0),
                    static=("a", "w", "bias", "counters", "counters2"),
                )
                e = torch.cuda.Event()
                e.record(s)
                ev_done.append(e)
        self._ev_done = ev_done
        for g in range(2):
            cur[g].wait_event(ev_done[g])
            if rep:
                cur[g].wait_event(ev_done[1 - g])  # the peer's range B reduce-adds into ys[g]
        if rep:
            return ys[0], ys[1]
        return ys[0][: rows[0]], ys[1][: rows[1]]

    __call__ = forward
