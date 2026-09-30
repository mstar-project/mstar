"""CTA-pipelined two-layer MLP on the CUTLASS (CuTe DSL) Hopper kernels.

Same interface and stream/event choreography as :class:`CTAPipelinedMLP`
(the Triton version in ``mlp.py``), except that the row counters count up
across forwards (epochs) instead of being zeroed each time, GEMM 1 is
column-split (the consumer GPU computes the last ``N1 - N1p`` columns of ``H``
itself, before its GEMM 2), and GEMM 1 and GEMM 2 are the
warp-specialised persistent WGMMA kernels from ``cute_gemm.py`` in the
producer / consumer roles. ``CutePlainMLP`` runs the same kernel in its
plain role on one GPU (fc1+act, then fc2) to separate kernel quality from
protocol overhead in the benchmark.
"""

from __future__ import annotations

import torch

from mstar.utils.cta_pipelining.cute_gemm import ROLE_CONSUMER, ROLE_PLAIN, ROLE_PRODUCER, CuteGemmOp
from mstar.utils.cta_pipelining.mlp import ensure_peer_access

DEFAULT_TILE_MN = (128, 256)
# Raster group of the plain / local fc1 (fix 6a/9, ``CTAPipeGemm.raster_group``): a wave reads W1 once
# per group of 32 row blocks. The producer keeps G = 1 (row-block-major): grouping delays row-block
# completion, and G = 8 measured equal to 12 % slower on the Wan2.2 FFN with the split and up to
# 30 % slower on the paper's 8192^2 shape (README).
PLAIN_RASTER_G = 32
# Default producer share f per GEMM shape (K, N1, N2), from the batch-3 sweeps on 2x H100 (README);
# unlisted shapes run unsplit (f = 1.0).
DEFAULT_PRODUCER_SHARE = {(3072, 14336, 3072): 48 / 56, (8192, 8192, 8192): 1.0}
# CUDA-graph mode (``use_graph``) starts GPU 1 ~0.2 ms earlier (no serial host launches), so GPU 1 can take
# a larger fc1 share at small M: ``((max_tokens bound, f), ...)``, first bound >= max_tokens wins, None =
# any; from the batch-4 graph-mode sweeps (README). Chosen once by ``max_tokens`` (the W1 split is fixed at
# construction), so a later larger M keeps it; without ``max_tokens`` the eager default applies.
DEFAULT_PRODUCER_SHARE_GRAPH = {(3072, 14336, 3072): ((4096, 40 / 56), (8192, 44 / 56), (None, 48 / 56))}
# Cluster of the consumer fc2 (batch 4 item 1, ``CTAPipeGemm.cluster_shape_mn``): (1, 2) pairs CTAs
# on adjacent n-tiles of one H row block and TMA-multicasts the H tile. Needs an even N2 / tile_n,
# else (1, 1).
FC2_CLUSTER_MN = (1, 2)

class CuteCTAPipelinedMLP:
    """``y = act(x @ W1^T + b1) @ W2^T + b2``, GEMM 1 on ``producer_device``,
    GEMM 2 on ``consumer_device``, CTA-pipelined through peer memory.

    Requires ``K % 64 == 0`` and ``N1 % tile_n == N2 % tile_n == 0``; ``M`` is
    free. See :class:`mstar.utils.cta_pipelining.CTAPipelinedMLP` for the
    argument semantics (identical). ``producer_share`` f: the producer computes
    columns ``[0, N1p)`` of ``H`` with ``N1p = round(f * N1 / tile_n) * tile_n``
    (clamped to ``[tile_n, N1]``); the consumer GPU computes ``[N1p, N1)`` locally
    (plain-role kernel on ``c_stream``) and then waits only for the producer's
    share. ``f = 1.0`` is the unsplit pipeline. ``None`` = ``DEFAULT_PRODUCER_SHARE`` for
    this shape (1.0 if unlisted); with ``use_graph`` and ``max_tokens``, ``DEFAULT_PRODUCER_SHARE_GRAPH``
    picks it by ``max_tokens`` instead (the split is fixed at construction).

    ``use_graph``: capture the forward in one CUDA graph per ``(M, dtype, x_consumer is None)``
    (after 3 eager warm-up forwards) and replay it. ``forward`` then copies ``x`` (and
    ``x_consumer``) into the graph's static input buffers and returns the graph's static output
    tensor, which the next call with the same ``M`` overwrites (torch's CUDA-graph contract).
    """

    def __init__(
        self,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        *,
        producer_device: torch.device | str,
        consumer_device: torch.device | str,
        activation: str = "gelu_tanh",
        output_device: torch.device | str | None = None,
        max_tokens: int = 0,
        tile_shape_mn: tuple[int, int] = DEFAULT_TILE_MN,
        fence: bool = True,
        producer_share: float | None = None,
        use_graph: bool = False,
    ):
        self.pdev = torch.device(producer_device)
        self.cdev = torch.device(consumer_device)
        self.odev = torch.device(output_device) if output_device is not None else self.pdev
        ensure_peer_access(self.pdev, self.cdev)

        if w2.shape[1] != w1.shape[0]:
            raise ValueError(f"W2 in-features {w2.shape[1]} != W1 out-features {w1.shape[0]}")
        self.N1, self.K = w1.shape
        self.N2 = w2.shape[0]
        self.tile_m, self.tile_n = tile_shape_mn
        shape = (self.K, self.N1, self.N2)
        f = DEFAULT_PRODUCER_SHARE.get(shape, 1.0) if producer_share is None else producer_share
        if producer_share is None and use_graph and max_tokens and shape in DEFAULT_PRODUCER_SHARE_GRAPH:
            f = next(g for bound, g in DEFAULT_PRODUCER_SHARE_GRAPH[shape] if bound is None or max_tokens <= bound)
        self.N1p = min(self.N1, max(self.tile_n, round(f * self.N1 / self.tile_n) * self.tile_n))
        # Weights are placed once (like TP2 shards): W1[:N1p] on the producer, W1[N1p:] on the consumer.
        w1, b1 = w1.detach(), None if b1 is None else b1.detach()
        self.w1 = w1[: self.N1p].to(self.pdev).contiguous()
        self.b1 = None if b1 is None else b1[: self.N1p].to(self.pdev).contiguous()
        self.w1_local = w1[self.N1p :].to(self.cdev).contiguous() if self.N1p < self.N1 else None
        self.b1_local = None if (b1 is None or self.w1_local is None) else b1[self.N1p :].to(self.cdev).contiguous()
        self.w2 = w2.detach().to(self.cdev).contiguous()
        self.b2 = None if b2 is None else b2.detach().to(self.cdev).contiguous()

        self.producer = CuteGemmOp(
            ROLE_PRODUCER, activation, b1 is not None, tile_shape_mn=tile_shape_mn, fence=fence
        )
        fc2_cluster = FC2_CLUSTER_MN if self.N2 % (self.tile_n * FC2_CLUSTER_MN[1]) == 0 else (1, 1)
        self.consumer = CuteGemmOp(
            ROLE_CONSUMER, "none", b2 is not None, tile_shape_mn=tile_shape_mn, fence=fence,
            cluster_shape_mn=fc2_cluster,
        )
        self.local = None if self.w1_local is None else CuteGemmOp(
            ROLE_PLAIN, activation, b1 is not None, tile_shape_mn=tile_shape_mn, raster_group=PLAIN_RASTER_G
        )
        # Producer tiles per row block == what a consumer row block waits for.
        self.n_ready = self.N1p // self.tile_n

        self.p_stream = torch.cuda.Stream(device=self.pdev)
        self.c_stream = torch.cuda.Stream(device=self.cdev)
        self._h: torch.Tensor | None = None
        self._counters: torch.Tensor | None = None
        # Epoch counters: row counters are never zeroed; forward number e waits
        # for counters[m] >= e * n_ready. Reset only when the buffer is reallocated.
        self._epoch = 0
        self._ev_done: torch.cuda.Event | None = None  # previous forward's consumer
        # CUDA-graph mode: key -> (graph, static x, static x on the consumer or None, static y).
        self.use_graph = use_graph
        self._graphs: dict[tuple, tuple] = {}
        self._ev_graph_done: torch.cuda.Event | None = None  # previous replay, on the caller's stream
        self._x_stream: torch.cuda.Stream | None = None  # producer-side stream of the captured X copy
        if max_tokens:
            self._reserve(max_tokens)

    def _reserve(self, M: int) -> None:
        if self._h is not None and self._h.shape[0] >= M:
            return
        with torch.cuda.device(self.cdev):
            torch.cuda.synchronize(self.cdev)  # only on (re)allocation
            self._h = torch.empty((M, self.N1), dtype=self.w1.dtype, device=self.cdev)
            self._counters = torch.zeros(-(-M // self.tile_m), dtype=torch.int32, device=self.cdev)
            self._epoch = 0
            self._graphs.clear()  # captured graphs point at the old ``_h`` / counters
            for op in (self.producer, self.local, self.consumer):
                if op is not None:
                    op.clear_cache()  # cached conversions of the old ``_h`` / counters

    def forward(self, x: torch.Tensor, x_consumer: torch.Tensor | None = None) -> torch.Tensor:
        """``x_consumer``: optional copy of ``x`` already on the consumer device (both
        workers hold the block input). Without it and with ``N1p < N1``, ``x`` is copied
        to the consumer inside the forward."""
        if x.device != self.pdev or x.dim() != 2 or x.shape[1] != self.K:
            raise ValueError(f"expected x [M, {self.K}] on {self.pdev}, got {tuple(x.shape)} on {x.device}")
        if x_consumer is not None and (x_consumer.device != self.cdev or x_consumer.shape != x.shape):
            raise ValueError(f"x_consumer must be x's shape on {self.cdev}, got {tuple(x_consumer.shape)}")
        if self.use_graph:
            return self._forward_graph(x, x_consumer)
        return self._forward_eager(x, x_consumer)

    def _forward_eager(self, x: torch.Tensor, x_consumer: torch.Tensor | None) -> torch.Tensor:
        M = x.shape[0]
        self._reserve(M)
        num_m = -(-M // self.tile_m)
        N1p = self.N1p
        has_local = self.local is not None
        if (self._epoch + 1) * self.n_ready >= 2**30:  # keep int32 counters far from overflow
            torch.cuda.synchronize(self.pdev)
            torch.cuda.synchronize(self.cdev)
            with torch.cuda.device(self.cdev):
                self._counters.zero_()
            torch.cuda.synchronize(self.cdev)
            self._epoch = 0
        self._epoch += 1

        caller = torch.cuda.current_stream(self.pdev)
        out_stream = torch.cuda.current_stream(self.odev)
        if x_consumer is not None and has_local:
            x_consumer = x_consumer.contiguous()
            ev_xc = torch.cuda.Event()
            ev_xc.record(torch.cuda.current_stream(self.cdev))  # x_consumer is ready there
        x = x.contiguous()
        x.record_stream(self.p_stream)
        y = torch.empty((M, self.N2), dtype=self.w2.dtype, device=self.odev)
        y.record_stream(self.c_stream)

        ev_inputs = torch.cuda.Event()
        ev_inputs.record(caller)
        ev_out_free = torch.cuda.Event()
        ev_out_free.record(out_stream)

        x_copied = has_local and x_consumer is None
        if x_copied:
            # X for the local share, issued first so the copy runs during the host launches.
            # torch runs a peer copy on the source device's current stream (the caller's,
            # after ``ev_inputs``, so the producer does not wait for it) and makes ``c_stream``
            # wait for it.
            with torch.cuda.device(self.cdev), torch.cuda.stream(self.c_stream):
                x_consumer = x.to(self.cdev, non_blocking=True)

        with torch.cuda.device(self.pdev), torch.cuda.stream(self.p_stream):
            self.p_stream.wait_event(ev_inputs)
            if self._ev_done is not None:
                # ``_h`` is reused: do not overwrite it before the previous
                # consumer has read it (already true if the caller waited on it).
                self.p_stream.wait_event(self._ev_done)
            self.producer.launch(
                x, self.w1, self.b1, self._h[:M, :N1p], self._counters, device=self.pdev, stream=self.p_stream,
                static=("w", "bias", "out", "counters"),
            )
            ev_prod = torch.cuda.Event()
            ev_prod.record(self.p_stream)

        with torch.cuda.device(self.cdev), torch.cuda.stream(self.c_stream):
            self.c_stream.wait_event(ev_out_free)
            if num_m < self._counters.numel():
                # Row blocks past M get no producer tiles this time; advance them
                # one epoch too so every counter stays at epoch * n_ready. Off the
                # critical path (the producer does not wait for it).
                self._counters[num_m:].add_(self.n_ready)
            if has_local:
                # GPU 1's own share of GEMM 1 (columns [N1p, N1) of H), launched after the
                # producer so GPU 0 starts first (host launches are serial, ~60-70 us each).
                # Stream-ordered after the previous consumer's reads of ``_h`` and before this
                # forward's consumer.
                if not x_copied:
                    self.c_stream.wait_event(ev_xc)
                    x_consumer.record_stream(self.c_stream)
                self.local.launch(
                    x_consumer, self.w1_local, self.b1_local, self._h[:M, N1p:], None,
                    device=self.cdev, stream=self.c_stream, static=("w", "bias", "out"),
                )
            self.consumer.launch(
                self._h[:M], self.w2, self.b2, y, self._counters,
                n_ready=self._epoch * self.n_ready, device=self.cdev, stream=self.c_stream,
                static=("a", "w", "bias", "counters"),
            )
            ev_done = torch.cuda.Event()
            ev_done.record(self.c_stream)
        self._ev_done = ev_done

        caller.wait_event(ev_prod)
        caller.wait_event(ev_done)
        if self.odev != self.pdev:
            out_stream.wait_event(ev_done)
        return y

    # ------------------------------------------------------------ CUDA-graph mode
    def _forward_graph(self, x: torch.Tensor, x_consumer: torch.Tensor | None) -> torch.Tensor:
        if self.local is None:
            x_consumer = None  # unsplit: GPU 1 does not read X
        key = (x.shape[0], x.dtype, x_consumer is None)
        if key not in self._graphs:
            self._capture(x, x_consumer, key)
        graph, sx, sxc, sy = self._graphs[key]
        caller = torch.cuda.current_stream(self.pdev)
        out_stream = torch.cuda.current_stream(self.odev)
        for ev in (self._ev_graph_done, self._ev_done):  # previous replay / eager forward still using H, sx, sy
            if ev is not None:
                caller.wait_event(ev)
        with torch.cuda.device(self.pdev):
            sx.copy_(x)
        if x_consumer is not None:
            # The previous replay's local fc1 has read ``sxc`` once it completed.
            cons = torch.cuda.current_stream(self.cdev)
            with torch.cuda.device(self.cdev):
                if self._ev_graph_done is not None:
                    cons.wait_event(self._ev_graph_done)
                sxc.copy_(x_consumer)
                ev_xc = torch.cuda.Event()
                ev_xc.record(cons)
            caller.wait_event(ev_xc)
        if self.odev != self.pdev:
            ev_out_free = torch.cuda.Event()
            ev_out_free.record(out_stream)  # earlier reads of the static ``y`` on the output device
            caller.wait_event(ev_out_free)
        with torch.cuda.device(self.pdev):
            graph.replay()  # on the caller's stream; the GPU-1 nodes keep their device
            ev_done = torch.cuda.Event()
            ev_done.record(caller)
        self._ev_graph_done = ev_done
        # Replays do not maintain the eager epoch invariant: make the next eager forward (e.g. the
        # warm-ups of the next capture) take the int32-guard path (sync, zero, epoch 0).
        self._epoch = 1 << 30
        if self.odev != self.pdev:
            out_stream.wait_event(ev_done)
        return sy

    def _capture(self, x: torch.Tensor, x_consumer: torch.Tensor | None, key: tuple) -> None:
        M = x.shape[0]
        for _ in range(3):  # compile, module load; may reallocate ``_h``, which drops all graphs
            self._forward_eager(x, x_consumer)
        torch.cuda.synchronize(self.pdev)
        torch.cuda.synchronize(self.cdev)
        with torch.cuda.device(self.cdev):
            self._counters.zero_()  # replays expect zero counters and leave them zero
        torch.cuda.synchronize(self.cdev)
        sx = torch.empty_like(x, memory_format=torch.contiguous_format)
        sxc = None if self.local is None else torch.empty(x.shape, dtype=x.dtype, device=self.cdev)
        sy = torch.empty((M, self.N2), dtype=self.w2.dtype, device=self.odev)
        x_copied = self.local is not None and x_consumer is None
        if x_copied and self._x_stream is None:
            self._x_stream = torch.cuda.Stream(device=self.pdev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(self.pdev), torch.cuda.graph(graph, stream=self.p_stream):
            self._graph_body(sx, sxc, sy, x_copied)
        torch.cuda.synchronize(self.pdev)
        torch.cuda.synchronize(self.cdev)
        self._graphs[key] = (graph, sx, sxc, sy)

    def _graph_body(self, sx, sxc, sy, x_copied: bool) -> None:
        """One forward as captured on ``p_stream``: ``c_stream`` (and the X-copy stream) join
        the capture through event waits and are joined back at the end."""
        M = sx.shape[0]
        num_m = -(-M // self.tile_m)
        N1p = self.N1p
        ev_fork = torch.cuda.Event()
        ev_fork.record(self.p_stream)
        with torch.cuda.device(self.cdev), torch.cuda.stream(self.c_stream):
            self.c_stream.wait_event(ev_fork)
            if x_copied:
                # X for the local share on its own producer-side stream, so the producer does not
                # wait for the copy (torch issues a peer copy on the source device's current stream).
                self._x_stream.wait_event(ev_fork)
                with torch.cuda.stream(self._x_stream):
                    sxc.copy_(sx)
        with torch.cuda.device(self.pdev), torch.cuda.stream(self.p_stream):
            self.producer.launch(
                sx, self.w1, self.b1, self._h[:M, :N1p], self._counters, device=self.pdev, stream=self.p_stream,
                static=("a", "w", "bias", "out", "counters"),
            )
        with torch.cuda.device(self.cdev), torch.cuda.stream(self.c_stream):
            if self.local is not None:
                self.local.launch(
                    sxc, self.w1_local, self.b1_local, self._h[:M, N1p:], None,
                    device=self.cdev, stream=self.c_stream, static=("a", "w", "bias", "out"),
                )
            self.consumer.launch(
                self._h[:M], self.w2, self.b2, sy, self._counters,
                n_ready=self.n_ready, device=self.cdev, stream=self.c_stream,
                static=("a", "w", "bias", "out", "counters"),
            )
            # The epoch target would be a constant of the graph: every replay starts from zero
            # counters, waits for N1p/tile_n tiles per row block and zeroes its rows again here (one
            # memset node after the consumer; zeroing before the producer measured 15-18 us slower).
            self._counters[:num_m].zero_()
            ev_done = torch.cuda.Event()
            ev_done.record(self.c_stream)
        self.p_stream.wait_event(ev_done)

    __call__ = forward


class CutePlainMLP:
    """Single-GPU ``act(x @ W1^T + b1) @ W2^T + b2`` with the same CuTe DSL
    kernel in its plain role: the kernel-quality baseline for the cutlass
    CTA-pipelined variant."""

    def __init__(self, w1, b1, w2, b2, *, activation="gelu_tanh", tile_shape_mn=DEFAULT_TILE_MN):
        self.dev = w1.device
        self.w1, self.b1 = w1.contiguous(), None if b1 is None else b1.contiguous()
        self.w2, self.b2 = w2.contiguous(), None if b2 is None else b2.contiguous()
        self.fc1 = CuteGemmOp(
            ROLE_PLAIN, activation, b1 is not None, tile_shape_mn=tile_shape_mn, raster_group=PLAIN_RASTER_G
        )
        self.fc2 = CuteGemmOp(ROLE_PLAIN, "none", b2 is not None, tile_shape_mn=tile_shape_mn)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream(self.dev)
        h = torch.empty((x.shape[0], self.w1.shape[0]), dtype=x.dtype, device=self.dev)
        y = torch.empty((x.shape[0], self.w2.shape[0]), dtype=x.dtype, device=self.dev)
        self.fc1.launch(x, self.w1, self.b1, h, None, device=self.dev, stream=stream, static=("w", "bias"))
        self.fc2.launch(h, self.w2, self.b2, y, None, device=self.dev, stream=stream, static=("w", "bias"))
        return y
