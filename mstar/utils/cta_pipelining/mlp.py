"""Single-process, two-device driver for the CTA-pipelined two-layer MLP.

``CTAPipelinedMLP`` holds ``W1``/``b1`` on the producer GPU and ``W2``/``b2`` on
the consumer GPU, keeps the cross-device activation buffer and row counters in
consumer memory, and runs the two persistent kernels concurrently on one
dedicated stream per device. All ordering is done with CUDA events; ``forward``
never synchronizes the host.

This is the standalone form used by the benchmark and the tests (one Python
process drives both GPUs, which is the simplest way to get two concurrently
resident kernels). Inside M* each GPU is a separate worker process; the same
kernels work there once ``h``, ``counters`` and ``y`` are allocated through
``torch.distributed._symmetric_memory`` so that each process can hand the
other's buffers to its kernel. See README.md, "Stage 2".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mstar.utils.cta_pipelining.kernels import (
    ACTIVATIONS,
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
    launch_consumer,
    launch_producer,
)


def ensure_peer_access(a: torch.device, b: torch.device) -> None:
    """Make sure kernels on ``a`` can dereference ``b``'s memory and vice versa.

    PyTorch calls ``cudaDeviceEnablePeerAccess`` lazily from inside a
    cross-device ``copy_`` (``at::cuda::get_p2p_access(src, dst)`` enables
    ``src -> dst``), so one tiny copy in each direction turns on both
    directions for this process. Raises if the hardware cannot do it, because
    a kernel touching an unmapped peer address is a sticky illegal-address
    fault, not a Python exception.
    """
    if a.type != "cuda" or b.type != "cuda" or a.index == b.index:
        raise ValueError(f"CTA-pipelining needs two distinct CUDA devices, got {a} and {b}")
    for src, dst in ((a, b), (b, a)):
        if not torch.cuda.can_device_access_peer(src.index, dst.index):
            raise RuntimeError(
                f"GPU {src.index} cannot access GPU {dst.index} peer memory (no NVLink/PCIe P2P). "
                "CTA-pipelining requires a unified peer-addressable memory domain."
            )
        torch.empty(1, device=dst).copy_(torch.empty(1, device=src))
    torch.cuda.synchronize(a)
    torch.cuda.synchronize(b)


@triton.jit
def _poke_kernel(ptr, value, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.full((BLOCK,), 0, tl.int32) + value, mask=offs < n)


def verify_peer_kernel_access(src: torch.device, dst: torch.device) -> None:
    """Launch a trivial kernel on ``src`` that writes into a tensor on ``dst``
    and check the write landed. Cheap, and turns the failure mode from an
    asynchronous CUDA fault into an exception at setup time."""
    ensure_peer_access(src, dst)  # Triton refuses a peer pointer until access is enabled
    target = torch.zeros(64, dtype=torch.int32, device=dst)
    with torch.cuda.device(src):
        _poke_kernel[(1,)](target, 7, target.numel(), BLOCK=64)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    if not bool((target == 7).all()):
        raise RuntimeError(f"peer write from GPU {src.index} into GPU {dst.index} memory did not land")


def mlp_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    activation: str = "gelu_tanh",
) -> torch.Tensor:
    """``linear_out(act(linear_in(x)))`` exactly as ``mstar.model.components.MLP``
    computes it (activation applied in the working dtype)."""
    h = F.linear(x, w1, b1)
    if activation == "gelu_tanh":
        h = F.gelu(h, approximate="tanh")
    elif activation == "silu":
        h = F.silu(h)
    elif activation != "none":
        raise ValueError(activation)
    return F.linear(h, w2, b2)


class CTAPipelinedMLP:
    """``y = act(x @ W1^T + b1) @ W2^T + b2`` with GEMM 1 on ``producer_device``
    and GEMM 2 on ``consumer_device``, pipelined at CTA granularity.

    Args:
        w1, b1: first linear (``[N1, K]``, ``[N1]``), any device; moved to the producer.
        w2, b2: second linear (``[N2, N1]``, ``[N2]``); moved to the consumer.
        producer_device, consumer_device: two distinct CUDA devices with P2P.
        activation: ``"gelu_tanh"`` (Wan2.2 FFN), ``"silu"`` or ``"none"``.
        output_device: where ``y`` is written. Defaults to the producer, i.e.
            the consumer's CTAs stream the result back over NVLink so the
            caller's residual add needs no separate copy.
        max_tokens: pre-size the activation buffer; grows on demand otherwise.
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
        block_m: int = DEFAULT_BLOCK_M,
        block_n1: int = DEFAULT_BLOCK_N,
        block_n2: int = DEFAULT_BLOCK_N,
        block_k: int = DEFAULT_BLOCK_K,
        num_warps: int = DEFAULT_NUM_WARPS,
        num_stages: int = DEFAULT_NUM_STAGES,
        fence: bool = True,
    ):
        self.pdev = torch.device(producer_device)
        self.cdev = torch.device(consumer_device)
        self.odev = torch.device(output_device) if output_device is not None else self.pdev
        ensure_peer_access(self.pdev, self.cdev)

        if w2.shape[1] != w1.shape[0]:
            raise ValueError(f"W2 in-features {w2.shape[1]} != W1 out-features {w1.shape[0]}")
        self.N1, self.K = w1.shape
        self.N2 = w2.shape[0]
        self.w1 = w1.detach().to(self.pdev).contiguous()
        self.b1 = None if b1 is None else b1.detach().to(self.pdev).contiguous()
        self.w2 = w2.detach().to(self.cdev).contiguous()
        self.b2 = None if b2 is None else b2.detach().to(self.cdev).contiguous()
        self.act = ACTIVATIONS[activation]

        self.block_m, self.block_n1, self.block_n2, self.block_k = block_m, block_n1, block_n2, block_k
        self.num_warps, self.num_stages, self.fence = num_warps, num_stages, fence
        # Producer tiles per row block: what a consumer row block waits for.
        self.n_ready = triton.cdiv(self.N1, self.block_n1)

        self.p_stream = torch.cuda.Stream(device=self.pdev)
        self.c_stream = torch.cuda.Stream(device=self.cdev)
        self._h: torch.Tensor | None = None
        self._counters: torch.Tensor | None = None
        if max_tokens:
            self._reserve(max_tokens)

    def _reserve(self, M: int) -> None:
        if self._h is not None and self._h.shape[0] >= M:
            return
        # Buffers live on the consumer: the producer writes them remotely once
        # per tile, the consumer polls/reads them locally many times.
        with torch.cuda.device(self.cdev):
            torch.cuda.synchronize(self.cdev)  # only on (re)allocation; never in steady state
            self._h = torch.empty((M, self.N1), dtype=self.w1.dtype, device=self.cdev)
            self._counters = torch.zeros(triton.cdiv(M, self.block_m), dtype=torch.int32, device=self.cdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.pdev or x.dim() != 2 or x.shape[1] != self.K:
            raise ValueError(f"expected x [M, {self.K}] on {self.pdev}, got {tuple(x.shape)} on {x.device}")
        M = x.shape[0]
        self._reserve(M)
        num_m = triton.cdiv(M, self.block_m)

        caller = torch.cuda.current_stream(self.pdev)
        out_stream = torch.cuda.current_stream(self.odev)
        x = x.contiguous()
        x.record_stream(self.p_stream)
        y = torch.empty((M, self.N2), dtype=self.w2.dtype, device=self.odev)
        y.record_stream(self.c_stream)

        ev_inputs = torch.cuda.Event()
        ev_inputs.record(caller)
        ev_out_free = torch.cuda.Event()
        ev_out_free.record(out_stream)

        # Consumer side first: reset the scoreboard. The c_stream is in order,
        # so this also waits for the previous forward's consumer to finish
        # reading ``_h`` before the producer may overwrite it.
        with torch.cuda.device(self.cdev), torch.cuda.stream(self.c_stream):
            self.c_stream.wait_event(ev_out_free)
            self._counters[:num_m].zero_()
            ev_reset = torch.cuda.Event()
            ev_reset.record(self.c_stream)

        with torch.cuda.device(self.pdev), torch.cuda.stream(self.p_stream):
            self.p_stream.wait_event(ev_inputs)
            self.p_stream.wait_event(ev_reset)
            launch_producer(
                x, self.w1, self.b1, self._h, self._counters,
                act=self.act, fence=self.fence,
                block_m=self.block_m, block_n=self.block_n1, block_k=self.block_k,
                num_warps=self.num_warps, num_stages=self.num_stages,
            )
            ev_prod = torch.cuda.Event()
            ev_prod.record(self.p_stream)

        with torch.cuda.device(self.cdev), torch.cuda.stream(self.c_stream):
            launch_consumer(
                self._h, self.w2, self.b2, y, self._counters,
                M=M, n_ready=self.n_ready, fence=self.fence,
                block_m=self.block_m, block_n=self.block_n2, block_k=self.block_k,
                num_warps=self.num_warps, num_stages=self.num_stages,
            )
            ev_done = torch.cuda.Event()
            ev_done.record(self.c_stream)

        caller.wait_event(ev_prod)
        caller.wait_event(ev_done)
        if self.odev != self.pdev:
            out_stream.wait_event(ev_done)
        return y

    __call__ = forward
