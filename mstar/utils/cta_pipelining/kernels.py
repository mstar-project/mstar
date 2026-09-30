"""Triton kernels for CTA-pipelining a two-layer MLP across two GPUs.

Implements the protocol of *CTA-Pipelining: A Latency-Oriented Spatial Scaling
Method for Multi-GPU Systems* (Liu et al., arXiv:2607.07862) for the 2-GPU,
2-layer-GEMM case that represents a transformer FFN::

    H = act(X @ W1^T + b1)        # producer kernel, runs on GPU 0
    Y = H @ W2^T + b2             # consumer kernel, runs on GPU 1, concurrently

Instead of tensor-parallel sharding (both GPUs compute half of every GEMM, then
all-reduce), each GPU owns one whole GEMM and the two kernels are resident at the
same time. Producer CTAs write their finished ``H`` tile straight into a buffer
that lives in the *consumer's* memory (an NVLink peer write), publish it with a
system-scope release atomic on a per-row-block counter that also lives in the
consumer's memory, and move on. Consumer CTAs spin on that counter (a local read
for them, as the paper recommends) until every producer tile of their row block
has landed, then run an unmodified GEMM tile. Both kernels are persistent and walk
tiles in the same row-major order, so rows become ready in roughly the order the
consumer wants them.

Simplifications relative to the paper (see README.md for the full list):

* The scoreboard is one counter per *row block* rather than per consumer CTA,
  and it lives on the consumer device. Every producer tile therefore does one
  remote atomic instead of a local atomic plus an occasional remote queue push.
* There is no dynamic work queue. The consumer polls its statically assigned
  next tile; because both kernels enumerate tiles row-major this is close to the
  paper's dynamic order for a balanced two-layer MLP.
* The fence is a full ``fence.acq_rel.sys`` (CUDA ``__threadfence_system``) after
  the tile store and before the release atomic, as in the paper's classical-kernel
  integration. No Lamport-style sentinel scheme.

The kernels are plain ``tl.load``/``tl.dot`` GEMMs (no TMA, no warp
specialization), so the protocol overhead is *not* hidden the way the paper's
SM100 warp-specialized integration hides it. This is the "classical kernel"
variant of the paper.

``FENCE`` is a constexpr so the kernels also run under ``TRITON_INTERPRET=1`` (the
interpreter has no inline asm); the CPU test drives them sequentially there.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

ACT_NONE = 0
ACT_GELU_TANH = 1
ACT_SILU = 2

ACTIVATIONS = {"none": ACT_NONE, "gelu_tanh": ACT_GELU_TANH, "silu": ACT_SILU}

# Default tile configuration. One row block of the producer (``BLOCK_M`` rows,
# all ``N1`` columns) is the unit of readiness the consumer waits on, so
# ``BLOCK_M`` must be identical for both kernels.
DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 64
DEFAULT_NUM_WARPS = 8
DEFAULT_NUM_STAGES = 3


@triton.jit
def _fence_sys():
    # ``fence.acq_rel.sys``: makes this thread's earlier global writes (the H
    # tile stored into peer memory) visible system-wide before anything it does
    # afterwards (the release atomic). The CUDA equivalent is
    # ``__threadfence_system()``. Same no-arg inline-asm idiom as Triton's own
    # ``triton.language.extra.cuda.gdc``.
    tl.inline_asm_elementwise("fence.acq_rel.sys; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _apply_act(x, ACT: tl.constexpr):
    y = x
    if ACT == 1:
        # gelu, tanh approximation, computed in fp32 on the accumulator.
        # tanh(z) = 2 / (1 + exp(-2z)) - 1 saturates correctly at both ends.
        z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        t = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
        y = 0.5 * x * (1.0 + t)
    if ACT == 2:
        y = x / (1.0 + tl.exp(-x))
    return y


@triton.jit
def _gemm_tile(
    a_ptr,
    w_ptr,
    bias_ptr,
    pid_m,
    pid_n,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    HAS_BIAS: tl.constexpr,
    A_CACHE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` fp32 tile of ``A @ W^T (+ bias)`` for an
    ``nn.Linear``-layout weight ``W[N, K]``."""
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
            other=0.0,
            cache_modifier=A_CACHE,
        )
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    return acc


@triton.jit
def cta_pipe_producer_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    h_ptr,  # [M, N] activation buffer; lives in the CONSUMER GPU's memory
    cnt_ptr,  # [cdiv(M, BLOCK_M)] int32 row-block counters; CONSUMER GPU's memory
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_w1n,
    stride_w1k,
    stride_hm,
    stride_hn,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    FENCE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    """Persistent GEMM ``H = act(X @ W1^T + b1)`` whose epilogue publishes each
    tile to the consumer GPU (paper Fig. 1, arrows 1-4, with the scoreboard and
    the queue collapsed into one remote counter per row block)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    for tile in range(pid, num_tiles, NUM_PROGRAMS):
        # Row-major tile walk: consecutive programs work on the same row block,
        # so whole row blocks complete early and in order.
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        acc = _gemm_tile(
            x_ptr, w1_ptr, b1_ptr, pid_m, pid_n, M, N, K,
            stride_xm, stride_xk, stride_w1n, stride_w1k,
            HAS_BIAS=HAS_BIAS, A_CACHE="", BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        acc = _apply_act(acc, ACT)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
        # (1) peer write of the finished tile into the consumer's H buffer.
        tl.store(h_ptrs, acc.to(h_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        # Every thread's stores are issued before any thread signals.
        tl.debug_barrier()
        if FENCE:
            _fence_sys()
        # (2-4) publish: one system-scope release increment of the row counter.
        # Triton issues a scalar atomic from a single thread of the CTA.
        tl.atomic_add(cnt_ptr + pid_m, 1, sem="release", scope="sys")


@triton.jit
def cta_pipe_consumer_kernel(
    h_ptr,  # [M, K] activation buffer, local to this (consumer) GPU
    w2_ptr,
    b2_ptr,
    y_ptr,  # [M, N] output; may live on either GPU
    cnt_ptr,  # [cdiv(M, BLOCK_M)] int32 row-block counters, local
    M,
    N,
    K,
    n_ready,  # producer tiles per row block == cdiv(N1, BLOCK_N1)
    stride_hm,
    stride_hk,
    stride_w2n,
    stride_w2k,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    FENCE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    """Persistent GEMM ``Y = H @ W2^T + b2`` whose prologue waits for the row
    block it is about to read (paper Fig. 1, arrow 5)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    for tile in range(pid, num_tiles, NUM_PROGRAMS):
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n
        # (5) busy-poll the local row counter with a system-scope acquire. A
        # scalar atomic is executed by one thread and broadcast, so the whole
        # CTA sees the same value and leaves the loop together.
        ready = tl.atomic_add(cnt_ptr + pid_m, 0, sem="acquire", scope="sys")
        while ready < n_ready:
            ready = tl.atomic_add(cnt_ptr + pid_m, 0, sem="acquire", scope="sys")
        if FENCE:
            _fence_sys()
        tl.debug_barrier()
        # H was written by the peer GPU: read it around L1 (``.cg``) so no
        # stale line from an earlier tile of this persistent program is used.
        acc = _gemm_tile(
            h_ptr, w2_ptr, b2_ptr, pid_m, pid_n, M, N, K,
            stride_hm, stride_hk, stride_w2n, stride_w2k,
            HAS_BIAS=HAS_BIAS, A_CACHE=".cg", BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _num_programs(device: torch.device, num_tiles: int) -> int:
    if device.type == "cuda":
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    else:
        sms = 4  # interpreter / CPU: any small persistent grid
    return max(1, min(sms, num_tiles))


def launch_producer(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    h: torch.Tensor,
    counters: torch.Tensor,
    *,
    act: int,
    fence: bool = True,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
) -> None:
    """Launch the producer on the *current* device/stream. ``x``/``w1``/``b1``
    are local; ``h`` and ``counters`` may be peer-device tensors (their
    ``data_ptr`` is a valid unified address once peer access is enabled)."""
    M, K = x.shape
    N = w1.shape[0]
    assert w1.shape[1] == K and h.shape[0] >= M and h.shape[1] == N
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    assert counters.numel() >= triton.cdiv(M, block_m)
    nprog = _num_programs(x.device, num_tiles)
    cta_pipe_producer_kernel[(nprog,)](
        x, w1, b1 if b1 is not None else w1, h, counters,
        M, N, K,
        x.stride(0), x.stride(1), w1.stride(0), w1.stride(1), h.stride(0), h.stride(1),
        HAS_BIAS=b1 is not None, ACT=act, FENCE=fence,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, NUM_PROGRAMS=nprog,
        num_warps=num_warps, num_stages=num_stages,
    )


def launch_consumer(
    h: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    y: torch.Tensor,
    counters: torch.Tensor,
    *,
    M: int,
    n_ready: int,
    fence: bool = True,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
) -> None:
    """Launch the consumer on the *current* device/stream. ``h``, ``counters``,
    ``w2``, ``b2`` are local; ``y`` may be a peer-device tensor."""
    K = w2.shape[1]
    N = w2.shape[0]
    assert h.shape[1] == K and y.shape[0] >= M and y.shape[1] == N
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    nprog = _num_programs(h.device, num_tiles)
    cta_pipe_consumer_kernel[(nprog,)](
        h, w2, b2 if b2 is not None else w2, y, counters,
        M, N, K, n_ready,
        h.stride(0), h.stride(1), w2.stride(0), w2.stride(1), y.stride(0), y.stride(1),
        HAS_BIAS=b2 is not None, FENCE=fence,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, NUM_PROGRAMS=nprog,
        num_warps=num_warps, num_stages=num_stages,
    )
