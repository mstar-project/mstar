"""Decode-sized linears in Triton: ``x @ w.T`` for a few rows, a weight stream.

With up to 16 rows a linear reads its weight once and does almost no math, so
the kernel's job is to keep enough loads in flight to stream the weight at
HBM speed. Each program reads a ``block_n`` x ``block_k`` weight tile per step
along K, for ``block_n`` output columns and one of ``split_k`` slices of K.
Split-K adds programs where a narrow weight has too few column tiles to fill
the GPU; its partial sums go to an fp32 workspace and the last program of a
column tile to arrive (an atomic counter per tile) adds them in order and
writes the output, so results do not depend on arrival order. The counters
are reset by that last program, so a captured graph replays without a memset.

``xcd_remap`` renumbers programs so that consecutive tiles, and so the split-K
partners of a tile, land on one XCD of an AMD GPU (the hardware hands out
workgroup ids round-robin across XCDs). ``use_dot`` computes each step on the
matrix cores with the rows padded to 16; without it each thread multiplies and
adds in fp32 and the reduction over K happens once, after the loop.

Epilogues fold the op that follows a linear into its store: ``silu_mul`` takes
a stacked ``[gate; up]`` weight and writes ``silu(gate) * up``, with both
rounded to the output dtype first, as the unfused pair would; a bias (or a
residual folded into it) is added before the one rounding.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Mapping
from typing import Any

import torch
import triton
import triton.language as tl

KNOBS = (
    "block_n", "block_k", "split_k", "num_warps", "waves_per_eu", "use_dot", "xcd_remap", "num_stages", "prefetch",
)
DEFAULT_KNOBS: Mapping[str, Any] = {
    "block_n": 16, "block_k": 256, "split_k": 1, "num_warps": 4, "waves_per_eu": 0, "use_dot": False,
    "xcd_remap": False, "num_stages": 1, "prefetch": False,
}
# column tiles whose split-K counters live in the per-device workspace; a
# launch with more tiles runs without split-K
MAX_COUNTED_TILES = 1 << 16
_DOT_MIN_ROWS = 16


@triton.jit
def _fma_step(acc3, w, x_ptr, offs_m, m_mask, ks, stride_xm, K):
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + ks[None, :], mask=m_mask[:, None] & (ks[None, :] < K), other=0.0)
    return acc3 + w[None, :, :].to(tl.float32) * x[:, None, :].to(tl.float32)


@triton.jit
def _tile_product(
    x_ptr, w_rows, offs_m, m_mask, k_start, k_stop, stride_xm, n_mask, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_DOT: tl.constexpr, PREFETCH: tl.constexpr, EVEN: tl.constexpr,
):
    """This program's [BLOCK_N, BLOCK_M] fp32 slice of ``w @ x.T`` over K in
    [k_start, k_stop)."""
    offs_k = tl.arange(0, BLOCK_K)
    if USE_DOT:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k0 in range(k_start, k_stop, BLOCK_K):
            ks = k0 + offs_k
            if EVEN:
                w = tl.load(w_rows + ks[None, :])
            else:
                w = tl.load(w_rows + ks[None, :], mask=n_mask[:, None] & (ks[None, :] < K), other=0.0)
            xt = tl.load(x_ptr + offs_m[None, :] * stride_xm + ks[:, None],
                         mask=m_mask[None, :] & (ks[:, None] < K), other=0.0)
            acc += tl.dot(w, xt)
    else:
        acc3 = tl.zeros((BLOCK_M, BLOCK_N, BLOCK_K), dtype=tl.float32)
        if PREFETCH:
            # Triton pipelines only loops around tl.dot: carry the next tile's
            # loads across the multiply-adds so two are in flight
            ks = k_start + offs_k
            if EVEN:
                w = tl.load(w_rows + ks[None, :])
            else:
                w = tl.load(w_rows + ks[None, :], mask=n_mask[:, None] & (ks[None, :] < K), other=0.0)
            for k0 in range(k_start + BLOCK_K, k_stop, BLOCK_K):
                ks_next = k0 + offs_k
                if EVEN:
                    w_next = tl.load(w_rows + ks_next[None, :])
                else:
                    w_next = tl.load(w_rows + ks_next[None, :], mask=n_mask[:, None] & (ks_next[None, :] < K),
                                     other=0.0)
                acc3 = _fma_step(acc3, w, x_ptr, offs_m, m_mask, ks, stride_xm, K)
                w, ks = w_next, ks_next
            acc3 = _fma_step(acc3, w, x_ptr, offs_m, m_mask, ks, stride_xm, K)
        else:
            for k0 in range(k_start, k_stop, BLOCK_K):
                ks = k0 + offs_k
                if EVEN:
                    w = tl.load(w_rows + ks[None, :])
                else:
                    w = tl.load(w_rows + ks[None, :], mask=n_mask[:, None] & (ks[None, :] < K), other=0.0)
                acc3 = _fma_step(acc3, w, x_ptr, offs_m, m_mask, ks, stride_xm, K)
        acc = tl.trans(tl.sum(acc3, axis=2))
    return acc


@triton.jit
def _decode_gemv(
    x_ptr, w_ptr, out_ptr, bias_ptr, part_ptr, count_ptr,
    M, N, K, stride_xm, stride_wn, stride_om, stride_bm, up_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    SILU_MUL: tl.constexpr, HAS_BIAS: tl.constexpr, USE_DOT: tl.constexpr, PREFETCH: tl.constexpr,
    XCD_REMAP: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if XCD_REMAP:
        # the host launches a multiple of NUM_XCDS programs when remapping
        pid = (pid % NUM_XCDS) * (tl.num_programs(0) // NUM_XCDS) + pid // NUM_XCDS
    tile = pid // SPLIT_K
    split = pid % SPLIT_K
    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    k_per_split = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_start = split * k_per_split
    k_stop = tl.minimum(k_start + k_per_split, K)

    w_rows = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn
    acc = _tile_product(x_ptr, w_rows, offs_m, m_mask, k_start, k_stop, stride_xm, n_mask, K,
                        BLOCK_M, BLOCK_N, BLOCK_K, USE_DOT, PREFETCH, EVEN)
    if SILU_MUL:
        acc_up = _tile_product(x_ptr, w_rows + up_rows * stride_wn, offs_m, m_mask, k_start, k_stop, stride_xm,
                               n_mask, K, BLOCK_M, BLOCK_N, BLOCK_K, USE_DOT, PREFETCH, EVEN)
    else:
        acc_up = acc

    mn_mask = n_mask[:, None] & m_mask[None, :]
    last = True
    if SPLIT_K > 1:
        # [SPLIT_K, (gate, up), M, N] partial sums
        part = part_ptr + offs_m[None, :] * N + offs_n[:, None]
        slot = 2 * M * N
        tl.store(part + split * slot, acc, mask=mn_mask)
        if SILU_MUL:
            tl.store(part + split * slot + M * N, acc_up, mask=mn_mask)
        # release this program's partials, acquire every earlier arrival's
        arrived = tl.atomic_add(count_ptr + tile, 1, sem="acq_rel", scope="gpu")
        last = arrived == SPLIT_K - 1
        if last:
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            acc_up = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                acc += tl.load(part + s * slot, mask=mn_mask, other=0.0)
                if SILU_MUL:
                    acc_up += tl.load(part + s * slot + M * N, mask=mn_mask, other=0.0)
            tl.store(count_ptr + tile, 0)
    if last:
        dtype = out_ptr.dtype.element_ty
        if SILU_MUL:
            gate = acc.to(dtype).to(tl.float32)
            up = acc_up.to(dtype).to(tl.float32)
            y = gate * tl.sigmoid(gate) * up
        else:
            y = acc
            if HAS_BIAS:
                y += tl.load(bias_ptr + offs_m[None, :] * stride_bm + offs_n[:, None], mask=mn_mask,
                             other=0.0).to(tl.float32)
        tl.store(out_ptr + offs_m[None, :] * stride_om + offs_n[:, None], y.to(dtype), mask=mn_mask)


class _Counters:
    """One zeroed int32 counter per column tile, per device; the kernel leaves
    them zeroed. Allocated on first use, which must not be inside a capture
    (the warm-up runs every shape first)."""

    def __init__(self):
        self._by_device: dict[torch.device, torch.Tensor] = {}
        self._lock = threading.Lock()

    def get(self, device: torch.device) -> torch.Tensor:
        counters = self._by_device.get(device)
        if counters is None:
            with self._lock:
                counters = self._by_device.get(device)
                if counters is None:
                    counters = torch.zeros(MAX_COUNTED_TILES, dtype=torch.int32, device=device)
                    self._by_device[device] = counters
        return counters


_COUNTERS = _Counters()


@functools.cache
def _num_xcds(device_index: int) -> int:
    from mstar.platforms import current_platform

    return current_platform.device_facts(device_index).xcds or 1


def _launch(x, weight, out, bias, n_out, up_rows, silu_mul, knobs) -> torch.Tensor:
    rows, k = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    block_n, block_k = int(knobs["block_n"]), int(knobs["block_k"])
    use_dot = bool(knobs.get("use_dot", False))
    block_m = max(_DOT_MIN_ROWS if use_dot else 1, triton.next_power_of_2(rows))
    tiles = triton.cdiv(n_out, block_n)
    split_k = int(knobs.get("split_k", 1))
    split_k = max(1, min(split_k, triton.cdiv(k, block_k)))
    if tiles > MAX_COUNTED_TILES:
        split_k = 1
    programs = tiles * split_k
    num_xcds = _num_xcds(x.device.index or 0) if torch.version.hip else 1
    remap = bool(knobs.get("xcd_remap", False)) and num_xcds > 1 and programs % num_xcds == 0
    if split_k > 1:
        part = torch.empty((split_k, 2, rows, n_out), dtype=torch.float32, device=x.device)
        counters = _COUNTERS.get(x.device)
    else:
        part = counters = out
    if bias is not None:
        stride_bm = bias.stride(0) if bias.dim() == 2 else 0
    else:
        bias, stride_bm = out, 0
    even = n_out % block_n == 0 and k % (block_k * split_k) == 0
    options = {"num_warps": int(knobs.get("num_warps", 4)), "num_stages": int(knobs.get("num_stages", 1))}
    if torch.version.hip:
        options["waves_per_eu"] = int(knobs.get("waves_per_eu", 0))
    _decode_gemv[(programs,)](
        x, weight, out, bias, part, counters,
        rows, n_out, k, x.stride(0), weight.stride(0), out.stride(0), stride_bm, up_rows,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        SILU_MUL=silu_mul, HAS_BIAS=bias is not out, USE_DOT=use_dot, PREFETCH=bool(knobs.get("prefetch", False)),
        XCD_REMAP=remap, NUM_XCDS=num_xcds, EVEN=even, **options,
    )
    return out


def decode_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    knobs: Mapping[str, Any] = DEFAULT_KNOBS,
) -> torch.Tensor:
    """``x @ weight.T (+ bias)``: x ``[rows, K]``, weight ``[N, K]`` contiguous,
    bias ``[N]`` or ``[rows, N]``."""
    n = weight.shape[0]
    out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
    if x.shape[0] == 0:
        return out
    return _launch(x, weight, out, bias, n, 0, False, knobs)


def decode_gemv_silu_mul(
    x: torch.Tensor, weight: torch.Tensor, knobs: Mapping[str, Any] = DEFAULT_KNOBS,
) -> torch.Tensor:
    """``silu(x @ gate.T) * (x @ up.T)`` for ``weight = [gate; up]``, ``[2I, K]``."""
    inter = weight.shape[0] // 2
    out = torch.empty((x.shape[0], inter), dtype=x.dtype, device=x.device)
    if x.shape[0] == 0:
        return out
    return _launch(x, weight, out, None, inter, inter, True, knobs)
