# SPDX-License-Identifier: Apache-2.0
# The int4 decode follows the int4 (``use_int4_w4a16``) branch of SGLang's / vLLM's ``fused_moe_kernel_gptq_awq`` and
# vLLM's ``awq_triton.py`` (Apache-2.0, Copyright the vLLM and SGLang authors). Changed here: a weight layout
# shuffled at load, weight rows on the MMA's M side, scales and zero points applied after the dot, a CUDA-core GEMV
# for a few tokens, and an e2m1 + e8m0 (MXFP4) decode beside the int4 one.
"""The CUDA kernels of ``mstar::w4a16_linear`` (``model.components.int4_formats``): ``x @ W.T`` for a 4-bit ``W``.

``W`` is int32 ``[N, K/8]`` in the planes layout (``int4_formats.to_planes``, built at load from compressed-tensors'
or an MXFP4 checkpoint's order): in each 128-column block of a row, word i's nibble j is column 16 j + i, so one
nibble of a block's 16 words is 16 contiguous columns. Machete and Marlin shuffle a weight at load into their MMA's
operand order for the same reason: the kernel decodes the weight where the dot reads it, and x's slice for it is
contiguous. An int4 code becomes ``code - 8`` in fp32 without a conversion instruction (``code | 0x4B000000`` is the
float ``2**23 + code``), and scales and zero points apply to the dot's output, ``scale * (dot + (8 - zero) *
sum(x))``, so the decode feeding a dot has no per-row operand. Paths:

- ``tinygemm`` (1-2 tokens): torch's int4 GEMM (``aten._weight_int4pack_mm``) on its own copy of the weight, which
  ``int4_formats`` builds at load for a symmetric or zero-point int4 weight in groups of 32-256;
- ``gemv`` (a few tokens): CUDA-core FMAs over a word's 8 nibbles in turn, each weight decoded once for all tokens,
  reduced once at the end; split along K;
- ``dot`` / ``dot8``: weight rows on the MMA's M side and tokens (padded to 16) on its N side; ``dot`` unpacks a
  block in column order and runs one dot per scale group, ``dot8`` one dot per nibble position with no unpacking;
  split along K;
- ``dequant``: past ``DEQUANT_M`` tokens, the weight dequantized to the activation's dtype and cuBLAS runs it.

Which path and tile runs at a token count is ``config``'s built-in rule, a hypothesis for other GPUs and shapes.
Every path matches the reference op (``int4_formats.dequantize_kind`` then a GEMM) to fp32 accumulation order.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch
import triton
import triton.language as tl

from mstar.model.components.int4_formats import KIND_E2M1, KIND_INT4_ZP, w4a16_linear

TINYGEMM_MAX = 2
GEMV_MAX = 4
DEQUANT_M = 192


@triton.jit
def _codes(words, ROWS: tl.constexpr, COLS: tl.constexpr):
    """int32 words ``[ROWS, COLS // 8]`` -> their 4-bit codes ``[ROWS, COLS]``: word i's nibble j is column 8i + j."""
    shifts = tl.arange(0, 8) * 4
    return tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (ROWS, COLS))


@triton.jit
def _e2m1(codes):
    """fp32 values of e2m1 codes: magnitude e:m (3 bits) as an fp32 exponent / mantissa, (e + 126) << 23 | m << 22,
    exact for e >= 1; e = 0 is 0 or 0.5."""
    v = ((((codes & 7) << 22) + (126 << 23)) | ((codes & 8) << 28)).to(tl.float32, bitcast=True)
    small = (codes & 1).to(tl.float32) * 0.5
    small = tl.where((codes & 8) != 0, -small, small)
    return tl.where((codes & 6) == 0, small, v)


@triton.jit
def _values(codes, zf, KIND: tl.constexpr):
    """fp32 values of 4-bit codes: an e2m1 code's value, or an int4 code minus its zero point, given as
    ``zf = 2**23 + zero`` (a float or a tensor broadcasting against ``codes``)."""
    if KIND == 2:
        return _e2m1(codes)
    return (codes | 0x4B000000).to(tl.float32, bitcast=True) - zf


@triton.jit
def _group_scale(s_ptr, n, live, g, stride_sn, KIND: tl.constexpr):
    """Rows ``n``'s fp32 scale at group ``g`` (shapes broadcast): e8m0 for MXFP4, the checkpoint's float for int4."""
    if KIND == 2:
        return tl.exp2(tl.load(s_ptr + n * stride_sn + g, mask=live, other=127).to(tl.float32) - 127.0)
    return tl.load(s_ptr + n * stride_sn + g, mask=live, other=0.0).to(tl.float32)


@triton.jit
def _group_zero(z_ptr, n, live, g, stride_zn):
    """Rows ``n``'s zero point at group ``g`` as ``2**23 + zero`` (fp32), from the words packing 8 rows each."""
    zw = tl.load(z_ptr + (n // 8) * stride_zn + g, mask=live, other=0)
    return 8388608.0 + ((zw >> ((n % 8) * 4)) & 0xF).to(tl.float32)


@triton.jit
def _plane_params(s_ptr, z_ptr, n, live, g, stride_sn, stride_zn, KIND: tl.constexpr):
    """Rows ``n``'s scale and ``2**23 + zero`` at group ``g`` (the zero a float constant without zero points)."""
    scale = _group_scale(s_ptr, n, live, g, stride_sn, KIND)
    if KIND == 1:
        return scale, _group_zero(z_ptr, n, live, g, stride_zn)
    return scale, scale * 0.0 + 8388616.0


@triton.jit
def _plane_codes(words, j0: tl.constexpr, P: tl.constexpr, BN: tl.constexpr):
    """Nibbles j0 .. j0 + P - 1 of a block's words ``[BN, 16]`` as codes ``[BN, 16 P]`` in column order (the planes
    layout puts nibble j of word i at column 16 j + i)."""
    shifts = 4 * (j0 + tl.arange(0, P))
    return tl.reshape((words[:, None, :] >> shifts[None, :, None]) & 0xF, (BN, 16 * P))


@triton.jit
def _chunk_update(acc, part, xsum, s_ptr, z_ptr, n, live_n, g, stride_sn, stride_zn, KIND: tl.constexpr):
    """``acc + scale * (part + (8 - zero) * xsum)`` for one scale group's partial ``part = (code - 8) @ x`` and
    ``xsum`` its x columns' sum (``[BN, TB]``, each row the same; the zero term only with zero points): scales and
    zero points meet the dot's output, so the weight decode before the dot has no per-row operand."""
    scale = _group_scale(s_ptr, n, live_n, g, stride_sn, KIND)
    if KIND == 1:
        part = part + (8388616.0 - _group_zero(z_ptr, n, live_n, g, stride_zn))[:, None] * xsum
    return acc + part * scale[:, None]


@triton.jit
def _w4_dot_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr, M, N, K,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_os, stride_om,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, TB: tl.constexpr, NB: tl.constexpr,
    SPLIT_K: tl.constexpr, STAGES: tl.constexpr,
):
    """``out[pid_k, t, n]`` for BN weight rows x TB tokens. A step is NB 128-column blocks of the planes layout (16
    words a row each); a block is one dot per scale group (K = min(GROUP, 128)) of its codes, unpacked in column
    order, against x's contiguous columns. Split ``pid_k`` takes steps pid_k, pid_k + SPLIT_K, ..."""
    P: tl.constexpr = 8 if GROUP >= 128 else GROUP // 16
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)
    n = pid_n * BN + tl.arange(0, BN)
    live_n = n < N
    t = pid_t * TB + tl.arange(0, TB)
    live_t = t < M
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_wn
    x_rows = x_ptr + t[:, None].to(tl.int64) * stride_xm
    i16 = tl.arange(0, 16)
    cols = tl.arange(0, 16 * P)
    acc = tl.zeros((BN, TB), dtype=tl.float32)
    for k0 in tl.range(pid_k * NB * 128, K, SPLIT_K * NB * 128, num_stages=STAGES):
        for b in tl.static_range(NB):
            kb = k0 + 128 * b
            words = tl.load(w_rows + (kb // 8 + i16)[None, :], mask=live_n[:, None], other=0)
            for c in tl.static_range(8 // P):
                kc = kb + 16 * P * c
                v = _values(_plane_codes(words, P * c, P, BN), 8388616.0, KIND)
                xk = tl.load(x_rows + (kc + cols)[None, :], mask=live_t[:, None], other=0.0)
                part = tl.dot(v.to(xk.dtype), tl.trans(xk))
                xsum = part
                if KIND == 1:
                    # x's column sums broadcast over the rows, from the tensor cores (a reduction across the
                    # tile's threads measured ~80-320 shuffles a block in its PTX)
                    xsum = tl.dot(tl.full((BN, 16 * P), 1.0, tl.float32).to(xk.dtype), tl.trans(xk))
                acc = _chunk_update(acc, part, xsum, s_ptr, z_ptr, n, live_n, kc // GROUP, stride_sn, stride_zn,
                                    KIND)
    out = out_ptr + pid_k * stride_os + t[None, :].to(tl.int64) * stride_om + n[:, None]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=live_n[:, None] & live_t[None, :])


@triton.jit
def _w4_dot8_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr, M, N, K,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_os, stride_om,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, TB: tl.constexpr, NB: tl.constexpr,
    SPLIT_K: tl.constexpr, STAGES: tl.constexpr,
):
    """``dot`` with one dot per nibble position (K = 16): nibble j of a block's words is the weight of x's 16
    contiguous columns 16 j.., so the words go to the dot's register layout once and every nibble decodes in place
    (no unpacking reshape), at the price of eight small dots a block."""
    P: tl.constexpr = 8 if GROUP >= 128 else GROUP // 16
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)
    n = pid_n * BN + tl.arange(0, BN)
    live_n = n < N
    t = pid_t * TB + tl.arange(0, TB)
    live_t = t < M
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_wn
    x_rows = x_ptr + t[:, None].to(tl.int64) * stride_xm
    i16 = tl.arange(0, 16)
    acc = tl.zeros((BN, TB), dtype=tl.float32)
    for k0 in tl.range(pid_k * NB * 128, K, SPLIT_K * NB * 128, num_stages=STAGES):
        for b in tl.static_range(NB):
            kb = k0 + 128 * b
            words = tl.load(w_rows + (kb // 8 + i16)[None, :], mask=live_n[:, None], other=0)
            for c in tl.static_range(8 // P):
                part = tl.zeros((BN, TB), dtype=tl.float32)
                xsum = tl.zeros((TB, 16), dtype=tl.float32)
                for p in tl.static_range(P):
                    j = P * c + p
                    v = _values((words >> (4 * j)) & 0xF, 8388616.0, KIND)
                    xj = tl.load(x_rows + (kb + 16 * j + i16)[None, :], mask=live_t[:, None], other=0.0)
                    part = tl.dot(v.to(xj.dtype), tl.trans(xj), part)
                    if KIND == 1:
                        xsum += xj.to(tl.float32)
                if KIND == 1:
                    xsum = tl.broadcast_to(tl.sum(xsum, axis=1)[None, :], (BN, TB))
                acc = _chunk_update(acc, part, xsum, s_ptr, z_ptr, n, live_n, (kb + 16 * P * c) // GROUP, stride_sn,
                                    stride_zn, KIND)
    out = out_ptr + pid_k * stride_os + t[None, :].to(tl.int64) * stride_om + n[:, None]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=live_n[:, None] & live_t[None, :])


@triton.jit
def _w4_gemv_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr, M, N, K,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_os, stride_om,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, NB: tl.constexpr, SPLIT_K: tl.constexpr,
    MB: tl.constexpr,
):
    """``out[pid_k, m, n]`` for BN weight rows and up to 4 tokens on CUDA cores. A step loads 16 NB words a row (NB
    128-column blocks); nibble j of every word meets x's columns 128 (w // 16) + 16j + w % 16, contiguous by 16.
    Accumulators keep the word axis (one per token, 2-D, no broadcast between layouts) and are reduced once after
    the loop; with a group of at least 128, a step's products are summed before their one scale multiply."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    live_n = n < N
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_wn
    wi = tl.arange(0, 16 * NB)
    col = ((wi // 16) * 128 + wi % 16)[None, :]
    acc0 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
    acc1 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
    acc2 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
    acc3 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
    for k0 in range(pid_k * NB * 128, K, SPLIT_K * NB * 128):
        words = tl.load(w_rows + (k0 // 8 + wi)[None, :], mask=live_n[:, None], other=0)
        if GROUP >= 128:
            scale, zf = _plane_params(s_ptr, z_ptr, n[:, None], live_n[:, None], (k0 + col - col % 128) // GROUP,
                                      stride_sn, stride_zn, KIND)
            p0 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
            p1 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
            p2 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
            p3 = tl.zeros((BN, 16 * NB), dtype=tl.float32)
            for j in tl.static_range(8):
                v = _values((words >> (4 * j)) & 0xF, zf, KIND)
                xo = x_ptr + k0 + col + 16 * j
                p0 += v * tl.load(xo, mask=0 < M, other=0.0).to(tl.float32)
                if MB > 1:
                    p1 += v * tl.load(xo + stride_xm, mask=1 < M, other=0.0).to(tl.float32)
                if MB > 2:
                    p2 += v * tl.load(xo + 2 * stride_xm, mask=2 < M, other=0.0).to(tl.float32)
                    p3 += v * tl.load(xo + 3 * stride_xm, mask=3 < M, other=0.0).to(tl.float32)
            acc0 += p0 * scale
            if MB > 1:
                acc1 += p1 * scale
            if MB > 2:
                acc2 += p2 * scale
                acc3 += p3 * scale
        else:
            for j in tl.static_range(8):
                scale, zf = _plane_params(s_ptr, z_ptr, n[:, None], live_n[:, None],
                                          (k0 + col - col % 128 + 16 * j) // GROUP, stride_sn, stride_zn, KIND)
                v = _values((words >> (4 * j)) & 0xF, zf, KIND) * scale
                xo = x_ptr + k0 + col + 16 * j
                acc0 += v * tl.load(xo, mask=0 < M, other=0.0).to(tl.float32)
                if MB > 1:
                    acc1 += v * tl.load(xo + stride_xm, mask=1 < M, other=0.0).to(tl.float32)
                if MB > 2:
                    acc2 += v * tl.load(xo + 2 * stride_xm, mask=2 < M, other=0.0).to(tl.float32)
                    acc3 += v * tl.load(xo + 3 * stride_xm, mask=3 < M, other=0.0).to(tl.float32)
    out = out_ptr + pid_k * stride_os + n
    dtype = out_ptr.dtype.element_ty
    tl.store(out, tl.sum(acc0, axis=1).to(dtype), mask=live_n)
    if MB > 1:
        tl.store(out + stride_om, tl.sum(acc1, axis=1).to(dtype), mask=live_n & (1 < M))
    if MB > 2:
        tl.store(out + 2 * stride_om, tl.sum(acc2, axis=1).to(dtype), mask=live_n & (2 < M))
        tl.store(out + 3 * stride_om, tl.sum(acc3, axis=1).to(dtype), mask=live_n & (3 < M))


@triton.jit
def _w4_dequant_kernel(
    w_ptr, s_ptr, z_ptr, out_ptr, N, K, stride_wn, stride_sn, stride_zn, stride_on,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr,
):
    """One 128-column block of BN rows from the planes layout, nibble j to columns 16j..16j + 15."""
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    kb = tl.program_id(1) * 128
    live_n = n < N
    i16 = tl.arange(0, 16)
    words = tl.load(w_ptr + n[:, None].to(tl.int64) * stride_wn + (kb // 8 + i16)[None, :], mask=live_n[:, None],
                    other=0)
    for j in tl.static_range(8):
        kc = kb + 16 * j
        scale, zf = _plane_params(s_ptr, z_ptr, n, live_n, kc // GROUP, stride_sn, stride_zn, KIND)
        v = _values((words >> (4 * j)) & 0xF, zf[:, None], KIND) * scale[:, None]
        tl.store(out_ptr + n[:, None].to(tl.int64) * stride_on + (kc + i16)[None, :],
                 v.to(out_ptr.dtype.element_ty), mask=live_n[:, None])


@lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _split(ctas: int, k_steps: int, sms: int, waves: int) -> int:
    return max(1, min(16, k_steps, (waves * sms) // max(ctas, 1)))


def _tb(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


def _nb(blocks: int, want: int) -> int:
    """The largest of want, want / 2, ..., 1 dividing ``blocks`` (a step must not run past K)."""
    while blocks % want:
        want //= 2
    return want


def config(m: int, n: int, k: int, group: int, kind: int, sms: int) -> tuple[str, dict]:
    """The built-in path and tile for ``m`` tokens of an ``[n, k]`` weight: the H200 winners of ``bench_w4a16`` at
    Qwen3.5-27B's shapes for the compressed-tensors layout (job 1492, ``res/w4r3k``) carried over to the planes
    layout, so a hypothesis until a sweep of these kernels: the GEMV at one token (MXFP4 to four; tinygemm takes
    int4's one and two), ``dot`` with 16 tokens a tile and 128 rows (8 warps) for N under ~128 tiles or 64 rows and
    two blocks a step above, 64 rows for more tokens (tiles of up to 64), and dequant + cuBLAS from ``DEQUANT_M``
    tokens."""
    if m >= DEQUANT_M:
        return "dequant", {}
    blocks = k // 128
    if m == 1 or (kind == KIND_E2M1 and m <= GEMV_MAX):
        nb = _nb(blocks, 4)
        return "gemv", {"BN": 16, "NB": nb, "SPLIT_K": _split(triton.cdiv(n, 16), blocks // nb, sms, 4),
                        "num_warps": 4}
    tb = _tb(m)
    wide = tb == 16 and triton.cdiv(n, 128) < sms
    # 64 x 64 tiles spill at 4 warps (255 registers in the sm_90 PTX), not at 8
    bn, warps = (128, 8) if wide else (64, 8 if tb == 64 else 4)
    nb = _nb(blocks, 2) if tb == 16 and not wide else 1
    ctas = triton.cdiv(n, bn) * triton.cdiv(m, tb)
    return "dot", {"BN": bn, "TB": tb, "NB": nb, "SPLIT_K": _split(ctas, blocks // nb, sms, 2),
                   "STAGES": 2 if nb == 2 else 3, "num_warps": warps}


# a stand-in for ``config`` (the benchmark sweeps tiles through it); None runs the built-in rule
CONFIG_OVERRIDE: Callable[[int, int, int, int, int, int], tuple[str, dict]] | None = None


def _words(weight: torch.Tensor) -> torch.Tensor:
    return weight.view(torch.int32) if weight.dtype == torch.uint8 else weight


def _scale_zero(scale, zero_point):
    # the kernels step along a scale or zero-point row by one element
    scale = scale.contiguous()
    zero_point = zero_point.contiguous() if zero_point is not None else None
    return (scale, scale.stride(0), zero_point if zero_point is not None else scale,
            zero_point.stride(0) if zero_point is not None else 0)


def dequantize(weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor | None, group: int, kind: int,
               dtype: torch.dtype) -> torch.Tensor:
    """The ``[N, K]`` weight (planes layout in) in ``dtype``: the large-M path's scratch, and tests."""
    words = _words(weight)
    n, k = words.shape[0], words.shape[1] * 8
    out = torch.empty((n, k), dtype=dtype, device=weight.device)
    s, ssn, z, szn = _scale_zero(scale, zero_point)
    bn = 32
    _w4_dequant_kernel[(triton.cdiv(n, bn), k // 128)](
        words, s, z, out, n, k, words.stride(0), ssn, szn, out.stride(0), KIND=kind, GROUP=group, BN=bn,
        num_warps=4,
    )
    return out


def gemm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor | None, group: int,
         kind: int, tinygemm_weight: torch.Tensor | None = None,
         tinygemm_scale_zero: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ W.T`` for ``W`` in the planes layout (uint8 ``[N, K/2]`` or int32 ``[N, K/8]``), K a multiple of 128."""
    words = _words(weight)
    m, k = x.shape
    n = words.shape[0]
    assert words.shape[1] * 8 == k and k % 128 == 0 and words.is_contiguous(), (tuple(weight.shape), k)
    assert kind != KIND_INT4_ZP or zero_point is not None
    if (CONFIG_OVERRIDE is None and m <= TINYGEMM_MAX and tinygemm_weight is not None and tinygemm_weight.numel()
            and x.dtype == tinygemm_scale_zero.dtype):
        return torch.ops.aten._weight_int4pack_mm(x, tinygemm_weight, group, tinygemm_scale_zero)
    path, cfg = (CONFIG_OVERRIDE or config)(m, n, k, group, kind, _sm_count(x.device.index or 0))
    if path == "tinygemm":
        return torch.ops.aten._weight_int4pack_mm(x, tinygemm_weight, group, tinygemm_scale_zero)
    if path == "dequant":
        return torch.nn.functional.linear(x, dequantize(weight, scale, zero_point, group, kind, x.dtype))
    split = cfg["SPLIT_K"]
    assert (k // 128) % cfg["NB"] == 0, (k, cfg)  # a step of NB blocks never runs past K
    out = torch.empty((split, m, n), dtype=torch.float32 if split > 1 else x.dtype, device=x.device)
    s, ssn, z, szn = _scale_zero(scale, zero_point)
    common = (x, words, s, z, out, m, n, k, x.stride(0), words.stride(0), ssn, szn, out.stride(0), out.stride(1))
    if path == "gemv":
        assert m <= 4, m
        _w4_gemv_kernel[(triton.cdiv(n, cfg["BN"]), split)](
            *common, KIND=kind, GROUP=group, BN=cfg["BN"], NB=cfg["NB"], SPLIT_K=split,
            MB=1 if m == 1 else 2 if m == 2 else 4, num_warps=cfg["num_warps"],
        )
    else:
        kernel = _w4_dot8_kernel if path == "dot8" else _w4_dot_kernel
        kernel[(triton.cdiv(n, cfg["BN"]), triton.cdiv(m, cfg["TB"]), split)](
            *common, KIND=kind, GROUP=group, BN=cfg["BN"], TB=cfg["TB"], NB=cfg["NB"], SPLIT_K=split,
            STAGES=cfg["STAGES"], num_warps=cfg["num_warps"],
        )
    return out[0] if split == 1 else out.sum(0).to(x.dtype)


@w4a16_linear.register_kernel("cuda")
def _(x, weight, scale, zero_point, group, kind, tinygemm_weight=None, tinygemm_scale_zero=None):
    return gemm(x, weight, scale, zero_point, group, kind, tinygemm_weight, tinygemm_scale_zero)
