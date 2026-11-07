# SPDX-License-Identifier: Apache-2.0
# The int4 decode follows the int4 (``use_int4_w4a16``) branch of SGLang's / vLLM's ``fused_moe_kernel_gptq_awq`` and
# vLLM's ``awq_triton.py`` (Apache-2.0, Copyright the vLLM and SGLang authors). Changed here: the compressed-tensors
# layout read in its own orientation, weight rows on the MMA's M side, one dot per scale group, a CUDA-core GEMV for
# a few tokens, and an e2m1 + e8m0 (MXFP4) decode beside the int4 one.
"""The CUDA kernels of ``mstar::w4a16_linear`` (``model.components.int4_formats``): ``x @ W.T`` for a 4-bit ``W``.

``W`` is compressed-tensors' ``[N, K/8]`` int32 words (an MXFP4 weight's ``[N, K/2]`` bytes read the same way), and
every kernel loads a ``[rows, words]`` block along K, the way it lies in memory. An int4 code becomes ``code - zero``
in fp32 without a conversion instruction (``code | 0x4B000000`` is the float ``2**23 + code``); scales apply once per
word or per group, not per weight. Paths:

- ``tinygemm`` (1-2 tokens): torch's int4 GEMM (``aten._weight_int4pack_mm``) on its own copy of the weight, which
  ``int4_formats`` builds at load for a symmetric or zero-point int4 weight in groups of 32-256;
- ``gemv`` (a few tokens): CUDA-core FMAs over a word's 8 nibbles in turn (no unpacking reshape), each weight decoded
  once for all tokens, reduced once at the end; split along K;
- ``dot`` / ``dot8``: weight rows on the MMA's M side and tokens (padded to 16) on its N side; ``dot`` unpacks a
  block and runs one dot per scale group, ``dot8`` runs one dot per nibble position (K in another order, which a sum
  doesn't mind); split along K;
- ``dequant``: past ``DEQUANT_M`` tokens, the weight dequantized to the activation's dtype and cuBLAS runs it.

Which path and tile runs at a token count is ``config``'s built-in rule, set from ``bench_w4a16``'s H200 sweep at
Qwen3.5-27B's shapes (``res/w4r2``, job 1479); a hypothesis for other GPUs and shapes. Every path matches the
reference op (``int4_formats.dequantize_kind`` then a GEMM) to fp32 accumulation order.
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
def _w4_dot_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr, M, N, K,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_os, stride_om,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, TB: tl.constexpr, BK: tl.constexpr,
    SB: tl.constexpr, SPLIT_K: tl.constexpr, STAGES: tl.constexpr,
):
    """``out[pid_k, t, n]`` for BN weight rows x TB tokens: steps of BK columns made of BK / SB dots, one scale group
    each (SB = min(BK, GROUP)); split ``pid_k`` takes steps pid_k, pid_k + SPLIT_K, ..."""
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)
    n = pid_n * BN + tl.arange(0, BN)
    live_n = n < N
    t = pid_t * TB + tl.arange(0, TB)
    live_t = t < M
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_wn
    x_rows = x_ptr + t[:, None].to(tl.int64) * stride_xm
    sub = tl.arange(0, SB)
    acc = tl.zeros((BN, TB), dtype=tl.float32)
    for k0 in tl.range(pid_k * BK, K, SPLIT_K * BK, num_stages=STAGES):
        for j in tl.static_range(BK // SB):
            k = k0 + j * SB
            live_k = k < K
            words = tl.load(w_rows + (k // 8 + tl.arange(0, SB // 8))[None, :], mask=live_n[:, None] & live_k, other=0)
            scale = _group_scale(s_ptr, n, live_n & live_k, k // GROUP, stride_sn, KIND)
            if KIND == 1:
                zf = _group_zero(z_ptr, n, live_n & live_k, k // GROUP, stride_zn)
                v = _values(_codes(words, BN, SB), zf[:, None], KIND)
            else:
                v = _values(_codes(words, BN, SB), 8388616.0, KIND)
            xk = tl.load(x_rows + (k + sub)[None, :], mask=live_t[:, None] & live_k, other=0.0)
            acc += tl.dot(v.to(xk.dtype), tl.trans(xk)) * scale[:, None]
    out = out_ptr + pid_k * stride_os + t[None, :].to(tl.int64) * stride_om + n[:, None]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=live_n[:, None] & live_t[None, :])


@triton.jit
def _w4_dot8_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr, M, N, K,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_os, stride_om,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, TB: tl.constexpr, BW: tl.constexpr,
    SPLIT_K: tl.constexpr, STAGES: tl.constexpr,
):
    """``dot`` with one dot per nibble position: a step of BW words (8 BW columns) is 8 dots of K = BW, nibble j of
    every word against columns 8i + j of x. With a group at least the step, the scale applies to the step's partial;
    with smaller groups, to each weight before its dot."""
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)
    n = pid_n * BN + tl.arange(0, BN)
    live_n = n < N
    t = pid_t * TB + tl.arange(0, TB)
    live_t = t < M
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_wn
    x_rows = x_ptr + t[:, None].to(tl.int64) * stride_xm
    wi = tl.arange(0, BW)
    acc = tl.zeros((BN, TB), dtype=tl.float32)
    for k0 in tl.range(pid_k * BW * 8, K, SPLIT_K * BW * 8, num_stages=STAGES):
        kw = k0 + 8 * wi
        live_w = live_n[:, None] & (kw < K)[None, :]
        words = tl.load(w_rows + (k0 // 8 + wi)[None, :], mask=live_w, other=0)
        if GROUP >= 8 * BW:
            scale = _group_scale(s_ptr, n, live_n & (k0 < K), k0 // GROUP, stride_sn, KIND)
            if KIND == 1:
                zf = _group_zero(z_ptr, n, live_n & (k0 < K), k0 // GROUP, stride_zn)[:, None]
            else:
                zf = 8388616.0
        else:
            scale = _group_scale(s_ptr, n[:, None], live_w, (kw // GROUP)[None, :], stride_sn, KIND)
            if KIND == 1:
                zf = _group_zero(z_ptr, n[:, None], live_w, (kw // GROUP)[None, :], stride_zn)
            else:
                zf = 8388616.0
        part = tl.zeros((BN, TB), dtype=tl.float32)
        for j in tl.static_range(8):
            v = _values((words >> (4 * j)) & 0xF, zf, KIND)
            if GROUP < 8 * BW:
                v = v * scale
            xj = tl.load(x_rows + (kw + j)[None, :], mask=live_t[:, None] & (kw < K)[None, :], other=0.0)
            part += tl.dot(v.to(xj.dtype), tl.trans(xj))
        if GROUP >= 8 * BW:
            acc += part * scale[:, None]
        else:
            acc += part
    out = out_ptr + pid_k * stride_os + t[None, :].to(tl.int64) * stride_om + n[:, None]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=live_n[:, None] & live_t[None, :])


@triton.jit
def _w4_gemv_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr, M, N, K,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_os, stride_om,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, BW: tl.constexpr, SPLIT_K: tl.constexpr,
    MB: tl.constexpr,
):
    """``out[pid_k, m, n]`` for BN weight rows and MB (>= M) tokens on CUDA cores. A step loads BW words a row and
    takes their 8 nibbles in turn against the matching strided columns of x, summing each word's 8 products before
    its one scale multiply; the accumulator keeps the word axis and is reduced once after the loop."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    live_n = n < N
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_wn
    wi = tl.arange(0, BW)
    rows_m = tl.arange(0, MB)
    x_rows = x_ptr + rows_m[:, None] * stride_xm
    live_m = rows_m[:, None] < M
    acc = tl.zeros((MB, BN, BW), dtype=tl.float32)
    for k0 in range(pid_k * BW * 8, K, SPLIT_K * BW * 8):
        kw = k0 + 8 * wi
        live_w = live_n[:, None] & (kw < K)[None, :]
        words = tl.load(w_rows + (k0 // 8 + wi)[None, :], mask=live_w, other=0)
        scale = _group_scale(s_ptr, n[:, None], live_w, (kw // GROUP)[None, :], stride_sn, KIND)
        if KIND == 1:
            zf = _group_zero(z_ptr, n[:, None], live_w, (kw // GROUP)[None, :], stride_zn)
        else:
            zf = 8388616.0
        part = tl.zeros((MB, BN, BW), dtype=tl.float32)
        for j in tl.static_range(8):
            v = _values((words >> (4 * j)) & 0xF, zf, KIND)
            xj = tl.load(x_rows + (kw + j)[None, :], mask=live_m & (kw < K)[None, :], other=0.0).to(tl.float32)
            part += v[None, :, :] * xj[:, None, :]
        acc += part * scale[None, :, :]
    out = out_ptr + pid_k * stride_os + rows_m[:, None] * stride_om + n[None, :]
    tl.store(out, tl.sum(acc, axis=2).to(out_ptr.dtype.element_ty), mask=live_m & live_n[None, :])


@triton.jit
def _w4_dequant_kernel(
    w_ptr, s_ptr, z_ptr, out_ptr, N, K, stride_wn, stride_sn, stride_zn, stride_on,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr,
    GS: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k0 = tl.program_id(1) * BK
    live_n = n < N
    words = tl.load(w_ptr + n[:, None].to(tl.int64) * stride_wn + (k0 // 8 + tl.arange(0, BK // 8))[None, :],
                    mask=live_n[:, None], other=0)
    groups = k0 // GROUP + tl.arange(0, NG)[None, :]
    scale = _group_scale(s_ptr, n[:, None], live_n[:, None], groups, stride_sn, KIND)
    if KIND == 1:
        zf = tl.reshape(tl.broadcast_to(_group_zero(z_ptr, n[:, None], live_n[:, None], groups, stride_zn)[:, :, None],
                                        (BN, NG, GS)), (BN, BK))
        v = _values(_codes(words, BN, BK), zf, KIND)
    else:
        v = _values(_codes(words, BN, BK), 8388616.0, KIND)
    v = tl.reshape(v, (BN, NG, GS)) * scale[:, :, None]
    tl.store(out_ptr + n[:, None].to(tl.int64) * stride_on + (k0 + tl.arange(0, BK))[None, :],
             tl.reshape(v, (BN, BK)).to(out_ptr.dtype.element_ty), mask=live_n[:, None])


@lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _block_k(group: int, k: int, want: int) -> int:
    """A K step that divides the group or is whole groups, and divides K."""
    for bk in (want, 512, 256, 128, 64, 32):
        if bk <= k and k % bk == 0 and (bk % group == 0 or group % bk == 0):
            return bk
    raise ValueError(f"no K step for K={k}, group={group}")


def _split(ctas: int, k_steps: int, sms: int, waves: int) -> int:
    return max(1, min(16, k_steps, (waves * sms) // max(ctas, 1)))


def _tb(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128


def config(m: int, n: int, k: int, group: int, kind: int, sms: int) -> tuple[str, dict]:
    """The built-in path and tile for ``m`` tokens of an ``[n, k]`` weight. From the H200 sweep at Qwen3.5-27B's
    shapes (job 1479): a GEMV step of 512 columns was 2-5x faster than 128-256 at one token; the dot path took
    64 x 16 tiles with 256-column steps for up to 16 tokens (128-row tiles, 8 warps, for the N = 5120 layers), 32 and
    64 tokens their own TB, and dequant + cuBLAS overtook every tile at 256."""
    if m >= DEQUANT_M:
        return "dequant", {}
    if m <= GEMV_MAX:
        # (a step reads each word's own group, and masks the columns past K)
        bn, bw = 32, 64 if k >= 512 else 16
        return "gemv", {"BN": bn, "BW": bw, "SPLIT_K": _split(triton.cdiv(n, bn), k // (8 * bw), sms, 4),
                        "num_warps": 4}
    tb = _tb(m)
    if tb == 128:
        bn, bk, st, w = 64, 64, 4, 4
    elif n <= 8192 and tb == 16:
        bn, bk, st, w = 128, 128, 3, 8
    elif tb == 16:
        bn, bk, st, w = 64, 256, 2, 4
    else:
        bn, bk, st, w = 64, 128, 3, 4
    bk = _block_k(group, k, bk)
    ctas = triton.cdiv(n, bn) * triton.cdiv(m, tb)
    return "dot", {"BN": bn, "TB": tb, "BK": bk, "SPLIT_K": _split(ctas, k // bk, sms, 2),
                   "STAGES": 2 if kind == KIND_E2M1 else st, "num_warps": w}


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
    """The ``[N, K]`` weight in ``dtype`` (the large-M path's scratch, and tests)."""
    words = _words(weight)
    n, k = words.shape[0], words.shape[1] * 8
    out = torch.empty((n, k), dtype=dtype, device=weight.device)
    bn, bk = 32, _block_k(group, k, 128)
    s, ssn, z, szn = _scale_zero(scale, zero_point)
    _w4_dequant_kernel[(triton.cdiv(n, bn), k // bk)](
        words, s, z, out, n, k, words.stride(0), ssn, szn, out.stride(0),
        KIND=kind, GROUP=group, BN=bn, BK=bk, NG=max(bk // group, 1), GS=min(group, bk), num_warps=4,
    )
    return out


def gemm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor | None, group: int,
         kind: int, tinygemm_weight: torch.Tensor | None = None,
         tinygemm_scale_zero: torch.Tensor | None = None) -> torch.Tensor:
    words = _words(weight)
    m, k = x.shape
    n = words.shape[0]
    assert words.shape[1] * 8 == k and words.is_contiguous(), (tuple(weight.shape), k)
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
    out = torch.empty((split, m, n), dtype=torch.float32 if split > 1 else x.dtype, device=x.device)
    s, ssn, z, szn = _scale_zero(scale, zero_point)
    common = (x, words, s, z, out, m, n, k, x.stride(0), words.stride(0), ssn, szn, out.stride(0), out.stride(1))
    if path == "gemv":
        _w4_gemv_kernel[(triton.cdiv(n, cfg["BN"]), split)](
            *common, KIND=kind, GROUP=group, BN=cfg["BN"], BW=cfg["BW"], SPLIT_K=split,
            MB=triton.next_power_of_2(m), num_warps=cfg["num_warps"],
        )
    elif path == "dot8":
        _w4_dot8_kernel[(triton.cdiv(n, cfg["BN"]), triton.cdiv(m, cfg["TB"]), split)](
            *common, KIND=kind, GROUP=group, BN=cfg["BN"], TB=cfg["TB"], BW=cfg["BW"], SPLIT_K=split,
            STAGES=cfg["STAGES"], num_warps=cfg["num_warps"],
        )
    else:
        bk = cfg["BK"]
        _w4_dot_kernel[(triton.cdiv(n, cfg["BN"]), triton.cdiv(m, cfg["TB"]), split)](
            *common, KIND=kind, GROUP=group, BN=cfg["BN"], TB=cfg["TB"], BK=bk, SB=min(bk, group), SPLIT_K=split,
            STAGES=cfg["STAGES"], num_warps=cfg["num_warps"],
        )
    return out[0] if split == 1 else out.sum(0).to(x.dtype)


@w4a16_linear.register_kernel("cuda")
def _(x, weight, scale, zero_point, group, kind, tinygemm_weight=None, tinygemm_scale_zero=None):
    return gemm(x, weight, scale, zero_point, group, kind, tinygemm_weight, tinygemm_scale_zero)
