# SPDX-License-Identifier: Apache-2.0
# The int4 decode follows the int4 (``use_int4_w4a16``) branch of SGLang's / vLLM's ``fused_moe_kernel_gptq_awq`` and
# vLLM's ``awq_triton.py`` (Apache-2.0, Copyright the vLLM and SGLang authors). Changed here: the compressed-tensors
# layout read in its own orientation, weight rows on the MMA's M side, one dot per scale group, a CUDA-core GEMV for
# a few tokens, and an e2m1 + e8m0 (MXFP4) decode beside the int4 one.
"""The CUDA kernels of ``mstar::w4a16_linear`` (``model.components.int4_formats``): ``x @ W.T`` for a 4-bit ``W``.

``W`` is compressed-tensors' ``[N, K/8]`` int32 words (an MXFP4 weight's ``[N, K/2]`` bytes read the same way), and
every kernel loads a ``[rows, cols / 8]`` block of words along K, the way they lie in memory, and unpacks it in
registers. An int4 code becomes ``code - zero`` in fp32 without a conversion instruction (``code | 0x4B000000`` is
the float ``2**23 + code``); scales apply once per group, to a partial sum, not to every weight. Three paths:

- ``gemv`` (a few tokens): CUDA-core FMAs, each weight decoded once for all tokens, split along K;
- ``dot``: weight rows on the MMA's M side and tokens (padded to 16) on its N side, one dot per scale group,
  split along K to fill the GPU;
- ``dequant``: past ``DEQUANT_M`` tokens, the weight dequantized to the activation's dtype and cuBLAS runs it.

Which path and tile runs at a token count is ``config``'s built-in rule (a hypothesis until
``benchmark/quantization/bench_w4a16.py`` sweeps it per shape on the GPU). Every path matches the reference op
(``int4_formats.dequantize_kind`` then a GEMM) to fp32 accumulation order; the scale applied to a partial sum,
rather than to each weight rounded to bf16, is the one numerical difference, and it is the more accurate one.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch
import triton
import triton.language as tl

from mstar.model.components.int4_formats import KIND_E2M1, KIND_INT4_ZP, w4a16_linear

GEMV_MAX = 4
DEQUANT_M = 512
_MAGIC = 8388608.0  # 2 ** 23


@triton.jit
def _codes(words, ROWS: tl.constexpr, COLS: tl.constexpr):
    """int32 words ``[ROWS, COLS // 8]`` -> their 4-bit codes ``[ROWS, COLS]``: word i's nibble j is column 8i + j."""
    shifts = tl.arange(0, 8) * 4
    return tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (ROWS, COLS))


@triton.jit
def _values(codes, z, KIND: tl.constexpr):
    """fp32 values of 4-bit codes: an int4 code minus its zero point ``z`` (exact), or an e2m1 code's value."""
    if KIND == 2:
        # e2m1 magnitude e:m (3 bits) as an fp32 exponent / mantissa: (e + 126) << 23 | m << 22, exact for e >= 1
        v = ((((codes & 7) << 22) + (126 << 23)) | ((codes & 8) << 28)).to(tl.float32, bitcast=True)
        small = (codes & 1).to(tl.float32) * 0.5
        small = tl.where((codes & 8) != 0, -small, small)
        return tl.where((codes & 6) == 0, small, v)
    return (codes | 0x4B000000).to(tl.float32, bitcast=True) - (8388608.0 + z)


@triton.jit
def _group_params(s_ptr, z_ptr, n, live, g, stride_sn, stride_zn, KIND: tl.constexpr):
    """Rows ``n``'s scale (fp32) and zero point at group ``g`` (``g`` a scalar or a ``[1, NG]`` row of groups)."""
    if KIND == 2:
        se = tl.load(s_ptr + n * stride_sn + g, mask=live, other=127)
        return tl.exp2(se.to(tl.float32) - 127.0), 0
    scale = tl.load(s_ptr + n * stride_sn + g, mask=live, other=0.0).to(tl.float32)
    if KIND == 1:
        zw = tl.load(z_ptr + (n // 8) * stride_zn + g, mask=live, other=0)
        return scale, (zw >> ((n % 8) * 4)) & 0xF
    return scale, 8


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
            scale, z = _group_params(s_ptr, z_ptr, n, live_n & live_k, k // GROUP, stride_sn, stride_zn, KIND)
            if KIND == 1:
                z = z[:, None]
            v = _values(_codes(words, BN, SB), z, KIND)
            xk = tl.load(x_rows + (k + sub)[None, :], mask=live_t[:, None] & live_k, other=0.0)
            acc += tl.dot(v.to(xk.dtype), tl.trans(xk)) * scale[:, None]
    out = out_ptr + pid_k * stride_os + t[None, :].to(tl.int64) * stride_om + n[:, None]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=live_n[:, None] & live_t[None, :])


@triton.jit
def _w4_gemv_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr, M, N, K,
    stride_xm, stride_wn, stride_sn, stride_zn, stride_os, stride_om,
    KIND: tl.constexpr, GROUP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NG: tl.constexpr,
    GS: tl.constexpr, SPLIT_K: tl.constexpr, MB: tl.constexpr,
):
    """``out[pid_k, m, n]`` for BN weight rows and MB (>= M) tokens on CUDA cores: a ``[BN, BK]`` block is decoded once
    and multiplied into every token; its NG groups of GS columns are summed first and scaled once."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    live_n = n < N
    w_rows = w_ptr + n[:, None].to(tl.int64) * stride_wn
    rows_m = tl.arange(0, MB)
    cols = tl.arange(0, BK)
    groups = tl.arange(0, NG)[None, :]
    acc = tl.zeros((MB, BN), dtype=tl.float32)
    for k0 in range(pid_k * BK, K, SPLIT_K * BK):
        words = tl.load(w_rows + (k0 // 8 + tl.arange(0, BK // 8))[None, :], mask=live_n[:, None], other=0)
        scale, z = _group_params(s_ptr, z_ptr, n[:, None], live_n[:, None], k0 // GROUP + groups, stride_sn,
                                 stride_zn, KIND)
        if KIND == 1:
            z = tl.reshape(tl.broadcast_to(z[:, :, None], (BN, NG, GS)), (BN, BK))
        v = tl.reshape(_values(_codes(words, BN, BK), z, KIND), (BN, NG, GS))
        for mi in tl.static_range(MB):
            xm = tl.load(x_ptr + mi * stride_xm + k0 + cols, mask=mi < M, other=0.0).to(tl.float32)
            part = tl.sum(tl.sum(v * tl.reshape(xm, (1, NG, GS)), axis=2) * scale, axis=1)
            acc += tl.where(rows_m[:, None] == mi, part[None, :], 0.0)
    out = out_ptr + pid_k * stride_os + rows_m[:, None] * stride_om + n[None, :]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=(rows_m[:, None] < M) & live_n[None, :])


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
    scale, z = _group_params(s_ptr, z_ptr, n[:, None], live_n[:, None], k0 // GROUP + tl.arange(0, NG)[None, :],
                             stride_sn, stride_zn, KIND)
    if KIND == 1:
        z = tl.reshape(tl.broadcast_to(z[:, :, None], (BN, NG, GS)), (BN, BK))
    v = tl.reshape(_values(_codes(words, BN, BK), z, KIND), (BN, NG, GS)) * scale[:, :, None]
    tl.store(out_ptr + n[:, None].to(tl.int64) * stride_on + (k0 + tl.arange(0, BK))[None, :],
             tl.reshape(v, (BN, BK)).to(out_ptr.dtype.element_ty), mask=live_n[:, None])


@lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _block_k(group: int, k: int, want: int) -> int:
    """A K step that divides the group or is whole groups, and divides K."""
    for bk in (want, 256, 128, 64, 32):
        if bk <= k and k % bk == 0 and (bk % group == 0 or group % bk == 0):
            return bk
    raise ValueError(f"no K step for K={k}, group={group}")


def _split(ctas: int, k_steps: int, sms: int, waves: int) -> int:
    return max(1, min(16, k_steps, (waves * sms) // max(ctas, 1)))


def config(m: int, n: int, k: int, group: int, kind: int, sms: int) -> tuple[str, dict]:
    """The built-in path and tile for ``m`` tokens of an ``[n, k]`` weight (see the module docstring)."""
    if m >= DEQUANT_M:
        return "dequant", {}
    if m <= GEMV_MAX:
        bn = 32
        bk = _block_k(group, k, 256)
        return "gemv", {"BN": bn, "BK": bk, "SPLIT_K": _split(triton.cdiv(n, bn), k // bk, sms, 4),
                        "num_warps": 4}
    tb = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    bn = 64 if tb <= 64 else 128
    bk = _block_k(group, k, 128)
    ctas = triton.cdiv(n, bn) * triton.cdiv(m, tb)
    return "dot", {"BN": bn, "TB": tb, "BK": bk, "SPLIT_K": _split(ctas, k // bk, sms, 2),
                   "STAGES": 2 if kind == KIND_E2M1 else 3, "num_warps": 8 if tb == 128 else 4}


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
         kind: int) -> torch.Tensor:
    words = _words(weight)
    m, k = x.shape
    n = words.shape[0]
    assert words.shape[1] * 8 == k and words.is_contiguous(), (tuple(weight.shape), k)
    assert kind != KIND_INT4_ZP or zero_point is not None
    path, cfg = (CONFIG_OVERRIDE or config)(m, n, k, group, kind, _sm_count(x.device.index or 0))
    if path == "dequant":
        return torch.nn.functional.linear(x, dequantize(weight, scale, zero_point, group, kind, x.dtype))
    split = cfg["SPLIT_K"]
    out = torch.empty((split, m, n), dtype=torch.float32 if split > 1 else x.dtype, device=x.device)
    s, ssn, z, szn = _scale_zero(scale, zero_point)
    common = (x, words, s, z, out, m, n, k, x.stride(0), words.stride(0), ssn, szn, out.stride(0), out.stride(1))
    if path == "gemv":
        bk = cfg["BK"]
        mb = 1 if m == 1 else 2 if m == 2 else 4 if m <= 4 else triton.next_power_of_2(m)
        _w4_gemv_kernel[(triton.cdiv(n, cfg["BN"]), split)](
            *common, KIND=kind, GROUP=group, BN=cfg["BN"], BK=bk, NG=max(bk // group, 1), GS=min(group, bk),
            SPLIT_K=split, MB=mb, num_warps=cfg["num_warps"],
        )
    else:
        bk = cfg["BK"]
        _w4_dot_kernel[(triton.cdiv(n, cfg["BN"]), triton.cdiv(m, cfg["TB"]), split)](
            *common, KIND=kind, GROUP=group, BN=cfg["BN"], TB=cfg["TB"], BK=bk, SB=min(bk, group), SPLIT_K=split,
            STAGES=cfg["STAGES"], num_warps=cfg["num_warps"],
        )
    return out[0] if split == 1 else out.sum(0).to(x.dtype)


@w4a16_linear.register_kernel("cuda")
def _(x, weight, scale, zero_point, group, kind):
    return gemm(x, weight, scale, zero_point, group, kind)
