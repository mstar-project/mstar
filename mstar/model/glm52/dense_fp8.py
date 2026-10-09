"""fp8 block-scale linears for GLM-5.2's non-expert weights: ``x @ (w * s).T``.

The checkpoint stores the attention projections, the shared expert and the dense-layer MLPs
as e4m3 with one fp32 scale per 128 x 128 block. Kept that way, a decode step reads half the
bytes of the bf16 copies.

Decode batches (up to ``_W8A16_MAX_TOKENS`` tokens) meet the weights as bf16 (w8a16): up to
2-4 tokens as fused_moe.decode's per-pair GEMV, fp32 FMAs on weights converted in registers with
each block's scale folded into x; above that a bf16 ``tl.dot`` per 128-column block with the
weight rows on its M side (e4m3 to bf16 is exact), scaled per block in fp32. Narrow outputs
split K across programs; the last program of an output tile to finish sums the partials in
split order, so the result does not depend on scheduling. Larger batches quantize x to e4m3
per 128 columns (W8A8): flashinfer's SM90 block-scale GEMM, then cuBLAS's for prefill.

With ``glu`` the weight is ``[gate; up]`` stacked on its rows and the output is
``silu(x @ gate.T) * (x @ up.T)``.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mstar.model.glm52.quantization import FP8_DTYPE

logger = logging.getLogger(__name__)

# Tokens up to which w8a16 is the fastest (H100): the shared expert's small GEMMs keep it
# longer. Above, W8A8: flashinfer's SM90 block-scale GEMM up to _FI_MAX_TOKENS, cuBLAS's
# block-scaled GEMM beyond.
_W8A16_MAX_TOKENS = 32
_W8A16_MAX_TOKENS_BY_SHAPE = {(256, 6144, True): 128, (6144, 256, False): 128}
_FI_MAX_TOKENS = 512
# block-scaled fp8 in torch's scaled_mm (1x128 activations, 128x128 weights), probed per device:
# the API exists from torch 2.10 but raises off SM90 or on a cuBLASLt before 12.9 (cu128 builds)
_W8A8_API = hasattr(F, "scaled_mm") and hasattr(F, "ScalingType")
_W8A8_OK: dict[torch.device, bool] = {}
_FI_OK: dict[torch.device, bool] = {}
# Split-K arrival counters, one per output tile; every launch leaves them at zero.
_COUNTERS: dict[torch.device, torch.Tensor] = {}
_MAX_TILES = 4096


@triton.jit
def _epilogue(acc, out, n0, t, live_t, N_OUT, BN: tl.constexpr, TB: tl.constexpr,
              GLU: tl.constexpr):
    """Store ``acc [rows, TB]`` (gate rows then up rows under GLU) as ``out[t, n0 + row]``."""
    n = n0 + tl.arange(0, BN)
    if GLU:
        g, u = tl.split(tl.permute(tl.reshape(acc, [2, BN, TB]), [1, 2, 0]))
        acc = g * tl.sigmoid(g) * u
    tl.store(out + t[None, :].to(tl.int64) * N_OUT + n[:, None],
             acc.to(out.dtype.element_ty), mask=live_t[None, :] & (n[:, None] < N_OUT))


@triton.jit
def _dot_loop(x, w, s, rows, live_n, t, live_t, k_lo, n0, K: tl.constexpr, I: tl.constexpr,
              BO: tl.constexpr, BI: tl.constexpr, BN: tl.constexpr, R: tl.constexpr,
              BK: tl.constexpr, SB: tl.constexpr, TB: tl.constexpr, KS: tl.constexpr,
              GLU: tl.constexpr, STAGES: tl.constexpr):
    """``[R, TB]`` partial over K columns ``[k_lo, k_lo + KS)``: steps of BK made of BK / SB
    dots of SB columns, each inside one scale block (a half's BN rows sit in one row block,
    so one scale per half). Weight rows sit on the dot's M side and tokens on its N side
    (TB >= 16)."""
    kk = tl.arange(0, SB)
    w_row = w + rows[:, None].to(tl.int64) * K + kk[None, :]
    x_row = x + t[:, None].to(tl.int64) * K + kk[None, :]
    sg_row = s + (n0 // BO) * (K // BI)
    su_row = s + ((I + n0) // BO) * (K // BI)
    up = tl.arange(0, R) >= BN
    acc = tl.zeros([R, TB], tl.float32)
    for k0 in tl.range(k_lo, k_lo + KS, BK, num_stages=STAGES):
        for j in tl.static_range(BK // SB):
            k = k0 + j * SB
            wk = tl.load(w_row + k, mask=live_n[:, None], other=0)
            wk = wk.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            xk = tl.load(x_row + k, mask=live_t[:, None], other=0.0)
            d = tl.dot(wk, tl.trans(xk))
            sg = tl.load(sg_row + k // BI)
            if GLU:
                acc += d * tl.where(up, tl.load(su_row + k // BI), sg)[:, None]
            else:
                acc += d * sg
    return acc


@triton.jit
def _block_scales(row, k0, kk, BK: tl.constexpr, BI: tl.constexpr):
    """Each of BK columns' block scale from one scale row: BK / BI loads, selected in
    registers (a per-column gather costs a load per element)."""
    sv = tl.load(row + k0 // BI)
    for j in tl.static_range(1, BK // BI):
        sv = tl.where(kk >= j * BI, tl.load(row + k0 // BI + j), sv)
    return sv


@triton.jit
def _fma_token(acc_g, acc_u, x, t, M, k0, kk, wg, wu, sg, su, K: tl.constexpr,
               GLU: tl.constexpr):
    xv = tl.load(x + t * K + k0 + kk, mask=(kk < K) & (t < M), other=0.0).to(tl.float32)
    acc_g += wg * (xv * sg)[None, :]
    if GLU:
        acc_u += wu * (xv * su)[None, :]
    return acc_g, acc_u


@triton.jit
def _fma_out(out, part, g, u, t, M, n0, N_OUT, BN: tl.constexpr, TB: tl.constexpr,
             SPLIT: tl.constexpr, GLU: tl.constexpr):
    """Token t's result: the output, or with SPLIT its partial rows (gate, then up)."""
    r = tl.arange(0, BN)
    n = n0 + r
    gs = tl.sum(g, 1)
    if SPLIT == 1:
        if GLU:
            gs = gs * tl.sigmoid(gs) * tl.sum(u, 1)
        tl.store(out + t * N_OUT + n, gs.to(out.dtype.element_ty), mask=(n < N_OUT) & (t < M))
    else:
        tl.store(part + r * TB + t, gs, mask=t < M)
        if GLU:
            tl.store(part + (BN + r) * TB + t, tl.sum(u, 1), mask=t < M)


@triton.jit
def _fma_tile(x, w, s, out, part, n0, live, k_lo, M, N_OUT, K: tl.constexpr,
              I: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, TB: tl.constexpr, KS: tl.constexpr, SPLIT: tl.constexpr,
              GLU: tl.constexpr, STAGES: tl.constexpr):
    """``_dot_loop``'s tile for up to 4 tokens, as fused_moe.decode's per-pair GEMV: fp32 FMAs on
    the weights straight from global memory into one [BN, BK] accumulator per token (and
    per half under GLU), each column's block scale folded into x in fp32 (a half's BN rows
    sit in one row block), summed over K once at the end."""
    kk = tl.arange(0, BK)
    wg_row = w + (n0 + tl.arange(0, BN))[:, None].to(tl.int64) * K + kk[None, :]
    sg_row = s + (n0 // BO) * (K // BI)
    su_row = s + ((I + n0) // BO) * (K // BI)
    UB: tl.constexpr = BN if GLU else 1
    UK: tl.constexpr = BK if GLU else 1
    g0 = tl.zeros([BN, BK], tl.float32)
    g1 = tl.zeros([BN if TB > 1 else 1, BK if TB > 1 else 1], tl.float32)
    g2 = tl.zeros([BN if TB > 2 else 1, BK if TB > 2 else 1], tl.float32)
    g3 = tl.zeros([BN if TB > 3 else 1, BK if TB > 3 else 1], tl.float32)
    u0 = tl.zeros([UB, UK], tl.float32)
    u1 = tl.zeros([UB if TB > 1 else 1, UK if TB > 1 else 1], tl.float32)
    u2 = tl.zeros([UB if TB > 2 else 1, UK if TB > 2 else 1], tl.float32)
    u3 = tl.zeros([UB if TB > 3 else 1, UK if TB > 3 else 1], tl.float32)
    for k0 in tl.range(k_lo, k_lo + KS, BK, num_stages=STAGES):
        wg = tl.load(wg_row + k0, mask=live[:, None], other=0)
        wg = wg.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sg = _block_scales(sg_row, k0, kk, BK, BI)
        wu = wg
        su = sg
        if GLU:
            wu = tl.load(wg_row + I * K + k0, mask=live[:, None], other=0)
            wu = wu.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            su = _block_scales(su_row, k0, kk, BK, BI)
        g0, u0 = _fma_token(g0, u0, x, 0, M, k0, kk, wg, wu, sg, su, K, GLU)
        if TB > 1:
            g1, u1 = _fma_token(g1, u1, x, 1, M, k0, kk, wg, wu, sg, su, K, GLU)
        if TB > 2:
            g2, u2 = _fma_token(g2, u2, x, 2, M, k0, kk, wg, wu, sg, su, K, GLU)
            g3, u3 = _fma_token(g3, u3, x, 3, M, k0, kk, wg, wu, sg, su, K, GLU)
    _fma_out(out, part, g0, u0, 0, M, n0, N_OUT, BN, TB, SPLIT, GLU)
    if TB > 1:
        _fma_out(out, part, g1, u1, 1, M, n0, N_OUT, BN, TB, SPLIT, GLU)
    if TB > 2:
        _fma_out(out, part, g2, u2, 2, M, n0, N_OUT, BN, TB, SPLIT, GLU)
        _fma_out(out, part, g3, u3, 3, M, n0, N_OUT, BN, TB, SPLIT, GLU)


@triton.jit
def _w8a16_kernel(x, w, s, out, part, cnt, M, N_OUT,
                  K: tl.constexpr, I: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr, SB: tl.constexpr, TB: tl.constexpr,
                  SPLIT: tl.constexpr, GLU: tl.constexpr, DOT: tl.constexpr,
                  STAGES: tl.constexpr):
    """Program (output tile, K split, token tile): BN output columns (and their up rows under
    GLU) for TB tokens over K / SPLIT."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    R: tl.constexpr = 2 * BN if GLU else BN
    r = tl.arange(0, R)
    n0 = pid_n * BN
    live_n = (n0 + r % BN) < N_OUT
    t = pid_m * TB + tl.arange(0, TB)
    live_t = t < M
    KS: tl.constexpr = K // SPLIT
    out_tile = pid_m * tl.num_programs(0) + pid_n
    tile = (part + out_tile.to(tl.int64) * (SPLIT * R * TB) + r[:, None] * TB
            + tl.arange(0, TB)[None, :])
    if DOT:
        if GLU:
            rows = tl.where(r < BN, n0 + r, I + n0 + r - BN)
        else:
            rows = n0 + r
        acc = _dot_loop(x, w, s, rows, live_n, t, live_t, pid_k * KS, n0, K, I, BO, BI, BN, R,
                        BK, SB, TB, KS, GLU, STAGES)
        if SPLIT == 1:
            _epilogue(acc, out, n0, t, live_t, N_OUT, BN, TB, GLU)
        else:
            tl.store(tile + pid_k * (R * TB), acc, mask=live_t[None, :])
    else:
        _fma_tile(x, w, s, out,
                  part + (out_tile.to(tl.int64) * SPLIT + pid_k) * (R * TB), n0,
                  (n0 + tl.arange(0, BN)) < N_OUT, pid_k * KS, M, N_OUT, K, I, BO, BI, BN,
                  BK, TB, KS, SPLIT, GLU, STAGES)
    if SPLIT > 1:
        tl.debug_barrier()
        if tl.atomic_add(cnt + out_tile, 1, sem="acq_rel") == SPLIT - 1:
            acc = tl.zeros([R, TB], tl.float32)
            for k in tl.static_range(SPLIT):
                acc += tl.load(tile + k * (R * TB), mask=live_t[None, :], other=0.0,
                               cache_modifier=".cg")
            _epilogue(acc, out, n0, t, live_t, N_OUT, BN, TB, GLU)
            tl.atomic_xchg(cnt + out_tile, 0)


def reserve(device: torch.device | str) -> None:
    """Allocate the split-K counters for ``device`` and load flashinfer's GEMM, outside any
    graph capture."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda" and device not in _COUNTERS:
        _COUNTERS[device] = torch.zeros(_MAX_TILES, dtype=torch.int32, device=device)
        _FI_OK[device] = _probe_fi(device)
        _W8A8_OK[device] = _probe_w8a8(device)


def _probe_w8a8(device: torch.device) -> bool:
    """cuBLAS's block-scaled GEMM runs here; else the large tiles take the Triton w8a16."""
    if not _W8A8_API:
        return False
    try:
        x = torch.zeros(4, 128, dtype=torch.bfloat16, device=device)
        _w8a8(x, torch.zeros(128, 128, dtype=torch.uint8, device=device),
              torch.ones(1, 1, device=device), (128, 128), False)
        return True
    except Exception as exc:  # noqa: BLE001 - NotImplementedError off SM90 / old cuBLASLt
        logger.warning("cuBLAS block-scaled fp8 GEMM unavailable (%r); dense fp8 above %d "
                       "tokens takes the w8a16 kernel", exc, _FI_MAX_TOKENS)
        return False


def _probe_fi(device: torch.device) -> bool:
    """flashinfer's SM90 block-scale GEMM runs here (and is loaded before any capture)."""
    if torch.cuda.get_device_capability(device) != (9, 0):
        return False
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        w = torch.zeros(128, 128, dtype=FP8_DTYPE, device=device)
        fp8_blockscale_gemm_sm90(torch.zeros(64, 128, dtype=torch.bfloat16, device=device), w,
                                 None, torch.ones(1, 1, device=device))
        return True
    except Exception:  # noqa: BLE001 — not built for this GPU / version: cuBLAS instead
        return False


def _counters(device: torch.device) -> torch.Tensor:
    if device not in _COUNTERS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "dense_fp8 split-K counters must be reserved before graph capture "
                "(dense_fp8.reserve at load)")
        reserve(device)
    return _COUNTERS[device]


# One TP8 rank's linears, (output columns, K, glu) -> tokens bucket -> (DOT, BN, BK, SPLIT,
# STAGES, warps[, TB]), the fastest of a sweep on H100 (cold graph replay).
_FMA, _DOT = False, True
_TILES = {
    (2624, 6144, False): {  # q_a + kv_a, replicated
        1: (_FMA, 8, 512, 1, 4, 4), 2: (_FMA, 8, 512, 1, 4, 2), 4: (_DOT, 32, 128, 8, 3, 2),
        16: (_DOT, 64, 128, 16, 2, 4), 32: (_DOT, 64, 128, 8, 3, 4),
        64: (_DOT, 64, 128, 8, 2, 4)},
    (2048, 2048, False): {  # q_b
        1: (_FMA, 8, 512, 1, 4, 2), 2: (_FMA, 8, 512, 1, 4, 2), 4: (_FMA, 8, 256, 1, 4, 2),
        16: (_DOT, 32, 128, 4, 3, 2), 32: (_DOT, 64, 128, 4, 3, 4),
        64: (_DOT, 64, 128, 4, 3, 4, 32)},
    (6144, 2048, False): {  # o_proj
        1: (_FMA, 8, 512, 1, 3, 2), 2: (_FMA, 8, 256, 1, 4, 2), 4: (_DOT, 64, 128, 4, 3, 4),
        16: (_DOT, 64, 128, 8, 2, 2), 32: (_DOT, 64, 128, 4, 3, 4),
        64: (_DOT, 64, 128, 4, 3, 4)},
    (256, 6144, True): {  # shared expert gate/up
        1: (_FMA, 8, 256, 8, 3, 2), 2: (_FMA, 8, 256, 8, 4, 2), 4: (_DOT, 16, 128, 16, 2, 2),
        16: (_DOT, 16, 128, 16, 3, 4), 32: (_DOT, 16, 128, 8, 3, 4),
        64: (_DOT, 32, 128, 16, 3, 4, 32)},
    (6144, 256, False): {  # shared expert down
        1: (_FMA, 8, 256, 1, 2, 2), 2: (_FMA, 16, 128, 1, 3, 2), 4: (_DOT, 32, 128, 1, 5, 2),
        16: (_DOT, 32, 128, 1, 4, 2), 32: (_DOT, 64, 256, 1, 3, 4),
        64: (_DOT, 64, 128, 1, 3, 4)},
    (1536, 6144, True): {  # dense-layer gate/up
        1: (_FMA, 8, 512, 1, 5, 2), 2: (_FMA, 8, 256, 1, 5, 2), 4: (_DOT, 32, 128, 8, 3, 4),
        16: (_DOT, 32, 128, 8, 3, 4), 32: (_DOT, 32, 128, 8, 3, 4),
        64: (_DOT, 32, 128, 8, 2, 4)},
    (6144, 1536, False): {  # dense-layer down
        1: (_FMA, 8, 512, 1, 3, 2), 2: (_FMA, 16, 256, 1, 4, 2), 4: (_DOT, 64, 128, 4, 3, 4),
        16: (_DOT, 64, 128, 4, 3, 4), 32: (_DOT, 64, 128, 4, 3, 4),
        64: (_DOT, 64, 128, 4, 3, 4)},
}


def _w8a16_tiles(M: int, N: int, K: int, glu: bool, block_size=(128, 128)) -> dict:
    """The swept config for GLM-5.2's shapes; otherwise FMA up to 2 tokens and dot above,
    splitting K until about 128 programs cover the SMs."""
    bucket = next(b for b in (1, 2, 4, 16, 32, 64) if M <= b or b == 64)
    tiled = _TILES.get((N, K, glu), {}).get(bucket) if tuple(block_size) == (128, 128) else None
    if tiled is not None:
        dot, bn, bk, split, stages, warps, *tb = tiled
        return dict(DOT=dot, BN=bn, BK=bk, SPLIT=split, STAGES=stages, num_warps=warps,
                    **(dict(TB=tb[0]) if tb else {}))
    bk = 128
    while K % bk:
        bk //= 2
    dot = M > 2 and bk >= 16
    bn = min(32 if dot else 8, block_size[0])
    tiles = triton.cdiv(N, bn)
    split = 1
    while tiles * split < 128 and K % (2 * split * bk) == 0 and K // (2 * split) >= 512:
        split *= 2
    return dict(DOT=dot, BN=bn, BK=bk, SPLIT=split, STAGES=3, num_warps=4)


def _w8a16(x, w, s, block_size, glu, cfg=None):
    M, K = x.shape
    N = w.shape[0] // 2 if glu else w.shape[0]
    bo, bi = block_size
    cfg = dict(cfg or _w8a16_tiles(M, N, K, glu, block_size))
    bn, bk, split, dot = cfg.pop("BN"), cfg.pop("BK"), cfg.pop("SPLIT"), cfg.pop("DOT", True)
    sb = min(bk, bi)  # each dot stays inside one scale block
    assert bk % sb == 0 and bi % sb == 0 and K % (split * bk) == 0 and bo % bn == 0
    assert not glu or N % bn == 0
    if dot:  # token tiles of up to 64
        tb = cfg.pop("TB", min(64, max(16, triton.next_power_of_2(M))))
    else:  # the FMA tile keeps one accumulator per token: up to 4
        tb = triton.next_power_of_2(M)
        assert tb <= 4 and (bk % bi == 0 or bi % bk == 0)
    tiles, m_tiles = triton.cdiv(N, bn), triton.cdiv(M, tb)
    if tiles * m_tiles > _MAX_TILES:  # enough programs without splitting K
        split = 1
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    rows = 2 * bn if glu else bn
    if split > 1:
        part = torch.empty(tiles * m_tiles * split * rows * tb, dtype=torch.float32,
                           device=x.device)
        cnt = _counters(x.device)
    else:
        part = cnt = out  # unused
    _w8a16_kernel[(tiles, split, m_tiles)](
        x, w, s, out, part, cnt, M, N, K=K, I=N if glu else 0, BO=bo, BI=bi, BN=bn, BK=bk,
        SB=sb, TB=tb, SPLIT=split, GLU=glu, DOT=dot, **cfg)
    return out


def _w8a8(x, w, s, block_size, glu):
    """cuBLAS's block-scaled fp8 GEMM: x group-quantized to e4m3 per 128 columns
    (fused_moe.prefill's kernel), the weight's 128 x 128 scales as they are. cuBLAS wants the
    activation scales token-major with the token count a multiple of 4 (x is zero-padded) and
    the weight's K blocks padded to a multiple of 4."""
    from mstar.utils.fused_moe.kernels import act_and_mul_triton
    from mstar.utils.fused_moe.prefill import _group_quant

    M, K = x.shape
    kb = K // block_size[1]
    xp = F.pad(x, (0, 0, 0, -M % 4)) if M % 4 else x
    a, a_scale = _group_quant(xp, block_size[1])
    s = F.pad(s, (0, -kb % 4)) if kb % 4 else s
    h = F.scaled_mm(a, w.view(FP8_DTYPE).t(), a_scale.t().contiguous().t(),
                    F.ScalingType.BlockWise1x128, s.t(), F.ScalingType.BlockWise128x128,
                    output_dtype=x.dtype)[:M]
    if not glu:
        return h
    out = torch.empty(M, h.shape[1] // 2, dtype=x.dtype, device=x.device)
    act_and_mul_triton(h, out, activation="silu")
    return out


def _fi(x, w, s, glu):
    """flashinfer's SM90 block-scale GEMM; it group-quantizes x per 128 columns itself."""
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    h = fp8_blockscale_gemm_sm90(x, w.view(FP8_DTYPE), None, s)
    if not glu:
        return h
    out = torch.empty(h.shape[0], h.shape[1] // 2, dtype=x.dtype, device=x.device)
    act_and_mul_triton(h, out, activation="silu")
    return out


def _linear(x, w, s, block_size, glu):
    x = x.contiguous()
    assert x.dtype == torch.bfloat16 and w.dtype in (torch.uint8, FP8_DTYPE)
    assert w.is_contiguous() and s.is_contiguous() and s.dtype == torch.float32
    M, K = x.shape
    N = w.shape[0] // 2 if glu else w.shape[0]
    if M == 0:
        return x.new_empty(0, N)
    w8a16_max = _W8A16_MAX_TOKENS_BY_SHAPE.get((N, K, glu), _W8A16_MAX_TOKENS)
    if M <= w8a16_max or tuple(block_size) != (128, 128):
        return _w8a16(x, w, s, block_size, glu)
    if M <= _FI_MAX_TOKENS and _FI_OK.get(x.device):
        return _fi(x, w, s, glu)
    if _W8A8_OK.get(x.device):
        return _w8a8(x, w, s, block_size, glu)
    return _w8a16(x, w, s, block_size, glu)


@torch.library.custom_op("glm52::fp8_linear", mutates_args=())
def fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, block_n: int,
               block_k: int, glu: bool) -> torch.Tensor:
    """``(M, K) -> (M, N)``; an opaque op to dynamo, so compiled graphs keep it whole."""
    return _linear(x, weight, scale, (block_n, block_k), glu)


@fp8_linear.register_fake
def _(x, weight, scale, block_n, block_k, glu):
    return x.new_empty(x.shape[0], weight.shape[0] // 2 if glu else weight.shape[0])


def linear(x, weight, scale, block_size, glu=False):
    """``x (..., K) @ (weight * scale).T`` (or its SwiGLU with ``glu``) on CUDA, in x.dtype."""
    lead = x.shape[:-1]
    y = fp8_linear(x.reshape(-1, x.shape[-1]), weight, scale, block_size[0], block_size[1], glu)
    return y.view(*lead, y.shape[-1])
