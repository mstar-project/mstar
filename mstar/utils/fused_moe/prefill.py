"""Prefill-size path of a block-fp8 MoE: ``fused_experts_fp8``'s kernels with tiles for one TP
rank at hundreds to thousands of tokens, the SwiGLU fused into the activation quant, and a
one-pass down GEMM at two quant groups.

The tile rows stay the runner's, which keeps its bits: other rows change the fp8 dot's
summation order.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import (
    _grid_rows,
    invoke_fused_moe_kernel_fp8_w8a8,
    moe_sum_reduce_triton,
)
from mstar.utils.quant_fp8 import FP8_DTYPE


@triton.jit
def _down_kernel(a, a_scale, w, w_scale, topk_w, sorted_ids, expert_ids, padded, out,
                 num_pairs, stride_we, stride_wn, stride_se, stride_sn,
                 N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GN: tl.constexpr, SPLIT: tl.constexpr, STAGES: tl.constexpr):
    """Rows ``pid(0)`` of the expert-sorted pairs times one expert's ``w [N, 2 BK]`` over
    columns ``pid(1)`` of SPLIT, scaled by the routing weight: the shared kernel's sum
    (per K group: dot * a scale * w scale), in the same order."""
    pid_m = tl.program_id(0)
    if pid_m * BM >= tl.load(padded):
        return
    pair = tl.load(sorted_ids + pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    live = pair < num_pairs  # padding slots hold num_pairs
    e = tl.load(expert_ids + pid_m).to(tl.int64)
    k = tl.arange(0, BK)
    a0 = tl.load(a + pair[:, None] * (2 * BK) + k[None, :], mask=live[:, None], other=0.0)
    a1 = tl.load(a + pair[:, None] * (2 * BK) + BK + k[None, :], mask=live[:, None], other=0.0)
    as0 = tl.load(a_scale + pair * 2, mask=live, other=0.0)
    as1 = tl.load(a_scale + pair * 2 + 1, mask=live, other=0.0)
    rw = tl.load(topk_w + pair, mask=live, other=0.0)
    wb, sb = w + e * stride_we, w_scale + e * stride_se
    span = N // SPLIT
    for n0 in tl.range(tl.program_id(1) * span, (tl.program_id(1) + 1) * span, BN,
                       num_stages=STAGES):
        n = n0 + tl.arange(0, BN)
        b0 = tl.load(wb + n[None, :] * stride_wn + k[:, None])
        b1 = tl.load(wb + n[None, :] * stride_wn + BK + k[:, None])
        bs0 = tl.load(sb + (n // GN) * stride_sn)
        bs1 = tl.load(sb + (n // GN) * stride_sn + 1)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        acc += tl.dot(a0, b0) * as0[:, None] * bs0[None, :]
        acc += tl.dot(a1, b1) * as1[:, None] * bs1[None, :]
        acc = acc * rw[:, None]
        tl.store(out + pair[:, None] * N + n[None, :], acc.to(tl.bfloat16), mask=live[:, None])


def _down(a, a_scale, w2, s2, topk_weights, ids, sorted_ids, expert_ids, padded, rows, out,
          block_size):
    """``out (pairs, N)`` = the weighted down projection of every routed pair. At two K steps a
    runner program is mostly setup; here one program loads an expert tile's activations once
    and walks a quarter of the output columns, so the weight loads pipeline across N tiles."""
    num_experts, n, _ = w2.shape
    assert a.is_contiguous() and a_scale.is_contiguous() and s2.stride(2) == 1
    em = _grid_rows(sorted_ids, ids, rows)
    _down_kernel[(triton.cdiv(em, rows), 4)](
        a, a_scale, w2, s2, topk_weights, sorted_ids, expert_ids, padded, out, ids.numel(),
        w2.stride(0), w2.stride(1), s2.stride(0), s2.stride(1),
        N=n, BM=rows, BN=64, BK=block_size[1], GN=block_size[0], SPLIT=4, STAGES=3,
        num_warps=4,
    )


@triton.jit
def _quantize(y, q, s, rows, live, r, col, eps, fp8_min, fp8_max, G: tl.constexpr,
              NG: tl.constexpr, NGP: tl.constexpr):
    """``per_token_group_quant_fp8``'s math on ``y [BR, NGP, G]`` fp32: one scale per group
    of G, e4m3 values into ``q (rows, NG * G)`` and scales into ``s (rows, NG)``."""
    y_s = tl.maximum(tl.max(tl.abs(y), 2), eps) / fp8_max
    y_q = tl.minimum(tl.maximum(y / y_s[:, :, None], fp8_min), fp8_max)
    tl.store(q + r.to(tl.int64)[:, None, None] * (NG * G) + col, y_q.to(q.dtype.element_ty),
             mask=live)
    g = tl.arange(0, NGP)[None, :]
    tl.store(s + r.to(tl.int64)[:, None] * NG + g, y_s, mask=(r < rows)[:, None] & (g < NG))


@triton.jit
def _groups(rows, BR: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr):
    """This program's BR rows, the column of each [row, group, element] and its validity."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    g = tl.arange(0, NGP)[None, :, None]
    col = g * G + tl.arange(0, G)[None, None, :]
    return r, col, (r < rows)[:, None, None] & (g < NG)


@triton.jit
def _quant_kernel(x, q, s, rows, eps, fp8_min, fp8_max,
                  G: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr, BR: tl.constexpr):
    """BR rows of ``x (rows, NG * G)`` per program (the shared kernel takes one group)."""
    r, col, live = _groups(rows, BR, G, NG, NGP)
    y = tl.load(x + r.to(tl.int64)[:, None, None] * (NG * G) + col, mask=live,
                other=0.0).to(tl.float32)
    _quantize(y, q, s, rows, live, r, col, eps, fp8_min, fp8_max, G, NG, NGP)


@triton.jit
def _act_quant_kernel(h1, q, s, rows, limit, eps, fp8_min, fp8_max,
                      G: tl.constexpr, NG: tl.constexpr, NGP: tl.constexpr, BR: tl.constexpr,
                      CLAMP: tl.constexpr):
    """``act_and_mul`` (SwiGLU, clamped with CLAMP, rounded to bf16) of BR rows of
    ``h1 (rows, 2 I)``, then quantized: the shared act kernel's expressions, without the bf16
    round trip to memory."""
    r, col, live = _groups(rows, BR, G, NG, NGP)
    row = h1 + r.to(tl.int64)[:, None, None] * (2 * NG * G)
    gate = tl.load(row + col, mask=live, other=0.0)
    up = tl.load(row + NG * G + col, mask=live, other=0.0)
    if CLAMP:
        gate = tl.minimum(gate, limit)
        up = tl.maximum(tl.minimum(up, limit), -limit)
    g32 = gate.to(tl.float32)
    act = (g32 * tl.sigmoid(g32)).to(h1.dtype.element_ty)
    y = (act * up).to(h1.dtype.element_ty).to(tl.float32)
    _quantize(y, q, s, rows, live, r, col, eps, fp8_min, fp8_max, G, NG, NGP)


@triton.jit
def _sum_add_kernel(h3, shared, out, T, K: tl.constexpr, H: tl.constexpr, BD: tl.constexpr):
    """``moe_sum_reduce`` (fp32 sum over the top-k, rounded to bf16) of one token's BD columns,
    plus the shared expert's output, rounded again: the two ops the block ran, fused."""
    t = tl.program_id(0).to(tl.int64)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros([BD], tl.float32)
    for k in tl.static_range(K):
        acc += tl.load(h3 + (t * K + k) * H + d).to(tl.float32)
    routed = acc.to(out.dtype.element_ty).to(tl.float32)
    total = routed + tl.load(shared + t * H + d).to(tl.float32)
    tl.store(out + t * H + d, total.to(out.dtype.element_ty))


def _group_quant(x, group, *, swiglu=False, limit=None):
    """(e4m3 values, fp32 scales per group) of ``x``, or with ``swiglu`` of
    ``act_and_mul(x)``, clamped at ``limit`` unless it is None."""
    rows, width = x.shape
    if swiglu:
        width //= 2
    q = torch.empty(rows, width, dtype=FP8_DTYPE, device=x.device)
    s = torch.empty(rows, width // group, dtype=torch.float32, device=x.device)
    ng = width // group
    br = max(1, 4096 // triton.next_power_of_2(width))
    fp8 = torch.finfo(FP8_DTYPE)
    args = (rows, 1e-10, fp8.min, fp8.max)
    kw = dict(G=group, NG=ng, NGP=triton.next_power_of_2(ng), BR=br, num_warps=4)
    if not swiglu:
        _quant_kernel[(triton.cdiv(rows, br),)](x, q, s, *args, **kw)
    else:
        _act_quant_kernel[(triton.cdiv(rows, br),)](
            x, q, s, rows, 0.0 if limit is None else float(limit), *args[1:],
            CLAMP=limit is not None, **kw)
    return q, s


def tiles(num_tokens: int, num_experts: int, block_k: int) -> tuple[dict, dict]:
    """(gate/up, down) launch configs: the runner's tile rows (16 up to one token per expert,
    64 above), wider N tiles and deeper pipelines than its defaults, and a K tile of one quant
    group, as the kernel requires."""
    rows = 16 if num_tokens <= num_experts else 64
    gate_up = dict(BLOCK_SIZE_M=rows, BLOCK_SIZE_N=128, BLOCK_SIZE_K=block_k, GROUP_SIZE_M=8,
                   num_warps=4, num_stages=4)
    return gate_up, dict(gate_up, num_stages=3)


def experts(x, w13, w2, s13, s2, topk_weights, topk_ids, *, block_size, swiglu_limit=None,
            shared=None):
    """``fused_experts_fp8``: the routed experts of ``x (T, H)``, weighted and summed per
    token, with ``tiles``; plus ``shared (T, H)`` when given. ``swiglu_limit`` clamps the
    SwiGLU's inputs, None: no clamp."""
    return torch.ops.mstar.fused_moe_prefill_experts(
        x, w13, w2, s13, s2, topk_weights, topk_ids, list(block_size), swiglu_limit, shared)


# One opaque op to dynamo, as runner.fused_experts_fp8: traced inline, the launch path's
# host logic breaks the graph (and fails outright on a symbolic token count in some torch
# versions).
@torch.library.custom_op("mstar::fused_moe_prefill_experts", mutates_args=())
def _experts_op(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_size: list[int],
    swiglu_limit: float | None,
    shared: torch.Tensor | None,
) -> torch.Tensor:
    return _experts(x, w13, w2, s13, s2, topk_weights, topk_ids, tuple(block_size),
                    swiglu_limit, shared)


@_experts_op.register_fake
def _(x, w13, w2, s13, s2, topk_weights, topk_ids, block_size, swiglu_limit, shared):
    return torch.empty_like(x)


def _experts(x, w13, w2, s13, s2, topk_weights, topk_ids, block_size, swiglu_limit, shared):
    T, H = x.shape
    top_k = topk_ids.shape[1]
    w13, w2 = w13.view(FP8_DTYPE), w2.view(FP8_DTYPE)
    num_experts, inter = w2.shape[0], w2.shape[2]
    # what fused_experts_fp8 asserts: these kernels index x, shared and the scales as dense
    # rows and write bf16
    bo, bi = block_size
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert H % bi == 0 and inter % bi == 0, f"hidden {H} / inter {inter} vs block {bi}"
    assert s13.is_contiguous() and s2.is_contiguous()
    assert s13.shape == (num_experts, 2 * inter // bo, H // bi)
    assert s2.shape == (num_experts, H // bo, inter // bi)
    assert shared is None or (shared.shape == x.shape and shared.dtype == x.dtype
                              and shared.is_contiguous())
    ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()
    gate_up, down = tiles(T, num_experts, block_size[1])
    sorted_ids, expert_ids, padded = moe_align_block_size(
        ids, gate_up["BLOCK_SIZE_M"], num_experts)
    routing = dict(topk_weights=topk_weights, topk_ids=ids, sorted_token_ids=sorted_ids,
                   expert_ids=expert_ids, num_tokens_post_padded=padded,
                   compute_type=tl.bfloat16, block_shape=block_size)

    a, a_scale = _group_quant(x, block_size[1])
    h1 = x.new_empty(T * top_k, 2 * inter)
    invoke_fused_moe_kernel_fp8_w8a8(A=a, B=w13, C=h1, A_scale=a_scale, B_scale=s13,
                                     mul_routed_weight=False, top_k=top_k, config=gate_up,
                                     **routing)
    a, a_scale = _group_quant(h1, block_size[1], swiglu=True, limit=swiglu_limit)
    h3 = x.new_empty(T, top_k, H)
    if inter == 2 * block_size[1] and H % (4 * 64) == 0:
        _down(a, a_scale, w2, s2, topk_weights, ids, sorted_ids, expert_ids, padded,
              gate_up["BLOCK_SIZE_M"], h3, block_size)
    else:
        invoke_fused_moe_kernel_fp8_w8a8(A=a, B=w2, C=h3.view(T * top_k, H), A_scale=a_scale,
                                         B_scale=s2, mul_routed_weight=True, top_k=1,
                                         config=down, **routing)
    out = torch.empty_like(x)
    if shared is not None and H % 2048 == 0:
        _sum_add_kernel[(T, H // 2048)](h3, shared, out, T, K=top_k, H=H, BD=2048,
                                        num_warps=16)
        return out
    moe_sum_reduce_triton(h3, out, routed_scaling_factor=1.0)
    return out if shared is None else out + shared
