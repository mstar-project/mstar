"""The draft block's own attention and its merge with the paged context part, as one launch per
(row, head): scores of the row's ``k`` queries against its ``k`` keys (latent and rope parts, all
positions see all positions), softmax and log-sum-exp, the probabilities applied to the keys'
latents, then the merge with FlashInfer's context output by natural log-sum-exp. Replaces some
thirty torch launches per draft layer (fp32 casts, two einsums, log-sum-exp, softmax, another
einsum, the exp / maximum / division of the merge)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dspark_block_attn_kernel(
    q_lat, q_pe, lat, o_ctx, lse_ctx, out, scale,
    stride_ql_tok, stride_ql_head, stride_qp_tok, stride_qp_head,
    K: tl.constexpr, BK: tl.constexpr, H: tl.constexpr, L: tl.constexpr, BL: tl.constexpr, BLC: tl.constexpr,
    R: tl.constexpr, BR: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid // H
    h = pid % H
    i = tl.arange(0, BK)  # the row's queries
    j = tl.arange(0, BK)  # and its keys
    im = i < K
    jm = j < K
    tok_i = r * K + i
    tok_j = r * K + j
    # scores: the latent part summed over L in chunks, then the rope part
    s = tl.zeros([BK, BK], dtype=tl.float32)
    for l0 in tl.static_range(0, BL, BLC):
        l = l0 + tl.arange(0, BLC)
        lm = l < L
        qc = tl.load(q_lat + (tok_i * stride_ql_tok + h * stride_ql_head)[:, None] + l[None, :], mask=im[:, None] & lm[None, :], other=0.0)
        cc = tl.load(lat + tok_j[:, None] * (L + R) + l[None, :], mask=jm[:, None] & lm[None, :], other=0.0)
        s += tl.sum(qc.to(tl.float32)[:, None, :] * cc.to(tl.float32)[None, :, :], axis=2)
    e = tl.arange(0, BR)
    em = e < R
    qp = tl.load(q_pe + (tok_i * stride_qp_tok + h * stride_qp_head)[:, None] + e[None, :], mask=im[:, None] & em[None, :], other=0.0)
    kp = tl.load(lat + tok_j[:, None] * (L + R) + L + e[None, :], mask=jm[:, None] & em[None, :], other=0.0)
    s += tl.sum(qp.to(tl.float32)[:, None, :] * kp.to(tl.float32)[None, :, :], axis=2)
    s = s * scale
    s = tl.where(jm[None, :], s, float("-inf"))
    m = tl.max(s, axis=1)
    p = tl.exp(s - m[:, None])
    ssum = tl.sum(p, axis=1)
    p = p / ssum[:, None]
    lse_blk = m + tl.log(ssum)
    # the merge weights against the context part (natural log-sum-exp from FlashInfer)
    lctx = tl.load(lse_ctx + tok_i * H + h, mask=im, other=float("-inf")).to(tl.float32)
    mm = tl.maximum(lctx, lse_blk)
    w_ctx = tl.exp(lctx - mm)
    w_blk = tl.exp(lse_blk - mm)
    denom = w_ctx + w_blk
    for l0 in tl.static_range(0, BL, BLC):
        l = l0 + tl.arange(0, BLC)
        lm = l < L
        cc = tl.load(lat + tok_j[:, None] * (L + R) + l[None, :], mask=jm[:, None] & lm[None, :], other=0.0)
        o_blk = tl.sum(p[:, :, None] * cc.to(tl.float32)[None, :, :], axis=1)  # [BK, BLC]
        oc = tl.load(o_ctx + (tok_i * H + h)[:, None] * L + l[None, :], mask=im[:, None] & lm[None, :], other=0.0)
        o = (oc.to(tl.float32) * w_ctx[:, None] + o_blk * w_blk[:, None]) / denom[:, None]
        tl.store(out + (tok_i * H + h)[:, None] * L + l[None, :], o.to(out.dtype.element_ty), mask=im[:, None] & lm[None, :])


def dspark_block_attention(
    q_lat: torch.Tensor, q_pe: torch.Tensor, lat: torch.Tensor, o_ctx: torch.Tensor, lse_ctx: torch.Tensor,
    rows: int, kv_lora_rank: int, scale: float,
) -> torch.Tensor:
    """``q_lat [rows * k, H, L]``, ``q_pe [rows * k, H, R]``, ``lat [rows * k, L + R]`` (the block's own
    keys: latent then roped rope part), ``o_ctx [rows * k, H, L]`` and ``lse_ctx [rows * k, H]`` (the
    paged context part and its natural log-sum-exp) -> the merged attention ``[rows * k, H, L]`` in
    ``q_lat``'s dtype. Static shapes, device tensors only."""
    t, h, l = q_lat.shape
    assert l == kv_lora_rank and t % rows == 0 and t > 0
    k = t // rows
    r = q_pe.shape[-1]
    assert q_pe.shape == (t, h, r) and lat.shape == (t, l + r) and o_ctx.shape == (t, h, l) and lse_ctx.shape == (t, h)
    # the queries are read with their strides (they are slices of the draft's q projection); the rest is dense
    assert q_lat.stride(2) == 1 and q_pe.stride(2) == 1
    lat, o_ctx, lse_ctx = (x.contiguous() for x in (lat, o_ctx, lse_ctx))
    out = torch.empty(t, h, l, dtype=q_lat.dtype, device=q_lat.device)
    bl = triton.next_power_of_2(l)
    _dspark_block_attn_kernel[(rows * h,)](
        q_lat, q_pe, lat, o_ctx, lse_ctx, out, float(scale),
        q_lat.stride(0), q_lat.stride(1), q_pe.stride(0), q_pe.stride(1),
        K=k, BK=triton.next_power_of_2(k), H=h, L=l, BL=bl, BLC=min(bl, 64), R=r, BR=triton.next_power_of_2(r),
        num_warps=4,
    )
    return out
