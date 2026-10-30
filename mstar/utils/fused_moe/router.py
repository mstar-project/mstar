"""Softmax top-k routing in one kernel.

``TopKRouter``'s softmax, top-k, renormalization and dtype casts were five launches per MoE
layer (``softmax``, ``topk``, ``sum``, ``div``, and the int32 cast ahead of the alignment),
each a few microseconds at decode width. One program per token does them all here.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_topk_kernel(logits, w_out, id_out, stride_t, E: tl.constexpr, EP: tl.constexpr,
                         K: tl.constexpr, KP: tl.constexpr, RENORM: tl.constexpr):
    """One program per token: the fp32 softmax of its ``E`` logits, the ``K`` largest
    (ties to the lower expert), optionally renormalized to sum to 1. The arithmetic is
    ``torch.softmax`` then ``topk`` then ``w / w.sum()``, in that order."""
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, EP)
    valid = e < E
    x = tl.load(logits + t * stride_t + e, mask=valid, other=float("-inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x, 0))
    prob = p / tl.sum(p, 0)
    # probabilities are >= 0, so -1 marks a taken or padding column
    cur = tl.where(valid, prob, -1.0)
    kk = tl.arange(0, KP)
    ids = tl.zeros([KP], tl.int32)
    ws = tl.zeros([KP], tl.float32)
    for j in tl.static_range(K):
        best = tl.max(cur, 0)
        idx = tl.min(tl.where(cur == best, e, EP), 0)
        ids = tl.where(kk == j, idx, ids)
        ws = tl.where(kk == j, best, ws)
        cur = tl.where(e == idx, -1.0, cur)
    if RENORM:
        ws = ws / tl.sum(tl.where(kk < K, ws, 0.0), 0)
    tl.store(w_out + t * K + kk, ws, mask=kk < K)
    tl.store(id_out + t * K + kk, ids.to(id_out.dtype.element_ty), mask=kk < K)


@torch.library.custom_op("mstar::softmax_topk", mutates_args=())
def softmax_topk(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weights fp32, ids int64)`` of each row's ``top_k`` softmax probabilities, as
    ``torch.topk(torch.softmax(logits.float(), -1), top_k)`` then, with ``renormalize``,
    ``weights / weights.sum(-1)``. ``logits (T, E)`` on CUDA, last dim contiguous."""
    T, E = logits.shape
    assert logits.stride(1) == 1 and 0 < top_k <= E
    weights = torch.empty(T, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(T, top_k, dtype=torch.int64, device=logits.device)
    if T:
        _softmax_topk_kernel[(T,)](
            logits, weights, ids, logits.stride(0), E=E, EP=triton.next_power_of_2(E), K=top_k,
            KP=triton.next_power_of_2(top_k), RENORM=renormalize, num_warps=1 if E <= 256 else 4)
    return weights, ids


@softmax_topk.register_fake
def _(logits, top_k, renormalize):
    T = logits.shape[0]
    return (logits.new_empty(T, top_k, dtype=torch.float32),
            logits.new_empty(T, top_k, dtype=torch.int64))


# a row is one program's registers; past this the unfused path is the better kernel
SOFTMAX_TOPK_MAX_EXPERTS = 1024


def softmax_topk_supported(logits: torch.Tensor, top_k: int) -> bool:
    return (logits.is_cuda and logits.dim() == 2 and logits.shape[1] <= SOFTMAX_TOPK_MAX_EXPERTS
            and top_k <= 64 and logits.stride(1) == 1)
