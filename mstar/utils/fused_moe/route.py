"""A decode batch's routing and block alignment in one kernel, where they are otherwise
a softmax or sigmoid, a top-k, a sort, a renormalize, a cast and two alignment kernels per
MoE layer, each a graph node of its own."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.runner import MoeAlignment


@triton.jit
def _route_align_kernel(
    logits_ptr, weights_ptr, ids_ptr, sorted_ptr, expert_ids_ptr, post_pad_ptr,
    T, stride_l,
    E: tl.constexpr, BLOCK_E: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr, PAD_LEN: tl.constexpr, BLOCK_PAD: tl.constexpr, SOFTMAX: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_T)
    experts = tl.arange(0, BLOCK_E)
    logits = tl.load(
        logits_ptr + rows[:, None] * stride_l + experts[None, :],
        mask=(rows[:, None] < T) & (experts[None, :] < E), other=-float("inf"),
    ).to(tl.float32)
    kk = tl.arange(0, K)
    top_vals = tl.zeros((BLOCK_T, K), dtype=tl.float32)
    top_ids = tl.zeros((BLOCK_T, K), dtype=tl.int32)
    for i in tl.static_range(K):
        best = tl.max(logits, axis=1)
        idx = tl.min(tl.where(logits == best[:, None], experts[None, :], BLOCK_E), axis=1)
        top_vals = tl.where(kk[None, :] == i, best[:, None], top_vals)
        top_ids = tl.where(kk[None, :] == i, idx[:, None], top_ids)
        logits = tl.where(experts[None, :] == idx[:, None], -float("inf"), logits)
    dtype = weights_ptr.dtype.element_ty
    if SOFTMAX:
        # renormalized over the top k, a softmax needs nothing of the other experts
        s = tl.exp(top_vals - tl.max(top_vals, axis=1)[:, None])
        total = tl.sum(s, axis=1)
    else:
        s = tl.sigmoid(top_vals).to(dtype).to(tl.float32)
        total = tl.sum(s, axis=1).to(dtype).to(tl.float32)
    row_ok = rows[:, None] < T
    tl.store(weights_ptr + rows[:, None] * K + kk[None, :], (s / total[:, None]).to(dtype), mask=row_ok)
    tl.store(ids_ptr + rows[:, None] * K + kk[None, :], top_ids, mask=row_ok)

    # Block alignment, matching moe_align_block_size: each expert's slots are
    # padded to a multiple of BLOCK_M, experts laid out in index order.
    num_slots = T * K
    slot = tl.arange(0, BLOCK_T * K)
    slot_expert = tl.reshape(top_ids, (BLOCK_T * K,))
    valid = slot < num_slots
    onehot = (slot_expert[:, None] == experts[None, :]) & valid[:, None]
    counts = tl.sum(onehot.to(tl.int32), axis=0)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    offsets = tl.cumsum(padded, axis=0) - padded
    earlier = (slot_expert[:, None] == slot_expert[None, :]) & (slot[None, :] < slot[:, None]) & valid[None, :]
    rank = tl.sum(earlier.to(tl.int32), axis=1)
    dest = tl.sum(tl.where(onehot, offsets[None, :], 0), axis=1) + rank

    pad = tl.arange(0, BLOCK_PAD)
    tl.store(sorted_ptr + pad, tl.full((BLOCK_PAD,), 0, tl.int32) + num_slots, mask=pad < PAD_LEN)
    tl.debug_barrier()
    tl.store(sorted_ptr + dest, slot, mask=valid)
    tl.store(expert_ids_ptr + dest // BLOCK_M, slot_expert, mask=valid & (rank % BLOCK_M == 0))
    tl.store(post_pad_ptr, tl.sum(padded, axis=0))


ROUTE_ALIGN_MAX_SLOTS = 128


def route_align(
    logits: torch.Tensor, k: int, block_m: int, dtype: torch.dtype, *, softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, MoeAlignment]:
    """Top-k routing, renormalized, plus ``moe_align_block_size`` in one
    single-program kernel. The weights are sigmoids rounded to ``dtype`` like
    an eager bf16 router's, or with ``softmax`` a softmax in fp32, rounded once.
    A tie goes to the lowest expert.

    Returns ``(weights, ids, alignment)``.
    The alignment buffers are sized for at most ``min(E, slots)`` active
    experts, the bound ``invoke_fused_moe_kernel`` sizes its grid with.
    Meant for decode-sized batches (``tokens * k <= ROUTE_ALIGN_MAX_SLOTS``).
    """
    tokens, experts = logits.shape
    num_slots = tokens * k
    assert logits.stride(1) == 1 and 0 < num_slots <= ROUTE_ALIGN_MAX_SLOTS
    pad_len = triton.cdiv(num_slots + min(experts, num_slots) * (block_m - 1), block_m) * block_m
    device = logits.device
    weights = torch.empty((tokens, k), dtype=dtype, device=device)
    ids = torch.empty((tokens, k), dtype=torch.int32, device=device)
    sorted_ids = torch.empty((pad_len,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((pad_len // block_m,), dtype=torch.int32, device=device)
    post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    _route_align_kernel[(1,)](
        logits, weights, ids, sorted_ids, expert_ids, post_pad, tokens, logits.stride(0),
        E=experts, BLOCK_E=triton.next_power_of_2(experts), K=k, BLOCK_M=block_m,
        BLOCK_T=triton.next_power_of_2(tokens), PAD_LEN=pad_len,
        BLOCK_PAD=triton.next_power_of_2(pad_len), SOFTMAX=softmax, num_warps=4,
    )
    return weights, ids, MoeAlignment(sorted_ids, expert_ids, post_pad, block_m)
