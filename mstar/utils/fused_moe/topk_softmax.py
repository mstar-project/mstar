"""A softmax router's top-k in one kernel: ``mstar::moe_topk_softmax``.

Given router logits ``(tokens, experts)``, returns each token's ``top_k``
largest softmax probabilities (fp32, renormalized to sum to one when asked)
and their expert ids, largest first, the lower id first among equal
probabilities. Unfused, a decode step runs the softmax, two sort kernels for
``torch.topk``, the renormalization and the cast of the ids to the int32 the
fused MoE reads per MoE layer; this is one launch. The ids are int32 here.

Only AMD GPUs have an implementation so far. ``TopKRouter`` reaches it
through ``route_topk_softmax``, which keeps today's torch code elsewhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mstar.kernels.numerics import Tolerance
from mstar.kernels.registry import KernelOp, ShapePolicy
from mstar.platforms.dispatch import PlatformDispatch


def _softmax_topk(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``TopKRouter``'s torch path: fp32 softmax, ``torch.topk`` (int64 ids), renormalize."""
    probs = F.softmax(logits, dtype=torch.float, dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(probs.dtype), ids


def _topk_softmax_reference(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids = _softmax_topk(logits, top_k, renormalize)
    return weights, ids.to(torch.int32)


def _topk_softmax_fake(logits, top_k, renormalize):
    rows = logits.shape[0]
    return (logits.new_empty((rows, top_k), dtype=torch.float32),
            logits.new_empty((rows, top_k), dtype=torch.int32))


@triton.jit
def _topk_softmax_kernel(
    logits_ptr, weights_ptr, ids_ptr, stride_row, num_experts,
    TOP_K: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr, RENORMALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    valid = offs < num_experts
    x = tl.load(logits_ptr + row * stride_row + offs, mask=valid, other=float("-inf")).to(tl.float32)
    e = tl.where(valid, tl.exp(x - tl.max(x, axis=0)), 0.0)
    # probabilities are >= 0, so -1 marks a lane that can no longer be picked
    p = tl.where(valid, e / tl.sum(e, axis=0), -1.0)
    offs_k = tl.arange(0, BLOCK_K)
    vals = tl.zeros([BLOCK_K], dtype=tl.float32)
    ids = tl.zeros([BLOCK_K], dtype=tl.int32)
    for k in tl.static_range(TOP_K):
        best = tl.max(p, axis=0)
        pick = tl.min(tl.where(p == best, offs, BLOCK_E), axis=0)
        vals = tl.where(offs_k == k, best, vals)
        ids = tl.where(offs_k == k, pick, ids)
        p = tl.where(offs == pick, -1.0, p)
    keep = offs_k < TOP_K
    if RENORMALIZE:
        vals = vals / tl.sum(tl.where(keep, vals, 0.0), axis=0)
    tl.store(weights_ptr + row * TOP_K + offs_k, vals, mask=keep)
    tl.store(ids_ptr + row * TOP_K + offs_k, ids, mask=keep)


def _triton_topk_softmax():
    def run(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        rows, experts = logits.shape
        weights = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
        ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
        if rows:
            _topk_softmax_kernel[(rows,)](
                logits, weights, ids, logits.stride(0), experts,
                TOP_K=top_k, BLOCK_E=triton.next_power_of_2(experts),
                BLOCK_K=triton.next_power_of_2(top_k), RENORMALIZE=renormalize, num_warps=1,
            )
        return weights, ids

    return run


def _rows_experts(logits, top_k, renormalize) -> dict[str, int]:
    return {"rows": logits.shape[0], "experts": logits.shape[1], "top_k": top_k}


def _inputs(shape, dtype, device, generator=None):
    # distinct values in every row (k / 16 is exact in bf16), so the top-k is unique
    order = torch.rand(shape["rows"], shape["experts"], device=device, generator=generator).argsort(dim=-1)
    logits = ((order - shape["experts"] // 2) / 16).to(dtype)
    return logits, shape["top_k"], True


MOE_TOPK_SOFTMAX = KernelOp(
    "moe_topk_softmax",
    "(Tensor logits, int top_k, bool renormalize) -> (Tensor, Tensor)",
    fake=_topk_softmax_fake,
    reference=_topk_softmax_reference,
    impls={"rocm": _triton_topk_softmax},
    shapes=ShapePolicy(
        dims=("rows", "experts", "top_k"),
        key=_rows_experts,
        inputs=_inputs,
        warm=({"rows": 1, "experts": 128, "top_k": 8}, {"rows": 64, "experts": 128, "top_k": 8},
              {"rows": 1, "experts": 128, "top_k": 6}),
    ),
    # fp32 softmax and renormalization, a few roundings each; the ids are exact
    tolerance=Tolerance.accumulated(rtol=1e-5, atol_scale=1e-6),
)


def _torch_router():
    return _softmax_topk


def _fused_router():
    op = torch.ops.mstar.moe_topk_softmax

    def route(logits: torch.Tensor, top_k: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return op(logits, top_k, renormalize)

    return route


# ``(logits, top_k, renormalize) -> (weights, ids)``: int32 ids from the op on
# AMD GPUs, int64 from torch.topk elsewhere. NVIDIA GPUs and the CPU keep the
# torch path; it can move onto the op once an NVIDIA A/B shows it neutral or
# better.
route_topk_softmax = PlatformDispatch(
    "moe_topk_softmax", rocm=_fused_router, default=_torch_router,
)
