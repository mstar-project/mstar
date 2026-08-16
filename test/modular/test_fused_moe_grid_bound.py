"""Stress the launch-grid bound in ``invoke_fused_moe_kernel``.

The grid is no longer sized from ``len(sorted_token_ids)`` (the worst case,
one partial block per expert) but from the tighter host-computable bound
``cdiv(num_valid, BLOCK_M) + min(E, num_valid)``.  If that bound is ever too
small, output tiles are silently skipped and the result is quietly wrong
rather than crashing -- so the routing distributions that could break it are
tested explicitly:

* uniform  -- tokens spread over all experts, the common case
* skewed   -- every token to one expert, the maximal per-expert block count
* two_hot  -- all tokens to two experts
* random   -- unconstrained

and against every ``BLOCK_SIZE_M`` the tuner may emit.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE requires CUDA")

BLOCK_MS = (16, 32, 64, 128)
M_VALUES = (1, 2, 3, 7, 8, 16, 17, 64, 129, 512)


def _routing(kind: str, m: int, num_experts: int, top_k: int, device) -> torch.Tensor:
    """``(m, top_k)`` int32 expert ids with the requested skew."""
    if kind == "skewed":
        ids = torch.zeros(m, top_k, device=device, dtype=torch.int32)
        # top_k distinct experts are required per token; pack them at the low end.
        ids += torch.arange(top_k, device=device, dtype=torch.int32)
        return ids
    if kind == "two_hot":
        base = torch.tensor([0, 1], device=device, dtype=torch.int32).repeat(top_k)[:top_k]
        return base.unsqueeze(0).expand(m, top_k).contiguous()
    if kind == "uniform":
        return (
            (torch.arange(m * top_k, device=device, dtype=torch.int32) % num_experts)
            .reshape(m, top_k)
            .contiguous()
        )
    g = torch.Generator(device="cpu").manual_seed(1234 + m)
    return torch.stack(
        [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(m)]
    ).to(device=device, dtype=torch.int32)


def _reference(hidden_states, w1, topk_ids, top_k):
    """Dense per-slot gate+up GEMM in fp32, cast back."""
    ids = topk_ids.reshape(-1).to(torch.int64)
    m = hidden_states.shape[0]
    rows = hidden_states[torch.arange(m, device=ids.device).repeat_interleave(top_k)]
    w = w1[ids]
    out = torch.bmm(rows.unsqueeze(1).float(), w.transpose(1, 2).float()).squeeze(1)
    return out.to(hidden_states.dtype)


@pytest.mark.parametrize("kind", ["uniform", "skewed", "two_hot", "random"])
@pytest.mark.parametrize("block_m", BLOCK_MS)
@pytest.mark.parametrize("m", M_VALUES)
def test_grid_bound_covers_every_tile(kind, block_m, m):
    import triton.language as tl

    from mstar.utils.fused_moe.align import moe_align_block_size
    from mstar.utils.fused_moe.kernels import invoke_fused_moe_kernel

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    E, top_k, hidden, inter = 32, 4, 256, 128

    hidden_states = torch.randn(m, hidden, device=device, dtype=dtype)
    w1 = torch.randn(E, 2 * inter, hidden, device=device, dtype=dtype) / (hidden**0.5)
    topk_ids = _routing(kind, m, E, top_k, device)
    topk_weights = torch.rand(m, top_k, device=device, dtype=dtype)

    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_m, E)
    # Poison the output so a skipped tile cannot pass by coincidence.
    out = torch.full((m * top_k, 2 * inter), float("nan"), device=device, dtype=dtype)

    invoke_fused_moe_kernel(
        A=hidden_states,
        B=w1,
        C=out,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_post,
        mul_routed_weight=False,
        top_k=top_k,
        config={
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        compute_type=tl.bfloat16,
    )
    torch.cuda.synchronize()

    assert not out.isnan().any(), "grid bound skipped an output tile"
    ref = _reference(hidden_states, w1, topk_ids, top_k)
    rel = (out.float() - ref.float()).abs().max() / ref.abs().float().max().clamp_min(1e-6)
    assert rel < 2e-2, f"rel_err={rel:.3g}"


@pytest.mark.parametrize("block_m", BLOCK_MS)
@pytest.mark.parametrize("m", M_VALUES)
def test_bound_is_never_below_the_true_block_count(block_m, m):
    """The analytic bound must dominate the count the align kernel produces."""
    import triton

    from mstar.utils.fused_moe.align import moe_align_block_size

    E, top_k = 128, 8
    for kind in ("uniform", "skewed", "two_hot", "random"):
        topk_ids = _routing(kind, m, E, top_k, "cuda")
        _, _, num_post = moe_align_block_size(topk_ids, block_m, E)
        real = triton.cdiv(int(num_post.item()), block_m)
        num_valid = topk_ids.numel()
        bound = triton.cdiv(num_valid, block_m) + min(E, num_valid)
        assert real <= bound, f"{kind} M={m} BM={block_m}: real={real} > bound={bound}"
