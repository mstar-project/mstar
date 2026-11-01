"""A TopKRouter routes a decode batch and aligns it in one kernel. Its choice of experts, its
weights and the expert GEMMs' output have to be the eager router's, and a batch it doesn't
cover has to take the eager path."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from mstar.model.components.moe import TopKRouter  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs Triton kernels")


def _router(experts=128, hidden=256, top_k=8, norm=True, device="cpu") -> TopKRouter:
    router = TopKRouter(hidden, experts, top_k, norm_topk_prob=norm).to(device)
    torch.manual_seed(0)
    router.weight.data.normal_(std=0.2)
    return router.to(torch.bfloat16)


def test_the_eager_router_stays_for_what_the_kernel_does_not_cover():
    cpu = _router()
    assert cpu.route_aligned(torch.zeros(1, 256, dtype=torch.bfloat16), None) is None, "CPU tensors"


@cuda
@pytest.mark.parametrize(("tokens", "top_k"), [(1, 8), (3, 8), (8, 8), (16, 8), (1, 6), (5, 6), (16, 6)])
def test_the_kernels_experts_and_weights_are_the_eager_routers(tokens, top_k):
    from mstar.utils.fused_moe.align import moe_align_block_size

    router = _router(device="cuda", top_k=top_k)
    torch.manual_seed(tokens)
    # fp32 logits rounded to bf16 tie often; torch.topk orders ties arbitrarily
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)
    weights, ids, alignment = router.route_aligned(x, w1)
    ref_weights, ref_ids, _ = router(x)

    logits = F.linear(x, router.weight).float()
    for row in range(tokens):
        values = logits[row, ids[row].long()]
        assert (values[:-1] >= values[1:]).all(), "ordered by logit"
        floor = values[-1]
        assert (logits[row] > floor).sum() <= top_k - 1, "no expert above the kth was left out"
    # where no tie crosses the kth, the two agree
    kth = logits.topk(top_k + 1, dim=-1).values
    clean = kth[:, top_k - 1] > kth[:, top_k]
    assert torch.equal(ids[clean].sort(dim=-1).values.long(), ref_ids[clean].sort(dim=-1).values)
    assert torch.allclose(
        weights[clean].sort(dim=-1).values.float(), ref_weights[clean].sort(dim=-1).values.float(), atol=1e-2,
    )

    ref_sorted, ref_experts, ref_post = moe_align_block_size(ids, alignment.block_m, 128)
    n = int(alignment.num_tokens_post_padded)
    assert n == int(ref_post)
    m = alignment.block_m
    assert torch.equal(alignment.expert_ids[:n // m], ref_experts[:n // m])
    for b in range(n // m):
        block = slice(b * m, (b + 1) * m)
        assert sorted(alignment.sorted_token_ids[block].tolist()) == sorted(ref_sorted[block].tolist())


@cuda
@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_the_experts_output_matches_through_the_aligned_path(tokens):
    from mstar.utils.fused_moe import fused_experts

    router = _router(device="cuda")
    torch.manual_seed(1000 + tokens)
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(128, 128, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(128, 256, 64, device="cuda", dtype=torch.bfloat16) * 0.05
    weights, ids, alignment = router.route_aligned(x, w1)
    together = fused_experts(x, w1, w2, weights, ids, alignment=alignment)
    apart = fused_experts(x, w1, w2, weights, ids)
    assert torch.allclose(together.float(), apart.float(), atol=2e-2, rtol=1e-2)


@cuda
def test_a_batch_past_the_kernel_takes_the_eager_router():
    router = _router(device="cuda")
    x = torch.zeros(17, 256, device="cuda", dtype=torch.bfloat16)
    assert router.route_aligned(x, torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)) is None


@cuda
def test_without_renormalizing_the_eager_router_stays():
    router = _router(device="cuda", norm=False)
    x = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    assert router.route_aligned(x, torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)) is None
