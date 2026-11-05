"""A TopKRouter with ``fused_softmax_topk`` routes a decode batch and aligns it in one kernel.
Its choice of experts, its weights and the expert GEMMs' output have to be the eager router's,
and a batch it doesn't cover has to take ``router_topk``."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from mstar.model.components import moe  # noqa: E402
from mstar.model.components.moe import (  # noqa: E402
    ParallelSparseMoeBlock,
    ParallelSparseMoeBlockWithSharedExpert,
    TopKRouter,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs Triton kernels")


def _router(experts=128, hidden=256, top_k=8, norm=True, device="cpu", fused=True) -> TopKRouter:
    router = TopKRouter(hidden, experts, top_k, norm_topk_prob=norm, fused_softmax_topk=fused).to(device)
    torch.manual_seed(0)
    router.weight.data.normal_(std=0.2)
    return router.to(torch.bfloat16)


def test_a_cpu_batch_takes_the_eager_router_with_the_flag_set():
    cpu = _router()
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    weights, ids, alignment = cpu.route(x, torch.zeros(128, 64, 256, dtype=torch.bfloat16))
    assert alignment is None
    probs = F.softmax(F.linear(x, cpu.weight), dtype=torch.float, dim=-1)
    ref_weights, ref_ids = probs.topk(8, dim=-1)
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights, ref_weights / ref_weights.sum(dim=-1, keepdim=True))


@pytest.mark.parametrize(("world", "masked", "aligns"), [(1, False, True), (1, True, False), (2, False, False)])
def test_a_block_offers_its_weights_only_where_it_can_use_the_alignment(monkeypatch, world, masked, aligns):
    block = ParallelSparseMoeBlock(
        hidden_size=8, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        comm_group=SimpleNamespace(world_size=world, rank=0),
    )
    seen = []
    real = moe._route
    monkeypatch.setattr(moe, "_route", lambda gate, flat, w1, states: seen.append(w1) or real(gate, flat, w1, states))
    monkeypatch.setattr(moe, "_dispatch", lambda flat, *args, **kwargs: torch.zeros_like(flat))
    block._dispatch_tp = lambda flat, *args, **kwargs: torch.zeros_like(flat)
    x = torch.randn(3, 8)
    block(x, token_valid=torch.ones(3, dtype=torch.bool) if masked else None)
    assert (seen[0] is block.experts.gate_up_proj) is aligns
    assert (seen[0] is None) is not aligns


@pytest.mark.parametrize(("world", "aligns"), [(1, True), (2, False)])
def test_a_shared_expert_block_offers_its_weights_on_one_rank(monkeypatch, world, aligns):
    block = ParallelSparseMoeBlockWithSharedExpert(
        hidden_size=8, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        shared_expert=torch.nn.Identity(), comm_group=SimpleNamespace(world_size=world, rank=0),
    )
    seen = []
    real = moe._route
    monkeypatch.setattr(moe, "_route", lambda gate, flat, w1, states: seen.append(w1) or real(gate, flat, w1, states))
    monkeypatch.setattr(moe, "_dispatch", lambda flat, *args, **kwargs: torch.zeros_like(flat))
    block._dispatch_tp = lambda flat, *args, **kwargs: torch.zeros_like(flat)
    block(torch.randn(3, 8))
    assert (seen[0] is block.experts.gate_up_proj) is aligns


@cuda
@pytest.mark.parametrize(("tokens", "top_k"), [(1, 8), (3, 8), (8, 8), (16, 8), (1, 6), (5, 6), (16, 6)])
def test_the_kernels_experts_and_weights_are_the_eager_routers(tokens, top_k):
    from mstar.utils.fused_moe.align import moe_align_block_size

    router = _router(device="cuda", top_k=top_k)
    torch.manual_seed(tokens)
    # fp32 logits rounded to bf16 tie often; torch.topk orders ties arbitrarily
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)
    weights, ids, alignment = router.route(x, w1)
    assert alignment is not None
    ref_weights, ref_ids, _ = _router(device="cuda", top_k=top_k, fused=False)(x)

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
    weights, ids, alignment = router.route(x, w1)
    assert alignment is not None
    together = fused_experts(x, w1, w2, weights, ids, alignment=alignment)
    apart = fused_experts(x, w1, w2, weights, ids)
    assert torch.allclose(together.float(), apart.float(), atol=2e-2, rtol=1e-2)


@cuda
@pytest.mark.parametrize("tokens", [3, 5])
def test_the_one_kernel_route_runs_under_a_symbolic_token_count(tokens):
    router = _router(device="cuda")
    w1 = torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)

    def run(x):
        return router.route(x, w1)[0]

    compiled = torch.compile(run)
    for rows in (tokens, tokens + 1):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        assert compiled(x).shape == (rows, 8)


def _takes_router_topk(router, x, align_w1):
    """The batch takes ``router_topk``, with its fp32 weights and int64 experts, and is what ``forward`` returns."""
    weights, ids, alignment = router.route(x, align_w1)
    ref_weights, ref_ids, _ = router(x)
    return (
        alignment is None and ids.dtype == torch.int64 and weights.dtype == torch.float32
        and torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)
    )


@cuda
def test_the_last_batch_that_fits_takes_the_kernel_and_the_next_takes_router_topk():
    router = _router(device="cuda")
    w1 = torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(moe.ROUTE_ALIGN_MAX_SLOTS // 8, 256, device="cuda", dtype=torch.bfloat16)
    assert router.route(x, w1)[2] is not None
    assert _takes_router_topk(router, torch.randn(x.shape[0] + 1, 256, device="cuda", dtype=torch.bfloat16), w1)


@cuda
def test_without_renormalizing_it_takes_router_topk():
    router = _router(device="cuda", norm=False)
    w1 = torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)
    assert _takes_router_topk(router, torch.randn(4, 256, device="cuda", dtype=torch.bfloat16), w1)


@cuda
def test_with_nothing_to_align_against_it_takes_router_topk():
    router = _router(device="cuda")
    assert _takes_router_topk(router, torch.randn(4, 256, device="cuda", dtype=torch.bfloat16), None)


@cuda
def test_a_batch_fused_experts_runs_per_pair_takes_router_topk():
    """Up to ``ROUTED_MAX_TOKENS`` tokens of a shape the per-pair GEMVs fit, ``fused_experts`` reads no
    alignment, so the router doesn't build one; past that the batch takes the one-kernel route."""
    from mstar.utils.fused_moe import decode

    router = _router(hidden=1024, device="cuda")
    w1 = torch.zeros(128, 256, 1024, device="cuda", dtype=torch.bfloat16)
    assert decode.routed_fits(1024, 128)
    small = torch.randn(decode.ROUTED_MAX_TOKENS, 1024, device="cuda", dtype=torch.bfloat16)
    assert _takes_router_topk(router, small, w1)
    larger = torch.randn(decode.ROUTED_MAX_TOKENS + 1, 1024, device="cuda", dtype=torch.bfloat16)
    assert router.route(larger, w1)[2] is not None
