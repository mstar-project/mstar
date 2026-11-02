"""A TopKRouter routes a decode batch and aligns it in one kernel. Its choice of experts, its
weights and the expert GEMMs' output have to be the eager router's, and a batch it doesn't
cover has to take the eager path."""
from __future__ import annotations

import json

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


@pytest.mark.parametrize(("tuned", "M", "picked"), [
    ({1: "a", 4: "b", 16: "c"}, 1, "a"),
    ({1: "a", 4: "b", 16: "c"}, 2, "a"),
    ({1: "a", 4: "b", 16: "c"}, 3, "b"),
    ({1: "a", 4: "b", 16: "c"}, 10, "b"),
    ({1: "a", 4: "b", 16: "c"}, 11, "c"),
    ({1: "a", 4: "b", 16: "c"}, 4096, "c"),
])
def test_the_nearest_tuned_entry_is_the_nearest_size(tuned, M, picked):
    from mstar.utils.fused_moe.kernels import _nearest_tuned

    assert _nearest_tuned(tuned, M) == picked


@cuda
def test_a_compile_with_a_symbolic_token_count_takes_tuned_tiles(monkeypatch, tmp_path):
    """A prompt past the capture buckets runs compiled, and a second length makes
    its token count symbolic. Tuned tiles have to stay constants there."""
    from mstar.utils.fused_moe import fused_experts, kernels

    def tile(stages):
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4,
                "num_stages": stages, "down": {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}}

    table = tmp_path / "tiles.json"
    # tiles that differ by size, as a real table's do: an int that changes between compiles of a frame turns symbolic
    table.write_text(json.dumps({"1": tile(3), "64": tile(4), "256": tile(2), "512": tile(5)}))
    monkeypatch.setattr(kernels, "config_path", lambda E, N, K: table)
    monkeypatch.setattr(kernels, "_tuned_tables", {})
    w1 = torch.randn(128, 128, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(128, 256, 64, device="cuda", dtype=torch.bfloat16) * 0.05

    def run(x):
        ids = torch.topk(x[:, :128].float(), 8, dim=-1).indices
        weights = torch.full(ids.shape, 0.125, device=x.device, dtype=x.dtype)
        return fused_experts(x, w1, w2, weights, ids)

    compiled = torch.compile(run)  # as the engine compiles: a second length makes the axis symbolic
    for tokens in (4, 300, 411, 800):
        x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
        assert compiled(x).shape == x.shape


@cuda
@pytest.mark.parametrize("tokens", [3, 5])
def test_the_one_kernel_route_runs_under_a_symbolic_token_count(tokens):
    router = _router(device="cuda")
    w1 = torch.zeros(128, 64, 256, device="cuda", dtype=torch.bfloat16)

    def run(x):
        return router.route_aligned(x, w1)[0]

    compiled = torch.compile(run)
    for rows in (tokens, tokens + 1):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        assert compiled(x).shape == (rows, 8)


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
