"""``mstar::moe_topk_softmax`` on an AMD GPU against the torch router it replaces.

The fused kernel picks the same experts as softmax + ``torch.topk`` and returns
the same weights to within fp32 rounding, eager, under graph replay and inside a
compiled ``TopKRouter``, at Qwen3-Omni's router sizes (128 experts, the Thinker's
top-8 renormalized, the Talker's top-6 not).

    pytest test/rocm/test_moe_router.py
"""

from __future__ import annotations

import pytest
import torch

from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm kernels run on AMD GPUs")

DEV = torch.device("cuda")
ROUTERS = [(128, 8, True), (128, 6, False)]  # experts, top_k, renormalize: Thinker, Talker


def _fused(logits, top_k, renormalize):
    import mstar.utils.fused_moe.topk_softmax  # noqa: F401  declares the op

    return torch.ops.mstar.moe_topk_softmax(logits, top_k, renormalize)


def _torch(logits, top_k, renormalize):
    """TopKRouter's torch path, int64 ids."""
    from mstar.utils.fused_moe.topk_softmax import _softmax_topk

    return _softmax_topk(logits, top_k, renormalize)


def _no_ties(rows, experts, seed):
    """Logits with distinct values per row, so the top-k is unique."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.stack([torch.randperm(experts, generator=g) for _ in range(rows)])
    return (perm.float() / 16 - 4).to(DEV, torch.bfloat16)


@pytest.mark.parametrize(("experts", "top_k", "renormalize"), ROUTERS)
@pytest.mark.parametrize("rows", [1, 3, 8, 32, 64, 512])
def test_matches_torch_router(experts, top_k, renormalize, rows):
    logits = _no_ties(rows, experts, seed=rows)
    w, ids = _fused(logits, top_k, renormalize)
    w_ref, ids_ref = _torch(logits, top_k, renormalize)
    # int32, what the fused MoE reads, so no cast kernel follows
    assert w.dtype == w_ref.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.equal(ids.long(), ids_ref)
    torch.testing.assert_close(w, w_ref, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize(("experts", "top_k", "renormalize"), ROUTERS)
def test_ties_go_to_the_lower_expert(experts, top_k, renormalize):
    # bf16 router logits tie often; equal probabilities are taken lowest id first
    logits = torch.zeros(4, experts, device=DEV, dtype=torch.bfloat16)
    logits[:, 5] = 2.0
    logits[1, 100] = 2.0
    _, ids = _fused(logits, top_k, renormalize)
    rest = [i for i in range(experts) if i not in (5, 100)]
    assert ids[0].tolist() == [5, *[i for i in range(experts) if i != 5][:top_k - 1]]
    assert ids[1].tolist() == [5, 100, *rest[:top_k - 2]]


def test_random_bf16_logits_pick_the_same_experts_as_torch():
    """Real router logits are bf16, so ties happen. Every row whose 8th and 9th
    largest logits differ picks torch's experts; a tie across that boundary is
    a free choice, which the kernel makes by id."""
    g = torch.Generator(device=DEV).manual_seed(0)
    logits = (torch.randn(4096, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
    w, ids = _fused(logits, 8, True)
    w_ref, ids_ref = _torch(logits, 8, True)
    top = logits.float().sort(-1, descending=True).values
    clear = top[:, 7] != top[:, 8]
    same_set = (ids.sort(-1).values == ids_ref.sort(-1).values).all(-1)
    assert same_set[clear].all()
    rows = clear & same_set
    torch.testing.assert_close(w[rows], w_ref[rows], rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("rows", [1, 32])
def test_graph_replay(rows):
    logits = _no_ties(rows, 128, seed=7)
    static = logits.clone()
    _fused(static, 8, True)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fused(static, 8, True)
    for seed in (1, 2):
        static.copy_(_no_ties(rows, 128, seed=seed))
        graph.replay()
        w_ref, ids_ref = _torch(static, 8, True)
        assert torch.equal(out[1].long(), ids_ref)
        torch.testing.assert_close(out[0], w_ref, rtol=2e-6, atol=1e-7)


def test_compiled_router_is_one_kernel_without_graph_breaks():
    from mstar.model.components.moe import TopKRouter

    router = TopKRouter(hidden_size=32, num_experts=128, num_experts_per_tok=8).to(DEV, torch.bfloat16)
    # small integers, so every logit is exact in bf16 and the compiled and eager
    # GEMMs agree to the bit; the picks then compare exactly
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        router.weight.copy_(torch.randint(-2, 3, (128, 32), generator=g).to(DEV, torch.bfloat16))
    x = torch.randint(-2, 3, (16, 32), generator=g).to(DEV, torch.bfloat16)
    torch._dynamo.reset()
    with torch.no_grad():
        explain = torch._dynamo.explain(router)(x)
        assert explain.graph_break_count == 0, explain.break_reasons
        w, ids, _ = torch.compile(router, fullgraph=True)(x)
        w_eager, ids_eager, _ = router(x)
    assert ids.dtype == torch.int32
    assert torch.equal(ids, ids_eager) and torch.equal(w, w_eager)
