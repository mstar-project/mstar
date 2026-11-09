"""GPU golden for the softmax mode of the router's top-k kernel (``router_topk(..., softmax=True)``
in ``mstar.utils.fused_moe.decode``), which ``TopKRouter(fused_softmax_topk=True)`` routes through."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the router kernel needs a GPU")

DEVICE = "cuda"


def _torch_route(logits, top_k, renormalize):
    probs = F.softmax(logits, dtype=torch.float, dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids


def _fused_route(logits, top_k, renormalize):
    from mstar.utils.fused_moe.decode import router_topk

    return router_topk(logits, None, top_k, 1.0, renormalize, softmax=True)


@pytest.mark.parametrize("tokens", [1, 8, 33, 1000])
@pytest.mark.parametrize("experts,top_k", [(128, 8), (64, 6), (60, 4), (512, 10), (1024, 64)])
@pytest.mark.parametrize("renormalize", [True, False])
def test_softmax_topk_is_softmax_then_topk(tokens, experts, top_k, renormalize):
    torch.manual_seed(tokens + experts)
    logits = (torch.randn(tokens, experts, device=DEVICE) * 2).to(torch.bfloat16)
    want_w, want_ids = _torch_route(logits, top_k, renormalize)
    got_w, got_ids = _fused_route(logits, top_k, renormalize)
    assert got_ids.dtype == torch.int64 and got_w.dtype == torch.float32
    # descending, as torch.topk; torch may order exact ties either way, so compare the logits picked
    assert torch.equal(logits.gather(1, got_ids), logits.gather(1, want_ids))
    torch.testing.assert_close(got_w, want_w, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("experts,top_k", [(128, 8), (60, 4), (1024, 64)])
@pytest.mark.parametrize("renormalize", [True, False])
def test_ties_go_to_the_lower_expert(experts, top_k, renormalize):
    """Logits on a coarse grid tie in almost every row, often across the k-th place: the picks are
    the stable descending order, lower expert id first among equals, with torch's weights."""
    torch.manual_seed(experts + top_k)
    logits = torch.randint(-3, 4, (257, experts), device=DEVICE).to(torch.bfloat16)
    probs = F.softmax(logits, dtype=torch.float, dim=-1)
    want_ids = torch.sort(probs, dim=-1, descending=True, stable=True).indices[:, :top_k]
    want_w, _ = _torch_route(logits, top_k, renormalize)
    got_w, got_ids = _fused_route(logits, top_k, renormalize)
    assert torch.equal(got_ids, want_ids)
    torch.testing.assert_close(got_w, want_w, rtol=1e-6, atol=1e-7)


def test_empty_batch():
    w, ids = _fused_route(torch.empty(0, 128, device=DEVICE, dtype=torch.bfloat16), 8, True)
    assert w.shape == (0, 8) and ids.shape == (0, 8)


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_topk_router_on_cuda_keeps_its_contract(norm_topk_prob):
    """fp32 weights, int64 ids, no state: the softmax-then-top-k of its own logits."""
    from mstar.model.components.moe import TopKRouter

    torch.manual_seed(1)
    router = TopKRouter(hidden_size=64, num_experts=32, num_experts_per_tok=4,
                        norm_topk_prob=norm_topk_prob, fused_softmax_topk=True).to(DEVICE)
    router.weight.data.normal_()
    x = torch.randn(9, 64, device=DEVICE)
    w, ids, state = router(x)
    assert state is None and ids.dtype == torch.int64 and w.dtype == torch.float32
    logits = F.linear(x, router.weight)
    want_w, want_ids = _torch_route(logits, 4, norm_topk_prob)
    assert torch.equal(logits.gather(1, ids), logits.gather(1, want_ids))
    torch.testing.assert_close(w, want_w, rtol=1e-6, atol=1e-7)


def test_compiles_whole():
    from mstar.utils.fused_moe.decode import router_topk

    def route(logits):
        return router_topk(logits, None, 8, 1.0, True, softmax=True)

    logits = torch.randn(17, 128, device=DEVICE, dtype=torch.bfloat16)
    compiled = torch.compile(route, fullgraph=True, dynamic=True)
    for got, want in zip(compiled(logits), route(logits), strict=True):
        assert torch.equal(got, want)
