"""GPU golden for the one-kernel softmax top-k router (``mstar.utils.fused_moe.router``)."""
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


@pytest.mark.parametrize("tokens", [1, 8, 33, 1000])
@pytest.mark.parametrize("experts,top_k", [(128, 8), (64, 6), (60, 4), (512, 10)])
@pytest.mark.parametrize("renormalize", [True, False])
def test_softmax_topk_is_softmax_then_topk(tokens, experts, top_k, renormalize):
    from mstar.utils.fused_moe.router import softmax_topk

    torch.manual_seed(tokens + experts)
    logits = (torch.randn(tokens, experts, device=DEVICE) * 2).to(torch.bfloat16)
    want_w, want_ids = _torch_route(logits, top_k, renormalize)
    got_w, got_ids = softmax_topk(logits, top_k, renormalize)
    assert got_ids.dtype == torch.int64 and got_w.dtype == torch.float32
    # descending, as torch.topk; exact ties may pick either expert, so compare the logits picked
    assert torch.equal(logits.gather(1, got_ids), logits.gather(1, want_ids))
    torch.testing.assert_close(got_w, want_w, rtol=1e-6, atol=1e-7)


def test_strided_rows_and_empty_batch():
    from mstar.utils.fused_moe.router import softmax_topk

    logits = torch.randn(6, 256, device=DEVICE, dtype=torch.bfloat16)[:, :128]
    want_w, want_ids = _torch_route(logits, 8, True)
    got_w, got_ids = softmax_topk(logits, 8, True)
    assert torch.equal(logits.gather(1, got_ids), logits.gather(1, want_ids))
    torch.testing.assert_close(got_w, want_w, rtol=1e-6, atol=1e-7)
    w, ids = softmax_topk(logits[:0], 8, True)
    assert w.shape == (0, 8) and ids.shape == (0, 8)


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_topk_router_on_cuda_takes_the_kernel_and_keeps_the_contract(norm_topk_prob):
    """TopKRouter on CUDA routes through the kernel: fp32 weights, int64 ids, no state, the
    softmax-then-top-k of its own logits."""
    from mstar.model.components.moe import TopKRouter

    torch.manual_seed(1)
    router = TopKRouter(hidden_size=64, num_experts=32, num_experts_per_tok=4,
                        norm_topk_prob=norm_topk_prob).to(DEVICE)
    router.weight.data.normal_()
    x = torch.randn(9, 64, device=DEVICE)
    w, ids, state = router(x)
    assert state is None and ids.dtype == torch.int64 and w.dtype == torch.float32
    logits = F.linear(x, router.weight)
    want_w, want_ids = _torch_route(logits, 4, norm_topk_prob)
    assert torch.equal(logits.gather(1, ids), logits.gather(1, want_ids))
    torch.testing.assert_close(w, want_w, rtol=1e-6, atol=1e-7)


def test_compiles_whole():
    from mstar.utils.fused_moe.router import softmax_topk

    def route(logits):
        return softmax_topk(logits, 8, True)

    logits = torch.randn(17, 128, device=DEVICE, dtype=torch.bfloat16)
    compiled = torch.compile(route, fullgraph=True, dynamic=True)
    for got, want in zip(compiled(logits), route(logits), strict=True):
        assert torch.equal(got, want)
