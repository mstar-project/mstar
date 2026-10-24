"""``TopKRouter``'s fused softmax -> top-k -> renormalize (``mstar::moe_router_topk``)
against the op sequence it replaces, and where the swept MoE tiles apply."""
from __future__ import annotations

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@needs_cuda
@pytest.mark.parametrize("tokens", [1, 7, 32, 300, 4096])
@pytest.mark.parametrize("experts,top_k,norm", [(128, 8, True), (128, 6, True), (60, 4, False)])
def test_softmax_topk_matches_the_torch_router(tokens, experts, top_k, norm):
    from mstar.utils.fused_moe.decode import router_topk

    torch.manual_seed(1)
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16) * 3
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    ref_w, ref_ids = torch.topk(probs, top_k, dim=-1)
    if norm:
        ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)

    w, ids = router_topk(logits, None, top_k, 1.0, norm, softmax=True)
    assert ids.dtype == torch.int64 and w.dtype == torch.float32
    # bf16 logits tie often and torch.topk orders ties arbitrarily: compare the
    # selected probabilities, and that every pick is a distinct expert holding them
    torch.testing.assert_close(w, ref_w, rtol=1e-6, atol=1e-7)
    assert (ids.sort(dim=-1).values.diff(dim=-1) > 0).all()
    picked = probs.gather(1, ids)
    if norm:
        picked = picked / picked.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(picked, w, rtol=1e-6, atol=1e-7)
    # with no ties the choice and its order are torch's
    distinct = (ref_w[:, :-1] != ref_w[:, 1:]).all(dim=-1)
    assert torch.equal(ids[distinct], ref_ids[distinct])


@needs_cuda
def test_router_compiles_with_the_fused_topk():
    from mstar.model.components.moe import TopKRouter

    router = TopKRouter(64, 128, 8, fused_softmax_topk=True).cuda().to(torch.bfloat16).requires_grad_(False)
    torch.nn.init.normal_(router.weight)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        eager = router(x)
        compiled = torch.compile(router, fullgraph=True)(x)
    assert torch.equal(eager[1], compiled[1])
    torch.testing.assert_close(eager[0], compiled[0], rtol=0, atol=0)


@needs_cuda
def test_router_defaults_to_the_torch_ops():
    from mstar.model.components.moe import TopKRouter

    torch.manual_seed(2)
    router = TopKRouter(64, 128, 8).cuda().to(torch.bfloat16).requires_grad_(False)
    torch.nn.init.normal_(router.weight)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        w, ids, _ = router(x)
        probs = torch.softmax(torch.nn.functional.linear(x, router.weight), dim=-1, dtype=torch.float)
        ref_w, ref_ids = torch.topk(probs, 8, dim=-1)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w / ref_w.sum(dim=-1, keepdim=True), rtol=0, atol=0)
