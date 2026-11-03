"""Which router path each platform takes, and the op's CPU reference against
TopKRouter's torch code. The kernel itself is checked on an AMD GPU in
test/rocm/test_moe_router.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.components.moe import TopKRouter
from mstar.platforms.dispatch import PlatformDispatch
from mstar.utils.fused_moe import topk_softmax


def test_amd_gpus_take_the_fused_op_and_the_rest_keep_torch():
    dispatch = topk_softmax.route_topk_softmax
    assert isinstance(dispatch, PlatformDispatch)
    assert dispatch._factories["rocm"] is topk_softmax._fused_router
    assert dispatch._factories["cuda"] is None and dispatch._default is topk_softmax._torch_router
    assert topk_softmax._torch_router() is topk_softmax._softmax_topk


def test_the_op_reference_is_the_routers_torch_code():
    torch.manual_seed(0)
    logits = torch.randn(5, 16)
    w, ids = torch.ops.mstar.moe_topk_softmax(logits, 4, True)
    probs = F.softmax(logits, dtype=torch.float, dim=-1)
    top, top_ids = torch.topk(probs, 4, dim=-1)
    # int32 ids, what the fused MoE reads
    assert ids.dtype == torch.int32 and torch.equal(ids.long(), top_ids)
    assert torch.equal(w, top / top.sum(-1, keepdim=True))
    w, _ = torch.ops.mstar.moe_topk_softmax(logits, 4, False)
    assert torch.equal(w, top)


def test_router_returns_the_contracts_types_on_cpu():
    router = TopKRouter(hidden_size=8, num_experts=16, num_experts_per_tok=4)
    with torch.no_grad():
        router.weight.normal_()
    w, ids, state = router(torch.randn(3, 8))
    fused = router._route is not topk_softmax._softmax_topk
    assert w.shape == ids.shape == (3, 4) and w.dtype == torch.float32
    assert ids.dtype == (torch.int32 if fused else torch.int64)
    assert state is None
    torch.testing.assert_close(w.sum(-1), torch.ones(3))


def test_the_router_picks_its_path_when_built():
    router = TopKRouter(hidden_size=8, num_experts=16, num_experts_per_tok=4)
    assert router._route is topk_softmax.route_topk_softmax.resolve()
