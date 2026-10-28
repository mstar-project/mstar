"""Tests for the injectable / state-aware MoE router interface.

These exercise the CPU naive-dispatch path (no CUDA required), covering:

* the stateless :class:`TopKRouter` 3-tuple contract,
* injecting a custom *stateful* router into a block,
* the opt-in ``return_router_states`` flag threading state in and out,
* the default bare-tensor return staying backward compatible,
* :class:`NoAuxTcRouter`'s routing, logit scale and split fp32 weight.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components import SparseMoeBlock, TopKRouter
from mstar.model.loader.base import load_weights_into


class _StatefulRouter(nn.Module):
    """Minimal stateful router matching the block router contract.

    ``router(x, router_states=None) -> (weights, experts, states_next)``.
    The next state is the running sum of the inputs it has seen, so a
    caller threading state across calls sees it accumulate.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.02)
        self.calls = 0

    def forward(self, hidden_states, router_states=None):
        self.calls += 1
        logits = F.linear(hidden_states, self.weight)
        probs = F.softmax(logits.float(), dim=-1)
        weights, experts = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        base = router_states if router_states is not None else torch.zeros_like(hidden_states)
        router_states_next = base + hidden_states
        return weights.to(hidden_states.dtype), experts.to(torch.int64), router_states_next


def _make_block(router=None):
    return SparseMoeBlock(
        hidden_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        norm_topk_prob=True,
        router=router,
    )


def test_topk_router_contract():
    """Stateless router returns (weights, experts, None), ignoring state."""
    router = TopKRouter(hidden_size=32, num_experts=8, num_experts_per_tok=2)
    x = torch.randn(5, 32)
    weights, experts, states_next = router(x)
    assert weights.shape == (5, 2)
    assert experts.shape == (5, 2)
    assert states_next is None
    # router_states is accepted and ignored.
    _, _, states_next2 = router(x, router_states=torch.randn(5, 32))
    assert states_next2 is None


def test_default_router_is_topk():
    block = _make_block()
    assert isinstance(block.gate, TopKRouter)


def test_injected_router_is_used():
    router = _StatefulRouter(hidden_size=32, num_experts=8, top_k=2)
    block = _make_block(router=router)
    assert block.gate is router
    x = torch.randn(4, 32)
    block(x)
    assert router.calls == 1


def test_return_router_states_threads_state():
    router = _StatefulRouter(hidden_size=32, num_experts=8, top_k=2)
    block = _make_block(router=router)
    x = torch.randn(4, 32)

    out, state = block(x, return_router_states=True)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    # First call: incoming state is None -> next state is x.
    torch.testing.assert_close(state, x)

    # Thread the state back in: next state accumulates to 2x.
    out2, state2 = block(x, router_states=state, return_router_states=True)
    assert out2.shape == x.shape
    torch.testing.assert_close(state2, 2 * x)


def test_default_return_is_bare_tensor():
    """Without the flag, forward returns a plain tensor (backward compat)."""
    block = _make_block(router=_StatefulRouter(32, 8, 2))
    x = torch.randn(4, 32)
    out = block(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# NoAuxTcRouter
# ---------------------------------------------------------------------------


def _noaux(logit_scale=1.0, scale=1.0, dtype=torch.float32, fp32_weight=True):
    from mstar.model.components.moe import NoAuxTcRouter

    torch.manual_seed(0)
    router = NoAuxTcRouter(
        16, 8, 2, routed_scaling_factor=scale, logit_scale=logit_scale, fp32_weight=fp32_weight,
    )
    w = torch.randn(8, 16).to(dtype)
    load_weights_into(router, [("weight", w)])
    with torch.no_grad():
        router.e_score_correction_bias.copy_(torch.linspace(-0.2, 0.2, 8))
    return router, w


@pytest.mark.parametrize("fp32_weight", [False, True])
def test_noaux_router_picks_by_biased_score_and_weighs_by_the_unbiased_one(fp32_weight):
    router, w = _noaux(logit_scale=0.25, scale=2.0, fp32_weight=fp32_weight)
    x = torch.randn(5, 16).bfloat16()
    weights, ids, state = router(x)
    assert state is None and ids.dtype == torch.int64 and weights.dtype == torch.float32
    # the logit scale applies after the fp32 GEMM
    scores = (x.float() @ w.T * 0.25).sigmoid()
    ref_ids = (scores + router.e_score_correction_bias).topk(2, dim=-1).indices
    assert torch.equal(ids.sort(-1).values, ref_ids.sort(-1).values)
    picked = scores.gather(1, ids)
    torch.testing.assert_close(weights, picked / picked.sum(-1, keepdim=True) * 2.0)


@pytest.mark.parametrize(("dtype", "parts"), [(torch.float32, 3), (torch.bfloat16, 1)])
def test_noaux_router_keeps_its_checkpoint_weight_through_a_cast(dtype, parts):
    """Under ``fp32_weight``, an fp32 checkpoint weight loads as three bf16
    parts that sum to it to fp32's 24 bits, a bf16 one as itself, and a cast
    leaves the parts and the fp32 bias as they are."""
    router, w = _noaux(dtype=dtype)
    router.to(torch.float64)
    assert router.weight.dtype == torch.bfloat16 and router.e_score_correction_bias.dtype == torch.float32
    assert router.weight.shape == (parts, 8, 16)
    assert ((router.weight.double().sum(0) - w.double()).abs() <= w.double().abs() * 2.0 ** -24).all()
    # the forward reads however many parts there are
    scores = (torch.ones(1, 16).double() @ w.double().T).sigmoid()
    weights, ids, _ = router(torch.ones(1, 16).bfloat16())
    torch.testing.assert_close(weights.double(), scores.gather(1, ids) / scores.gather(1, ids).sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("tokens", [1, 64, 300])
def test_noaux_router_keeps_the_fp32_weight_s_precision_on_cuda(tokens):
    """The parts' bf16 GEMM on CUDA, with the engine's TF32 setting, routes as
    the fp32 weight does in fp64 (TF32 or bf16 would round the weight to 11 or
    8 bits, moving the routing weights ~1e-4)."""
    from mstar.model.components.moe import NoAuxTcRouter

    torch.manual_seed(tokens)
    x = torch.randn(tokens, 6144, device="cuda").bfloat16()
    # every mantissa bit set, as the checkpoint's fp32 router weights
    w = torch.randn(32, 6144) * 0.02
    router = NoAuxTcRouter(6144, 32, 8, routed_scaling_factor=2.5, logit_scale=0.25, fp32_weight=True)
    load_weights_into(router, [("weight", w)])
    with torch.no_grad():
        router.e_score_correction_bias.normal_(0, 0.5)
    router.cuda()
    scores = (x.double() @ w.double().cuda().T * 0.25).float().sigmoid()
    ref_ids = (scores + router.e_score_correction_bias).topk(8, dim=-1).indices
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        weights, ids, _ = router(x)
    finally:
        torch.set_float32_matmul_precision(prev)
    ids, order = ids.sort(-1)
    assert torch.equal(ids, ref_ids.sort(-1).values)
    picked = scores.gather(1, ids)
    torch.testing.assert_close(weights.gather(1, order), picked / picked.sum(-1, keepdim=True) * 2.5,
                               rtol=1e-5, atol=1e-6)


def test_noaux_router_refuses_a_weight_of_another_shape():
    from mstar.model.components.moe import NoAuxTcRouter

    router = NoAuxTcRouter(16, 8, 2, routed_scaling_factor=1.0, fp32_weight=True)
    with pytest.raises(ValueError, match="router weight"):
        load_weights_into(router, [("weight", torch.randn(1, 16))])


def test_noaux_router_plugs_into_a_block():
    block = SparseMoeBlock(16, 8, 2, 32, router=_noaux()[0]).bfloat16()
    assert block(torch.randn(3, 16).bfloat16()).shape == (3, 16)
