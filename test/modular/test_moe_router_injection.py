"""Tests for the injectable / state-aware MoE router interface.

These exercise the CPU naive-dispatch path (no CUDA required), covering:

* the stateless :class:`TopKRouter` 3-tuple contract,
* injecting a custom *stateful* router into a block,
* the opt-in ``return_router_states`` flag threading state in and out,
* the default bare-tensor return staying backward compatible.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components import SparseMoeBlock, TopKRouter


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
