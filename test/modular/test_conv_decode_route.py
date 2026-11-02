"""Which conv kernel a decode step takes: AMD GPUs the decode kernel where it fits, every other platform today's."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.linear_attn import wrappers
from mstar.engine.resources.linear_attn.wrappers import GDNDecodeWrapper
from mstar.utils import causal_conv1d
from mstar.utils.causal_conv1d.kernels import decode_kernel_fits


def _wrapper():
    w = GDNDecodeWrapper.__new__(GDNDecodeWrapper)
    w._plan_state = wrappers.GDNDecodePlan(slots=torch.tensor([1, 2]), num_rows=2)
    w._null_slot_id = 0
    return w


@pytest.mark.parametrize(("rocm", "expected"), [(True, "decode"), (False, "update")])
def test_route(monkeypatch, rocm, expected):
    called = []
    # the module's reference, not an attribute of the shared proxy, which undoing would leave pinned
    monkeypatch.setattr(wrappers, "current_platform", SimpleNamespace(is_rocm=lambda: rocm))
    monkeypatch.setattr(causal_conv1d, "causal_conv1d_decode", lambda **kw: called.append("decode"))
    monkeypatch.setattr(causal_conv1d, "causal_conv1d_update", lambda **kw: called.append("update"))
    _wrapper().run_conv(torch.zeros(2, 8), torch.zeros(3, 8, 3), torch.zeros(8, 4))
    assert called == [expected]


def test_a_plan_with_write_slots_takes_the_decode_kernel(monkeypatch):
    called = {}
    w = _wrapper()
    w._plan_state = wrappers.GDNDecodePlan(slots=torch.tensor([1, 2]), num_rows=2, write_slots=torch.tensor([3, 4]))
    monkeypatch.setattr(causal_conv1d, "causal_conv1d_decode", lambda **kw: called.update(kw))
    w.run_conv(torch.zeros(2, 8), torch.zeros(5, 8, 3), torch.zeros(8, 4))
    assert called["write_indices"].tolist() == [3, 4]


def test_fits():
    x, state, weight = torch.zeros(2, 8), torch.zeros(3, 8, 3), torch.zeros(8, 4)
    assert decode_kernel_fits(x, state, weight)
    # a strided column, another filter width, another state length, a second dtype
    assert not decode_kernel_fits(torch.zeros(2, 16)[:, ::2], state, weight)
    assert not decode_kernel_fits(x, torch.zeros(3, 8, 4), torch.zeros(8, 5))
    assert not decode_kernel_fits(x, torch.zeros(3, 8, 4), weight)
    assert not decode_kernel_fits(x.to(torch.bfloat16), state, weight)
