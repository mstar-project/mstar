"""FlashInferCrossManager picks its wrapper kind from the query spans and
skips re-planning an unchanged step.

CPU only: the FlashInfer wrapper classes are replaced by recorders.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn import cross as cross_mod
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.cross import FlashInferCrossManager
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.step import BucketKey, SlotLease

PAGE = 16


class _Wrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.plans = []

    def plan(self, **kwargs):
        self.plans.append(kwargs)


class _Decode(_Wrapper):
    pass


class _Prefill(_Wrapper):
    pass


@pytest.fixture(autouse=True)
def _recorders(monkeypatch):
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setattr(cross_mod, "FlashInferDecodeWrapper", _Decode)
    monkeypatch.setattr(cross_mod, "FlashInferPrefillWrapper", _Prefill)


def _manager():
    kv = PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=64, max_seq_len=256,
                       max_num_pages=16, page_size=PAGE)
    return FlashInferCrossManager(
        kv_cache="ctx", query_kv_cache="kv", context_label="context",
        device=torch.device("cpu"), dtype=torch.bfloat16, kv_config=kv,
    )


def _view(rid, pages, length):
    return SimpleNamespace(
        request_id=rid, length=length, page_idxs=pages,
        last_page_len=lambda page_size, n=length: (n - 1) % page_size + 1,
    )


def _context(pages=((1,), (2,))):
    return SimpleNamespace(views=[_view(r, list(p), 10) for r, p in zip("ab", pages, strict=True)])


def _ctx(spans, context, lease=None, force_prefill=False):
    qo = [0]
    for s in spans:
        qo.append(qo[-1] + s)
    query = SimpleNamespace(
        views=[SimpleNamespace(request_id=r) for r in "ab"],
        cpu_indptrs=SimpleNamespace(qo_indptr=torch.tensor(qo, dtype=torch.int32)),
    )
    return SimpleNamespace(
        slot_lease=lease, is_preplan=False, force_prefill=force_prefill, slot=0,
        plan_results={"ctx": {"context": context}, "kv": {"text": query}},
    )


def _planned(manager):
    return manager._current_plan_states["text"]


@pytest.mark.parametrize("spans,kind", [
    ([1, 1], _Decode),
    # two tokens over two rows, but not one each: the prefill kernel
    ([2, 0], _Prefill),
    ([3, 2], _Prefill),
])
def test_wrapper_kind_follows_the_spans(spans, kind):
    manager = _manager()
    manager.plan(AttentionStep(), _ctx(spans, _context()))
    assert type(_planned(manager)) is kind


def test_force_prefill_takes_the_prefill_kernel():
    manager = _manager()
    manager.plan(AttentionStep(), _ctx([1, 1], _context(), force_prefill=True))
    assert type(_planned(manager)) is _Prefill


def test_an_unchanged_step_is_not_planned_again():
    manager = _manager()
    context = _context()
    manager.plan(AttentionStep(), _ctx([1, 1], context))
    wrapper = _planned(manager)
    manager.plan(AttentionStep(), _ctx([1, 1], context))
    assert _planned(manager) is wrapper and len(wrapper.plans) == 1
    # a new context plan over the same pages still skips FlashInfer's plan
    manager.plan(AttentionStep(), _ctx([1, 1], _context()))
    assert len(wrapper.plans) == 1
    # other pages plan again
    manager.plan(AttentionStep(), _ctx([1, 1], _context(((1,), (3,)))))
    assert len(wrapper.plans) == 2


def test_a_captured_key_keeps_its_kind():
    manager = _manager()
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=2, num_tokens=2))
    manager.plan(AttentionStep(), _ctx([1, 1], _context(), lease=lease))
    assert type(_planned(manager)) is _Decode
    with pytest.raises(AssertionError, match="captured as decode"):
        manager.plan(AttentionStep(), _ctx([2, 0], _context(), lease=lease))
