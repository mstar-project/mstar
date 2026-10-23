"""``KVManager`` write addressing built on the host equals the packed path's."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import Segment, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager

PAGE_SIZE = 4
RIDS = ("a", "b", "c")


class _StubTransferManager:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _make_manager() -> KVManager:
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4,
        max_seq_len=64 * PAGE_SIZE, max_num_pages=64, page_size=PAGE_SIZE,
    )
    mgr = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    for rid in RIDS:
        mgr.ingest_request(rid)
    return mgr


def _step(mgr: KVManager, spans: tuple[int, ...]):
    step = KVStep(segments=tuple(Segment(r, "main", s) for r, s in zip(RIDS, spans, strict=True)))
    ctx = StepContext(request_ids=RIDS, graph_walk="decode", slot=0, capture=False)
    assert mgr.admit(step, ctx).ok
    out = mgr.plan(step, ctx)
    state = mgr._current_plan_states["main"]
    mgr.commit(step, ctx)
    return out["main"], state


@pytest.mark.parametrize("spans", [(1, 1, 1), (4, 4, 4), (3, 5, 2), (1, 4, 7)])
def test_host_plan_matches_packed_path(spans):
    mgr = _make_manager()
    _step(mgr, (3, 6, 8))  # resident lengths off the page boundary, on it, past it
    out, state = _step(mgr, spans)

    packed = mgr._compute_plan_state(
        out.cpu_indptrs.to_device(torch.device("cpu")), total_tokens=sum(spans))

    assert state.total_tokens == sum(spans)
    assert torch.equal(state.token_to_page, packed.token_to_page)
    assert torch.equal(state.token_to_cache, packed.token_to_cache)


def test_small_multi_token_steps_skip_the_packed_path(monkeypatch):
    mgr = _make_manager()
    calls = []
    real = mgr._compute_plan_state
    monkeypatch.setattr(mgr, "_compute_plan_state", lambda *a, **k: calls.append(1) or real(*a, **k))

    _step(mgr, (4, 4, 4))  # an MTP verify step: k + 1 rows per request
    assert not calls
    monkeypatch.setattr(manager_mod, "HOST_PLAN_MAX_TOKENS", 4)
    _step(mgr, (2, 2, 2))
    assert calls
