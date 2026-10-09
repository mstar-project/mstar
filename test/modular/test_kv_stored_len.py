"""``KVManager.stored_len``: the committed length of a stream."""

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


def _make_manager(max_num_pages: int = 8) -> KVManager:
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4,
        max_seq_len=max_num_pages * PAGE_SIZE, max_num_pages=max_num_pages,
        page_size=PAGE_SIZE,
    )
    mgr = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    mgr.ingest_request("r")
    return mgr


def _step(mgr: KVManager, span: int, commit: bool = True):
    step = KVStep(segments=(Segment("r", "main", span),), commit=commit)
    ctx = StepContext(request_ids=("r",), graph_walk="decode", slot=0, capture=False)
    assert mgr.admit(step, ctx).ok
    out = mgr.plan(step, ctx)
    mgr.commit(step, ctx)
    return out["main"].views[0]


def test_stored_len_follows_committed_steps():
    mgr = _make_manager()
    _step(mgr, 6)
    assert mgr.stored_len("r") == 6
    view = _step(mgr, 1)
    assert (view.length, view.to_compute) == (7, 1)
    assert mgr.stored_len("r") == 7


def test_uncommitted_step_leaves_the_length_alone():
    mgr = _make_manager()
    _step(mgr, 2)
    view = _step(mgr, 3, commit=False)
    # the plan covered 5 tokens (pages reserved), the stream still holds 2
    assert (view.length, view.to_compute) == (5, 3)
    assert mgr.stored_len("r") == 2
    assert len(mgr._streams["r"]["main"].page_indices) == 2


def test_unknown_request_raises():
    mgr = _make_manager()
    with pytest.raises(KeyError):
        mgr.stored_len("missing")
