"""``KVManager.has_room``: whether a row's tokens fit before its step is built."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 16


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 8) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _grow(kv: KVManager, rid: str, span: int) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk="w", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.commit(step, ctx)


def test_a_row_fits_while_the_free_pages_cover_it():
    kv = _manager()
    kv.ingest_request("r0")
    free = kv._arena.num_free

    assert kv.has_room("r0", "node", "w", free * PAGE_SIZE)
    assert not kv.has_room("r0", "node", "w", free * PAGE_SIZE + 1)


def test_tokens_already_in_a_held_page_need_no_new_one():
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 1)
    kv._arena.acquire(kv._arena.num_free)  # the pool is now full

    assert kv.has_room("r0", "node", "w", PAGE_SIZE - 1)
    assert not kv.has_room("r0", "node", "w", PAGE_SIZE)


def test_an_unknown_request_is_not_refused():
    assert _manager().has_room("ghost", "node", "w", 10_000)
