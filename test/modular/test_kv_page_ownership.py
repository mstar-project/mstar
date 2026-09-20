"""A page goes back to the free list when its last owner lets go.

Today that is the same moment its first owner lets go: a page belongs to one
request, the one whose ``page_indices`` names it, and teardown frees it.
Cross-request prefix reuse adds a second owner, the index, so the rule has to
become "when the count reaches zero" before a page can be shared at all. These
tests cover that rule while every count is still one, which is why none of it
changes what the engine does.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 16


class _StubTransfer:
    """No engine, no bytes moved: retrieves complete immediately."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 16) -> KVManager:
    """A manager on the CPU. ``cpu_offload_pages`` stays 0: the host pool pins
    its memory as it is built, and pinning needs a GPU even though this manager
    does not."""
    return KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids: str) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk="w", slot=0, capture=False,
    )


def _grow(kv: KVManager, rid: str, span: int, label: str = "main") -> None:
    """Admit and commit one step extending ``label`` by ``span`` tokens."""
    step = KVStep(segments=(Segment(rid, label, span),))
    ctx = _ctx(rid)
    assert kv.admit(step, ctx).ok
    kv.commit(step, ctx)


# ── the count ───────────────────────────────────────────────────────────


def test_acquire_takes_pages_at_one_owner():
    kv = _manager()

    pages = kv._arena.acquire(3)

    assert [kv._arena.num_owners[page] for page in pages] == [1, 1, 1]


def test_retain_adds_an_owner():
    kv = _manager()
    pages = kv._arena.acquire(2)

    kv._arena.retain(pages)

    assert [kv._arena.num_owners[page] for page in pages] == [2, 2]


def test_a_page_with_an_owner_left_stays_off_the_free_list():
    kv = _manager()
    free = kv._arena.num_free
    pages = kv._arena.acquire(1)
    kv._arena.retain(pages)

    kv._arena.release(pages)

    assert kv._arena.num_owners[pages[0]] == 1
    assert kv._arena.num_free == free - 1, "a page a second owner still reads was freed"


def test_a_page_released_to_zero_is_back_on_the_free_list():
    kv = _manager()
    free = kv._arena.num_free
    pages = kv._arena.acquire(1)
    kv._arena.retain(pages)
    kv._arena.release(pages)

    kv._arena.release(pages)

    assert kv._arena.num_owners[pages[0]] == 0
    assert kv._arena.num_free == free, "the last owner let go and the page stayed out"


def test_the_sink_page_keeps_its_one_owner():
    """It is acquired once at construction and never released."""
    kv = _manager()

    assert kv._arena.num_owners[SINK_PAGE] == 1


# ── through the manager ─────────────────────────────────────────────────


def test_an_admitted_request_owns_each_of_its_pages_once():
    kv = _manager()
    kv.ingest_request("r0")

    _grow(kv, "r0", 100)

    pages = kv._streams["r0"]["main"].page_indices
    assert [kv._arena.num_owners[page] for page in pages] == [1] * len(pages), (
        "nothing calls retain yet, so a request is the only owner of its pages"
    )
