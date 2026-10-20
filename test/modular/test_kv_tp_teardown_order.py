"""Every rank of a TP instance has to free a request's pages where rank 0 did.

Rank 0 decides each step and every rank admits it against its own allocator, so the
allocators have to stay in step. A teardown that lands on the other side of an admit from
rank 0's changes what that admit finds free; one that lands in the same gap but in another
order changes which pages come back out, as the free list hands them out in the order they
were freed. The worker keeps both (``test_tp_async_teardown_order``); this shows what each
slip costs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402
import torch  # noqa: E402

from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402

PAGE_SIZE = 16
WALK = "prefill"


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(rank: int, max_num_pages: int) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096, num_qo_heads=4,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=types.SimpleNamespace(rank=rank, world_size=2),
        transfer_engine_info=None, device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(rid: str) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)


def _run(kv: KVManager, rid: str, span: int) -> bool:
    """One prefill step of ``span`` tokens: admit, plan, commit. Whether it was admitted."""
    kv.ingest_request(rid)
    step = KVStep(segments=(Segment(rid, "main", span),))
    if not kv.admit(step, _ctx(rid)).ok:
        return False
    kv.plan(step, _ctx(rid))
    kv.commit(step, _ctx(rid))
    return True


def _pages(kv: KVManager, rid: str) -> list[int]:
    return kv._streams[rid]["main"].page_indices


def test_ranks_that_tear_down_on_either_side_of_an_admit_disagree_on_whether_it_fits():
    """The pool is full while ``r`` runs. Rank 0 tears ``r`` down before it admits ``s``,
    the other rank after: rank 0's ``s`` takes pages ``r`` let go, and the other rank's
    admit is refused for a step rank 0 runs. With room to spare they would agree until the
    free list wraps round to ``r``'s pages, and differ from there. So a teardown has to land
    in the same gap on every rank."""
    ranks = leader, follower = _manager(0, max_num_pages=4), _manager(1, max_num_pages=4)
    for kv in ranks:
        assert _run(kv, "r", 2 * PAGE_SIZE + 1)
        assert kv._arena.num_free == 0
    let_go = list(_pages(leader, "r"))
    leader.remove_request("r")
    assert _run(leader, "s", PAGE_SIZE + 1)
    assert set(_pages(leader, "s")) <= set(let_go)
    assert not _run(follower, "s", PAGE_SIZE + 1)


def test_ranks_that_free_two_requests_in_either_order_hand_out_other_pages_once_the_pool_wraps():
    """The free list hands pages back out in the order they were freed. Two teardowns in
    one gap, landed in either order, leave the ranks agreeing until the allocator reaches
    the pages they freed; the step that does takes other pages. The counts still agree, so
    nothing is refused: it costs only where ranks share page ids."""
    ranks = leader, follower = _manager(0, max_num_pages=9), _manager(1, max_num_pages=9)
    for kv in ranks:
        for rid in ("a", "b"):
            assert _run(kv, rid, 2 * PAGE_SIZE + 1)
    for rid in ("a", "b"):
        leader.remove_request(rid)
    for rid in ("b", "a"):
        follower.remove_request(rid)

    for kv in ranks:
        for rid in ("fresh", "wrapped"):
            assert _run(kv, rid, PAGE_SIZE + 1)
    assert _pages(leader, "fresh") == _pages(follower, "fresh"), "pages nobody freed"
    assert _pages(leader, "wrapped") != _pages(follower, "wrapped")
    assert leader._arena.num_free == follower._arena.num_free
