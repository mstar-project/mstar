"""``KVManager.rewind``: a verify step's rejected rows taken back off the stream's length."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources import Segment, StepContext
from mstar.engine.resources.draft_verify.config import VerdictBatch
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


def _make_manager(max_num_pages: int = 8, device: str = "cpu", cpu_offload_pages: int = 0) -> KVManager:
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4,
        max_seq_len=max_num_pages * PAGE_SIZE, max_num_pages=max_num_pages,
        page_size=PAGE_SIZE, cpu_offload_pages=cpu_offload_pages,
    )
    mgr = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device(device), dtype=torch.float32,
    )
    mgr.ingest_request("r")
    return mgr


def _ctx() -> StepContext:
    return StepContext(request_ids=("r",), graph_walk="decode", slot=0, capture=False)


def _step(mgr: KVManager, span: int):
    step = KVStep(segments=(Segment("r", "main", span),))
    assert mgr.admit(step, _ctx()).ok
    out = mgr.plan(step, _ctx())
    mgr.commit(step, _ctx())
    return out["main"].views[0]


def _stream(mgr: KVManager):
    return mgr._streams["r"]["main"]


def _rewind(mgr: KVManager, rows: int, label: str = "main", rid: str = "r"):
    mgr.rewind(VerdictBatch([rid], [label], [rows]))


def test_a_rewind_shortens_the_stream_and_keeps_its_pages():
    mgr = _make_manager()
    _step(mgr, 6)  # two pages
    pages = list(_stream(mgr).page_indices)
    gen, content = _stream(mgr).generation, _stream(mgr).reset_generation

    _rewind(mgr, 4)

    assert _stream(mgr).stored_len == 2
    assert _stream(mgr).page_indices == pages
    assert (_stream(mgr).generation, _stream(mgr).reset_generation) == (gen + 1, content + 1)
    assert mgr._arena.num_free == 8 - 1 - len(pages)  # sink + the two
    mgr.assert_pages_conserved()
    # the next step plans from the shortened length
    view = _step(mgr, 1)
    assert (view.length, view.to_compute) == (3, 1)
    assert view.page_idxs == pages[:1]


def test_a_rewound_stream_regrows_over_the_same_pages():
    mgr = _make_manager()
    _step(mgr, 5)
    pages = list(_stream(mgr).page_indices)
    _rewind(mgr, 3)
    view = _step(mgr, 4)  # 2 + 4 = 6: still two pages, no allocation
    assert view.page_idxs == pages
    assert _stream(mgr).page_indices == pages


def test_a_rewind_past_zero_stops_at_zero():
    mgr = _make_manager()
    _step(mgr, 3)
    _rewind(mgr, 5)
    assert _stream(mgr).stored_len == 0


def test_a_rewind_of_no_rows_moves_nothing():
    mgr = _make_manager()
    _step(mgr, 3)
    gen = _stream(mgr).generation
    with mgr._lock:
        mgr._take_back("r", "main", 0)
    assert (_stream(mgr).stored_len, _stream(mgr).generation) == (3, gen)


@pytest.mark.parametrize(("rid", "label"), [("nope", "main"), ("r", "side")])
def test_an_unknown_stream_raises(rid, label):
    mgr = _make_manager()
    with pytest.raises(KeyError, match=label if rid == "r" else rid):
        _rewind(mgr, 1, label=label, rid=rid)


def test_refused_between_admit_and_commit():
    """The admitted step's plan addresses the old length; moving it under the
    step would put the step's writes and the committed length out of step."""
    mgr = _make_manager()
    _step(mgr, 5)
    step = KVStep(segments=(Segment("r", "main", 1),))
    assert mgr.admit(step, _ctx()).ok
    with pytest.raises(AssertionError, match="between admit and commit"):
        _rewind(mgr, 2)
    assert _stream(mgr).stored_len == 5
    mgr.plan(step, _ctx())
    mgr.commit(step, _ctx())
    _rewind(mgr, 2)
    assert _stream(mgr).stored_len == 4


def test_a_rewind_across_a_sealed_page_drops_it():
    mgr = _make_manager()
    _step(mgr, 12)  # three full pages
    pages = list(_stream(mgr).page_indices)
    mgr._arena.retain(pages[1:2])  # a second owner reads page 1
    mgr._arena.seal(pages[1:2])
    free = mgr._arena.num_free

    _rewind(mgr, 8)  # back to 4, the end of page 0

    assert _stream(mgr).page_indices == pages[:1]
    assert mgr._arena.num_owners[pages[1]] == 1, "the other owner keeps its page"
    assert mgr._arena.num_free == free + 1, "the unsealed page 2 is freed with it"
    view = _step(mgr, 1)
    assert view.page_idxs[0] == pages[0]
    assert view.page_idxs[1] not in pages, "the next write landed on a sealed page"


def test_a_rewind_ending_inside_a_sealed_page_keeps_a_copy():
    mgr = _make_manager()
    _step(mgr, 8)
    pages = list(_stream(mgr).page_indices)
    mgr.layer_view(0)[pages[1]].fill_(7.0)
    mgr._arena.retain(pages[1:])
    mgr._arena.seal(pages[1:])

    _rewind(mgr, 2)  # back to 6, inside page 1

    copy = _stream(mgr).page_indices[1]
    assert copy != pages[1] and not mgr._arena.sealed[copy]
    assert mgr.layer_view(0)[copy].eq(7.0).all(), "the kept rows were not copied"
    assert mgr._arena.num_owners[pages[1]] == 1
    assert _stream(mgr).stored_len == 6


def test_a_rewind_above_the_sealed_pages_keeps_them():
    mgr = _make_manager()
    _step(mgr, 10)
    pages = list(_stream(mgr).page_indices)
    mgr._arena.retain(pages[:1])
    mgr._arena.seal(pages[:1])

    _rewind(mgr, 3)  # back to 7, page 0 untouched

    assert _stream(mgr).page_indices == pages


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs pinned host memory")
def test_an_offloaded_stream_rewinds_its_host_copy():
    """A request evicted between its verify step and the next one: the reload restores the
    host copy's length, so the rewind has to land there."""
    mgr = _make_manager(device="cuda", cpu_offload_pages=8)
    _step(mgr, 7)
    assert mgr.offload("r") > 0
    assert _stream(mgr).offloaded

    _rewind(mgr, 3)

    assert mgr._cpu_pool.offloaded["r"]["main"].stored_len == 4
    assert mgr.reload("r")
    assert _stream(mgr).stored_len == 4
    view = _step(mgr, 1)
    assert (view.length, view.to_compute) == (5, 1)
