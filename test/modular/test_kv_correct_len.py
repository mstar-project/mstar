"""``KVManager.correct_len``: a verify step's rejected tail taken back."""

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


def test_correct_len_shortens_the_stream_and_keeps_its_pages():
    mgr = _make_manager()
    _step(mgr, 6)  # two pages
    pages = list(_stream(mgr).page_indices)
    gen = _stream(mgr).generation

    mgr.correct_len("r", "main", -4)

    assert mgr.stored_len("r") == 2
    assert _stream(mgr).page_indices == pages
    assert _stream(mgr).generation == gen + 1
    assert mgr._arena.num_free == 8 - 1 - len(pages)  # sink + the two
    mgr.assert_pages_conserved()
    # the next step plans from the shortened length
    view = _step(mgr, 1)
    assert (view.length, view.to_compute) == (3, 1)
    assert view.page_idxs == pages[:1]


def test_correct_len_then_regrow_reuses_the_same_pages():
    mgr = _make_manager()
    _step(mgr, 5)
    pages = list(_stream(mgr).page_indices)
    mgr.correct_len("r", "main", -3)
    view = _step(mgr, 4)  # 2 + 4 = 6: still two pages, no allocation
    assert view.page_idxs == pages
    assert _stream(mgr).page_indices == pages


def test_zero_is_a_no_op_and_a_rewind_past_zero_clamps():
    mgr = _make_manager()
    _step(mgr, 3)
    gen = _stream(mgr).generation
    mgr.correct_len("r", "main", 0)
    assert (mgr.stored_len("r"), _stream(mgr).generation) == (3, gen)
    mgr.correct_len("r", "main", -5)
    assert mgr.stored_len("r") == 0


@pytest.mark.parametrize("delta", [0, -1])
def test_unknown_stream_raises(delta):
    mgr = _make_manager()
    with pytest.raises(KeyError, match="nope"):
        mgr.correct_len("nope", "main", delta)
    with pytest.raises(KeyError, match="side"):
        mgr.correct_len("r", "side", delta)


def test_refused_between_admit_and_commit():
    """The admitted step's plan addresses the old length; moving it under the
    step would put the step's writes and the committed length out of step."""
    mgr = _make_manager()
    _step(mgr, 5)
    step = KVStep(segments=(Segment("r", "main", 1),))
    assert mgr.admit(step, _ctx()).ok
    with pytest.raises(AssertionError, match="between admit and commit"):
        mgr.correct_len("r", "main", -2)
    assert mgr.stored_len("r") == 5
    mgr.plan(step, _ctx())
    mgr.commit(step, _ctx())
    mgr.correct_len("r", "main", -2)
    assert mgr.stored_len("r") == 4


def test_a_rewind_across_a_sealed_page_drops_it():
    mgr = _make_manager()
    _step(mgr, 12)  # three full pages
    pages = list(_stream(mgr).page_indices)
    mgr._arena.retain(pages[1:2])  # a second owner reads page 1
    mgr._arena.seal(pages[1:2])
    free = mgr._arena.num_free

    mgr.correct_len("r", "main", -8)  # back to 4, the end of page 0

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

    mgr.correct_len("r", "main", -2)  # back to 6, inside page 1

    copy = _stream(mgr).page_indices[1]
    assert copy != pages[1] and not mgr._arena.sealed[copy]
    assert mgr.layer_view(0)[copy].eq(7.0).all(), "the kept rows were not copied"
    assert mgr._arena.num_owners[pages[1]] == 1
    assert mgr.stored_len("r") == 6


def test_a_rewind_above_the_sealed_pages_keeps_them():
    mgr = _make_manager()
    _step(mgr, 10)
    pages = list(_stream(mgr).page_indices)
    mgr._arena.retain(pages[:1])
    mgr._arena.seal(pages[:1])

    mgr.correct_len("r", "main", -3)  # back to 7, page 0 untouched

    assert _stream(mgr).page_indices == pages
