"""``KVManager.rewind``: take committed tokens back off a stream.

Speculative decoding commits the k+1 verify rows of a step and keeps only the
accepted prefix. The pages stay (``page_indices`` is a high-water mark), the
next step overwrites the freed slots in place, and its plan sees the shorter
stream.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import Segment, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager

PAGE_SIZE = 4


class _StubTransferManager:
    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _make_manager(max_num_pages: int = 8) -> KVManager:
    cfg = KVConfig(
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


def _step(mgr: KVManager, span: int, commit: bool = True):
    step = KVStep(segments=(Segment("r", "main", span),), commit=commit)
    ctx = _ctx()
    assert mgr.admit(step, ctx).ok
    out = mgr.plan(step, ctx)
    mgr.commit(step, ctx)
    return out["main"].views[0]


def test_rewind_shortens_the_stream_and_keeps_its_pages():
    mgr = _make_manager()
    _step(mgr, 6)  # two pages, stored_len 6
    pages = list(mgr._streams["r"]["main"].page_indices)
    assert mgr.stored_len("r") == 6
    gen = mgr._streams["r"]["main"].generation

    mgr.rewind("r", 4)
    assert mgr.stored_len("r") == 2
    assert mgr._streams["r"]["main"].page_indices == pages
    assert mgr._streams["r"]["main"].generation == gen + 1
    assert mgr._arena.num_free == 8 - 1 - len(pages)  # sink + the two

    # the next step plans from the shortened length: one page, three tokens
    view = _step(mgr, 1)
    assert (view.length, view.to_compute) == (3, 1)
    assert view.page_idxs == pages[:1]
    assert mgr.stored_len("r") == 3


def test_rewind_zero_is_a_no_op_and_negative_or_past_zero_raise():
    mgr = _make_manager()
    _step(mgr, 3)
    mgr.rewind("r", 0)
    assert mgr.stored_len("r") == 3
    with pytest.raises(ValueError):
        mgr.rewind("r", -1)
    with pytest.raises(ValueError):
        mgr.rewind("r", 4)
    assert mgr.stored_len("r") == 3


def test_rewind_then_regrow_reuses_the_same_pages():
    mgr = _make_manager()
    _step(mgr, 5)  # pages A, B
    pages = list(mgr._streams["r"]["main"].page_indices)
    mgr.rewind("r", 3)  # back to 2, page B kept
    view = _step(mgr, 4)  # 2 + 4 = 6 -> still two pages, no allocation
    assert view.page_idxs == pages
    assert mgr._streams["r"]["main"].page_indices == pages
    assert mgr.stored_len("r") == 6


def test_uncommitted_step_leaves_the_length_alone():
    mgr = _make_manager()
    _step(mgr, 2)
    view = _step(mgr, 3, commit=False)
    # the plan covered 5 tokens (pages reserved), the stream still holds 2
    assert (view.length, view.to_compute) == (5, 3)
    assert mgr.stored_len("r") == 2
    assert len(mgr._streams["r"]["main"].page_indices) == 2
