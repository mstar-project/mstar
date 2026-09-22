"""A replay's padding rows run against SINK_PAGE and hold no KV pages.

The padding rows of a captured replay carry the template span (one token for
decode), so left to the ordinary admit they take a page each. Freed after the
step, that is a page allocated and released per padding row per step, and on
a near-full arena the allocation fails and the whole batch is held; kept
resident, it is a page per dummy name for good. Flagged on the step context
instead, they reserve nothing and read and write the sink page, which is held
out of circulation for this.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources import Segment, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KVManager allocates on device"
)

PAGE_SIZE = 8


class _StubTransferManager:
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
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _make_manager(max_num_pages: int) -> KVManager:
    cfg = KVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4,
        max_seq_len=max_num_pages * PAGE_SIZE, max_num_pages=max_num_pages,
        page_size=PAGE_SIZE, cpu_offload_pages=0,
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda"), dtype=torch.float32,
    )


def _padded_ctx(real: list[str], padded: list[str]) -> StepContext:
    ctx = StepContext(request_ids=tuple(real), graph_walk="decode", slot=0, capture=False)
    ctx.set_padded_rids(tuple(padded))
    return ctx


@requires_cuda
def test_padding_rows_take_no_pages_and_address_the_sink():
    mgr = _make_manager(max_num_pages=8)
    real = ["a", "b"]
    dummies = [f"__cg_LLM_0_{i}__" for i in range(8)]
    for rid in real + dummies:
        mgr.ingest_request(rid)
    free_before = mgr._arena.num_free
    padded = [*real, *dummies[2:]]
    step = KVStep(segments=tuple(Segment(rid, "main", 1) for rid in padded))
    ctx = _padded_ctx(real, padded)
    assert mgr.admit(step, ctx).ok
    assert mgr._arena.num_free == free_before - 2, "one page per real request, none for padding"
    out = mgr.plan(step, ctx)["main"]
    kv_indices = out.cpu_indptrs.paged_kv_indices.tolist()
    kv_indptr = out.cpu_indptrs.paged_kv_indptr.tolist()
    # rows 2..7 are the padding rows: one page each, all the sink
    for row in range(2, 8):
        pages = kv_indices[kv_indptr[row]:kv_indptr[row + 1]]
        assert pages == [SINK_PAGE], f"row {row} pages {pages}"
    real_pages = kv_indices[kv_indptr[0]:kv_indptr[2]]
    assert SINK_PAGE not in real_pages and len(set(real_pages)) == 2
    mgr.commit(step, ctx)
    for rid in dummies:
        stream = mgr._streams[rid]["main"]
        assert stream.stored_len == 0 and stream.page_indices == []


@requires_cuda
def test_a_full_arena_still_admits_a_padded_step():
    """Two real requests fill a 3-page arena (one page is the sink); a bucket
    of 8 around them must still admit, since padding needs no page."""
    mgr = _make_manager(max_num_pages=3)
    dummies = [f"__cg_LLM_0_{i}__" for i in range(8)]
    for rid in ["a", "b", *dummies]:
        mgr.ingest_request(rid)
    padded = ["a", "b", *dummies[2:]]
    step = KVStep(segments=tuple(Segment(rid, "main", 1) for rid in padded))
    ctx = _padded_ctx(["a", "b"], padded)
    assert mgr.admit(step, ctx).ok
    assert mgr._arena.num_free == 0
    mgr.plan(step, ctx)
    mgr.commit(step, ctx)
