"""A decoder attending an encoder context it wrote at prefill declares a
zero-span segment on that context every later step. The context does not move,
so its KV plan is reused per replay slot while the step's rows are the same and
no stream of theirs changed, nothing is staged for a label no segment writes,
and cross-attention does not re-plan a wrapper over the same context plan.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn import cross as cross_mod
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs, PagedIndptrs, SequenceView
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

PAGE, CONTEXT = 16, 40  # three pages a context
BUCKET = BucketKey(graph_walk="decode", bs=4, num_tokens=4)
PADDING = {0: (-1, -2), 1: (-3, -4)}  # per replay slot, as `DummyRowPool` keys them


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def cleanup(self):
        pass

    def remove_request(self, rid):
        pass


@pytest.fixture
def kv(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=CONTEXT,
            max_num_pages=1 + 3 * 4, page_size=PAGE,
        ),
        name="ctx", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.build_accelerator_graph_buffers([], max_bs=4, max_seq_len=4)
    for rids in PADDING.values():
        for rid in rids:
            kv.ingest_capture_row(rid)
    return kv


def _prefill(kv: KVManager, rids) -> None:
    for rid in rids:
        kv.ingest_request(rid, KVReqConfig())
    step = KVStep(segments=tuple(Segment(rid, "main", CONTEXT) for rid in rids))
    ctx = StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _read(kv: KVManager, rids, slot: int, preplan: bool = False):
    """One leased read-only step over ``rids``, padded with the slot's rows:
    admit, plan, commit, and the padding rows released as a replay does.
    Returns the plan, or the staged plan when ``preplan``."""
    padded = (*rids, *PADDING[slot][:BUCKET.bs - len(rids)])
    step = KVStep(segments=tuple(Segment(rid, "main", 0) for rid in padded))
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk="decode", slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=BUCKET), is_preplan=preplan,
    )
    ctx.set_padded_rids(padded)
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    if preplan:
        return res, step, ctx
    kv.commit(step, ctx)
    for rid in padded[len(rids):]:
        kv.reset_request(rid, free=True)
    return res


def _pages(res) -> list[int]:
    return res["main"].cpu_indptrs.paged_kv_indices.tolist()


def test_a_repeated_read_only_step_reuses_its_slots_plan(kv):
    _prefill(kv, [0, 1])
    first = _read(kv, [0, 1], slot=0)
    assert _read(kv, [0, 1], slot=0) is first
    assert kv.readonly_stats == {"hit": 1, "miss": 1}


def test_each_slot_keeps_its_own_plan(kv):
    """Slots alternate and each pads with rows of its own, so one entry for
    the node would miss every step."""
    _prefill(kv, [0, 1])
    plans = [_read(kv, [0, 1], slot=i % 2) for i in range(6)]
    assert plans[2] is plans[0] and plans[4] is plans[0]
    assert plans[3] is plans[1] and plans[5] is plans[1]
    assert kv.readonly_stats == {"hit": 4, "miss": 2}


def test_the_reused_plan_is_the_plan_a_fresh_build_makes(kv):
    _prefill(kv, [0, 1])
    _read(kv, [0, 1], slot=0)
    reused = _read(kv, [0, 1], slot=0)
    kv._drop_readonly_plans()
    fresh = _read(kv, [0, 1], slot=0)
    assert fresh is not reused
    assert [v[:5] for v in fresh["main"].views] == [v[:5] for v in reused["main"].views]
    for a, b in zip(fresh["main"].cpu_indptrs[:4], reused["main"].cpu_indptrs[:4], strict=True):
        assert torch.equal(a, b)
    # three pages for each real row, none for the padding rows
    assert len(_pages(fresh)) == 6


def test_a_new_row_or_a_departed_one_is_a_new_plan(kv):
    _prefill(kv, [0, 1, 2])
    pair = _read(kv, [0, 1], slot=0)
    trio = _read(kv, [0, 1, 2], slot=0)
    assert trio is not pair and len(_pages(trio)) == 9
    assert _read(kv, [0, 1], slot=0) is not trio


def test_a_reused_handle_reads_the_new_requests_context(kv):
    """Handles are recycled: the step's segments are the same, its pages are
    the new request's."""
    _prefill(kv, [0, 1])
    before = _read(kv, [0, 1], slot=0)
    kv.remove_request(1)
    _prefill(kv, [1])
    after = _read(kv, [0, 1], slot=0)
    assert after is not before
    assert _pages(after)[3:] == kv._streams[1]["main"].page_indices[:3]


def test_an_offloaded_stream_comes_back_on_new_pages(kv):
    _prefill(kv, [0, 1])
    before = _read(kv, [0, 1], slot=0)
    kv._touch_plan_cache(1)  # what an offload or a reload does to its streams
    assert _read(kv, [0, 1], slot=0) is not before


def test_releasing_the_padding_rows_keeps_the_plan(kv):
    """Every replay resets its padding rows; that touches no real row."""
    _prefill(kv, [0])
    first = _read(kv, [0], slot=0)
    assert _read(kv, [0], slot=0) is first


def test_a_write_to_a_cached_row_is_a_new_plan(kv):
    _prefill(kv, [0, 1])
    first = _read(kv, [0, 1], slot=0)
    step = KVStep(segments=(Segment(0, "main", 1),))
    ctx = StepContext(request_ids=(0,), graph_walk="other", slot=0, capture=False)
    kv._streams[0]["main"].stored_len -= 1  # room for the token, within its pages
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    assert _read(kv, [0, 1], slot=0) is not first


def test_a_read_only_label_stages_nothing(kv):
    """No segment writes it, so no device addressing is built or copied into
    the captured buffers, whose write the graph baked in as empty."""
    _prefill(kv, [0, 1])
    static = kv._static_plan_state(0, "main")
    static.token_to_page.fill_(7)
    _read(kv, [0, 1], slot=0)
    assert kv._current_plan_states["main"].total_tokens == 0
    assert static.token_to_page.eq(7).all()


def test_a_staged_read_only_plan_is_the_one_promoted(kv):
    _prefill(kv, [0, 1])
    staged, step, ctx = _read(kv, [0, 1], slot=1, preplan=True)
    ctx.is_preplan = False
    assert kv.plan(step, ctx) is staged
    kv.commit(step, ctx)
    assert kv._current_plan_states["main"].total_tokens == 0


# --- cross-attention over the context plan -------------------------------

class _Wrapper:
    plans = 0

    def __init__(self, **kwargs):
        pass

    def plan(self, **kwargs):
        type(self).plans += 1


@pytest.fixture
def cross(monkeypatch):
    _Wrapper.plans = 0
    monkeypatch.setattr(cross_mod, "FlashInferPrefillWrapper", _Wrapper)
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    return cross_mod.FlashInferCrossManager(
        kv_cache="ctx", query_kv_cache="kv", context_label="main",
        device=torch.device("cpu"), dtype=torch.bfloat16,
        kv_config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=64, max_seq_len=CONTEXT,
            max_num_pages=16, page_size=PAGE,
        ),
    )


def _context(pages: dict[int, list[int]], length: int = CONTEXT) -> KVPlanOutput:
    views = [SequenceView(rid, "main", p, length, 0) for rid, p in pages.items()]
    return KVPlanOutput(cpu_indptrs=None, views=views)


def _query(rids) -> KVPlanOutput:
    """A cached decode plan's output: rows, and views it must not be asked for."""
    def unbuilt():
        raise AssertionError("cross-attention built the query side's views")

    return KVPlanOutput(
        cpu_indptrs=PagedIndptrs(
            qo_indptr=torch.arange(len(rids) + 1, dtype=torch.int32),
            paged_kv_indptr=None, paged_kv_indices=None, paged_kv_last_page_len=None,
        ),
        rows=tuple((rid, "main", 1) for rid in rids), view_builder=unbuilt,
    )


def _cross_plan(cross, context: KVPlanOutput, rids=(0, 1)) -> None:
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=BucketKey("decode", len(rids), len(rids))),
    )
    ctx.plan_results = {
        "ctx": KVPlanOutputs({"main": context}),
        "kv": KVPlanOutputs({"main": _query(rids)}),
    }
    cross.plan(AttentionStep(causal=False), ctx)


def test_the_same_context_plan_over_the_same_rows_plans_nothing(cross):
    context = _context({0: [1, 2, 3], 1: [4, 5, 6]})
    _cross_plan(cross, context)
    _cross_plan(cross, context)
    assert _Wrapper.plans == 1


def test_a_rebuilt_context_plan_over_the_same_pages_skips_flashinfer(cross):
    _cross_plan(cross, _context({0: [1, 2, 3], 1: [4, 5, 6]}))
    _cross_plan(cross, _context({0: [1, 2, 3], 1: [4, 5, 6]}))
    assert _Wrapper.plans == 1


def test_other_pages_are_planned(cross):
    _cross_plan(cross, _context({0: [1, 2, 3], 1: [4, 5, 6]}))
    _cross_plan(cross, _context({0: [1, 2, 3], 1: [7, 8, 9]}))
    assert _Wrapper.plans == 2


class _GraphWrapper(_Wrapper):
    """A captured wrapper whose backend schedules off the lengths alone."""
    replans_from_indices = True
    sets = 0

    def set_indices(self, indices):
        type(self).sets += 1


def test_other_pages_of_the_same_lengths_take_only_the_indices(cross, monkeypatch):
    monkeypatch.setattr(cross_mod, "FlashInferPrefillWrapper", _GraphWrapper)
    _GraphWrapper.plans = _GraphWrapper.sets = 0
    _cross_plan(cross, _context({0: [1, 2, 3], 1: [4, 5, 6]}))
    _cross_plan(cross, _context({0: [1, 2, 3], 1: [7, 8, 9]}))
    assert (_GraphWrapper.plans, _GraphWrapper.sets) == (1, 1)
    # another length is another schedule
    _cross_plan(cross, _context({0: [1, 2, 3], 1: [7, 8, 9]}, length=CONTEXT - 1))
    assert (_GraphWrapper.plans, _GraphWrapper.sets) == (2, 1)
