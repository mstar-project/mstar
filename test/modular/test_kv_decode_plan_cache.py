"""The KV plan cache (MSTAR_KV_PLAN_CACHE): a captured decode step planned off
the previous step's plan gives the plan the streams would, page crossings
and padding rows included, and anything else that touches a stream makes the
next plan start from the streams again."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kv_prefix_decode_helpers", pathlib.Path(__file__).with_name("test_kv_prefix_decode.py"),
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

PAGE = 4
WALK = "decode"


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(H.manager_mod, "KVTransferManager", H._StubTransfer)


def _manager(mode: int) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512,
            max_num_pages=256, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._plan_cache_mode = mode
    kv._cg_max_seq_len = 64
    return kv


def _prefill(kv: KVManager, rid: str, n: int) -> None:
    kv.ingest_request(rid, KVReqConfig())
    step = KVStep(segments=(Segment(rid, "main", n),))
    ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _decode(kv: KVManager, rids: tuple[str, ...], slot: int, pad: int = 0, commit: bool = True):
    """One captured decode step over ``rids`` plus ``pad`` padding rows; returns the plan."""
    padded = rids + tuple(f"pad{i}" for i in range(pad))
    step = KVStep(segments=tuple(Segment(r, "main", 1) for r in padded))
    bucket = BucketKey(graph_walk=WALK, bs=len(padded), num_tokens=len(padded))
    ctx = StepContext(
        request_ids=rids, graph_walk=WALK, slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=bucket),
    )
    ctx.set_padded_rids(padded)
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    if commit:
        kv.commit(step, ctx)
    return res, step, ctx


def _flat(res):
    out = {}
    for label, o in res.items():
        ind = o.cpu_indptrs
        out[label] = (
            [tuple(v[:5]) for v in o.views],
            ind.qo_indptr.tolist(), ind.paged_kv_indptr.tolist(), ind.paged_kv_indices.tolist(),
            ind.paged_kv_last_page_len.tolist(), ind.kv_lens.tolist(),
            [v.page_idxs[-1] for v in o.views],
            [(v.last_page_len(PAGE) or PAGE) - 1 for v in o.views],
        )
    return out


def _run(mode: int, steps: int, pad: int):
    kv = _manager(mode)
    _prefill(kv, "a", 5)
    _prefill(kv, "b", 7)
    _prefill(kv, "c", 4)
    plans = []
    for t in range(steps):
        res, _, _ = _decode(kv, ("a", "b", "c"), slot=t % 2, pad=pad)
        plans.append(_flat(res))
        for o in res.values():
            if o.decode_pages is not None:
                assert o.decode_pages.tolist() == plans[-1]["main"][6]
                assert o.decode_offsets.tolist() == plans[-1]["main"][7]
                assert o.is_decode
    return kv, plans


@pytest.mark.parametrize("pad", [0, 2])
def test_cached_decode_plans_match_the_full_ones(pad):
    _, full = _run(0, 12, pad)
    kv, fast = _run(1, 12, pad)
    assert fast == full
    # the first decode step plans from the streams, every later one off it
    assert kv.plan_cache_stats == {"hit": 11, "miss": 0, "mismatch": 0}


def test_verify_mode_sees_no_difference():
    kv, _ = _run(2, 12, 1)
    assert kv.plan_cache_stats["mismatch"] == 0 and kv.plan_cache_stats["hit"] == 11


def test_stream_changes_and_batch_changes_start_from_the_streams_again():
    ref = _manager(0)
    kv = _manager(1)
    for m in (ref, kv):
        _prefill(m, "a", 5)
        _prefill(m, "b", 7)
        _decode(m, ("a", "b"), slot=0)
        _decode(m, ("a", "b"), slot=1)
    assert kv.plan_cache_stats["hit"] == 1
    # a reset of one request: the next plan misses, and matches the reference
    for m in (ref, kv):
        m.reset_request("b")
        _prefill(m, "b", 3)
    r, k = _decode(ref, ("a", "b"), slot=0)[0], _decode(kv, ("a", "b"), slot=0)[0]
    assert _flat(k) == _flat(r) and kv.plan_cache_stats["miss"] == 1
    assert _flat(_decode(kv, ("a", "b"), slot=1)[0]) == _flat(_decode(ref, ("a", "b"), slot=1)[0])
    assert kv.plan_cache_stats["hit"] == 2
    # a different batch (b finished): miss, then hits again
    for m in (ref, kv):
        m.remove_request("b")
    assert _flat(_decode(kv, ("a",), slot=0)[0]) == _flat(_decode(ref, ("a",), slot=0)[0])
    assert _flat(_decode(kv, ("a",), slot=1)[0]) == _flat(_decode(ref, ("a",), slot=1)[0])
    assert kv.plan_cache_stats["miss"] == 2 and kv.plan_cache_stats["hit"] == 3
    # a new request joins: miss (its prefill planned eagerly in between), then hits
    for m in (ref, kv):
        _prefill(m, "d", 9)
    assert _flat(_decode(kv, ("a", "d"), slot=0)[0]) == _flat(_decode(ref, ("a", "d"), slot=0)[0])
    assert _flat(_decode(kv, ("a", "d"), slot=1)[0]) == _flat(_decode(ref, ("a", "d"), slot=1)[0])
    assert kv.plan_cache_stats["miss"] == 3 and kv.plan_cache_stats["hit"] == 4


def test_a_step_that_does_not_commit_or_a_dropped_preplan_misses():
    ref = _manager(0)
    kv = _manager(1)
    for m in (ref, kv):
        _prefill(m, "a", 5)
        _decode(m, ("a",), slot=0)
    # planned off the cache but never committed (the step was dropped): the
    # next plan of the same rows must not assume a commit happened
    for m in (ref, kv):
        _decode(m, ("a",), slot=1, commit=False)
        m.clear_preplan()
    assert kv.plan_cache_stats["hit"] == 1
    assert _flat(_decode(kv, ("a",), slot=1)[0]) == _flat(_decode(ref, ("a",), slot=1)[0])
    assert kv.plan_cache_stats == {"hit": 1, "miss": 1, "mismatch": 0}
    # steady again
    assert _flat(_decode(kv, ("a",), slot=0)[0]) == _flat(_decode(ref, ("a",), slot=0)[0])
    assert kv.plan_cache_stats["hit"] == 2


def test_eager_and_capture_steps_are_never_cached():
    kv = _manager(1)
    _prefill(kv, "a", 5)
    step = KVStep(segments=(Segment("a", "main", 1),))
    ctx = StepContext(request_ids=("a",), graph_walk=WALK, slot=0, capture=False)
    for _ in range(3):
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    assert kv.plan_cache_stats["hit"] == 0 and kv._decode_plan_cache is None


def test_a_prefill_of_another_request_keeps_the_cache():
    """A new request's prefill between two decode steps commits other rows:
    the decode batch's next plan still comes off the cache."""
    ref = _manager(0)
    kv = _manager(1)
    for m in (ref, kv):
        _prefill(m, "a", 5)
        _prefill(m, "b", 7)
        _decode(m, ("a", "b"), slot=0)
        _decode(m, ("a", "b"), slot=1)
        _prefill(m, "c", 6)
    assert _flat(_decode(kv, ("a", "b"), slot=0)[0]) == _flat(_decode(ref, ("a", "b"), slot=0)[0])
    assert kv.plan_cache_stats == {"hit": 2, "miss": 0, "mismatch": 0}
    # c joins: a new batch, planned from the streams, then cached again
    assert _flat(_decode(kv, ("a", "b", "c"), slot=1)[0]) == _flat(_decode(ref, ("a", "b", "c"), slot=1)[0])
    assert _flat(_decode(kv, ("a", "b", "c"), slot=0)[0]) == _flat(_decode(ref, ("a", "b", "c"), slot=0)[0])
    assert kv.plan_cache_stats == {"hit": 3, "miss": 1, "mismatch": 0}
    # a step of the cached rows in another batch (b alone) invalidates
    for m in (ref, kv):
        _decode(m, ("b",), slot=1)
    assert _flat(_decode(kv, ("a", "b", "c"), slot=0)[0]) == _flat(_decode(ref, ("a", "b", "c"), slot=0)[0])
    assert kv.plan_cache_stats["miss"] == 3 and kv.plan_cache_stats["hit"] == 3


def test_padding_rows_may_change_their_dummy_ids_between_steps():
    """A replay's padding rows are the slot's dummy requests, so consecutive
    steps carry different dummy ids with the same layout: the cache keys on
    the real rows and rebuilds the padding part per step."""
    ref = _manager(0)
    kv = _manager(1)
    for m in (ref, kv):
        _prefill(m, "a", 5)
        _prefill(m, "b", 6)
    plans = {0: [], 1: []}
    for t in range(6):
        for mode, m in ((0, ref), (1, kv)):
            rids = ("a", "b")
            padded = rids + (f"slot{t % 2}_pad0", f"slot{t % 2}_pad1")
            step = KVStep(segments=tuple(Segment(r, "main", 1) for r in padded))
            bucket = BucketKey(graph_walk=WALK, bs=4, num_tokens=4)
            ctx = StepContext(request_ids=rids, graph_walk=WALK, slot=t % 2, capture=False,
                              slot_lease=SlotLease(slot=t % 2, bucket=bucket))
            ctx.set_padded_rids(padded)
            assert m.admit(step, ctx).ok
            plans[mode].append(_flat(m.plan(step, ctx)))
            m.commit(step, ctx)
    assert plans[1] == plans[0]
    assert kv.plan_cache_stats == {"hit": 5, "miss": 0, "mismatch": 0}
