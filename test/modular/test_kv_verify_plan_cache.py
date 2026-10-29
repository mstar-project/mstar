"""The KV plan cache for verify steps: a captured step of a block of ``k + 1`` tokens a row,
planned off the previous step's plan after a rewind took the rejected rows back, gives the
plan and write slots the streams would, page crossings both ways and padding rows included."""
from __future__ import annotations

import importlib.util
import pathlib
import random

import pytest
import torch

from mstar.engine.resources.draft_verify.config import VerdictBatch
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

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
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=1024,
            max_num_pages=512, page_size=PAGE,
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


def _ctx(rids, padded, slot, span, preplan=False):
    bucket = BucketKey(graph_walk=WALK, bs=len(padded), num_tokens=len(padded) * span)
    ctx = StepContext(
        request_ids=rids, graph_walk=WALK, slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=bucket),
    )
    ctx.set_padded_rids(padded)
    ctx.is_preplan = preplan
    return ctx


def _verify(kv: KVManager, rids, slot: int, span: int, pad: int = 0, preplan: bool = False,
            verify: bool = True):
    """One captured verify step (pre-planned and promoted when ``preplan``), committed."""
    padded = tuple(rids) + tuple(f"pad{i}" for i in range(pad))
    step = KVStep(segments=tuple(Segment(r, "main", span) for r in padded), verify=verify and span > 1)
    if preplan:
        ctx = _ctx(tuple(rids), padded, slot, span, preplan=True)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
    ctx = _ctx(tuple(rids), padded, slot, span)
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    slots = _slots(kv, len(rids) * span)
    kv.commit(step, ctx)
    return res, slots


def _slots(kv: KVManager, n: int):
    state = kv._current_plan_states["main"]
    return state.token_to_page[:n].tolist(), state.token_to_cache[:n].tolist()


def _flat(res):
    o = res["main"]
    ind = o.cpu_indptrs
    return (
        [tuple(v[:5]) + (v.generation,) for v in o.views],
        ind.qo_indptr.tolist(), ind.paged_kv_indptr.tolist(), ind.paged_kv_indices.tolist(),
        ind.paged_kv_last_page_len.tolist(), ind.kv_lens.tolist(),
    )


def _run(mode: int, steps: int, pad: int, k: int, seed: int = 0, preplan: bool = False):
    rng = random.Random(seed)
    kv = _manager(mode)
    rids = ("a", "b", "c")
    for rid, n in zip(rids, (5, 7, 4), strict=True):
        _prefill(kv, rid, n)
    plans = []
    for t in range(steps):
        res, slots = _verify(kv, rids, t % 2, k + 1, pad, preplan=preplan)
        plans.append((_flat(res), slots))
        rejected = [rng.randint(0, k) for _ in rids]
        kept = [(rid, r) for rid, r in zip(rids, rejected, strict=True) if r]
        if kept:
            kv.rewind(VerdictBatch([rid for rid, _ in kept], ["main"] * len(kept), [r for _, r in kept]))
    return kv, plans


@pytest.mark.parametrize("pad", [0, 2])
@pytest.mark.parametrize("k", [1, 3, 4])
def test_cached_verify_plans_match_the_full_ones(pad, k):
    _, full = _run(0, 16, pad, k)
    kv, fast = _run(1, 16, pad, k)
    assert fast == full
    # the first verify step plans from the streams, every later one off it, rewinds and all
    assert kv.plan_cache_stats == {"hit": 15, "miss": 0, "mismatch": 0}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_verify_mode_sees_no_difference(seed):
    kv, _ = _run(2, 16, 1, 3, seed=seed)
    assert kv.plan_cache_stats == {"hit": 15, "miss": 0, "mismatch": 0}


@pytest.mark.parametrize("preplan", [False, True])
def test_a_rewind_back_across_a_page_boundary(preplan):
    """A block that ends a page in, rewound to before the page: the next plan drops the page
    from the row and the block after takes it again."""
    ref, kv = _manager(0), _manager(1)
    for m in (ref, kv):
        _prefill(m, "a", 7)
    want, got = [], []
    for m, out in ((ref, want), (kv, got)):
        out.append(_verify(m, ("a",), 0, 4, preplan=preplan))  # 7 -> 11 slots
        m.rewind(VerdictBatch(["a"], ["main"], [3]))          # -> 8: back to two pages
        out.append(_verify(m, ("a",), 1, 4, preplan=preplan))  # 8 -> 12 (3 pages planned)
        m.rewind(VerdictBatch(["a"], ["main"], [3]))          # -> 9
        out.append(_verify(m, ("a",), 0, 4, preplan=preplan))
    assert [(_flat(r), s) for r, s in got] == [(_flat(r), s) for r, s in want]
    assert kv.plan_cache_stats["hit"] == 2 and kv.plan_cache_stats["mismatch"] == 0


def test_preplanned_verify_steps_hit_after_promotion():
    _, full = _run(0, 12, 2, 3)
    kv, fast = _run(1, 12, 2, 3, preplan=True)
    assert fast == full
    assert kv.plan_cache_stats == {"hit": 11, "miss": 0, "mismatch": 0}


def test_a_rewind_of_another_request_keeps_the_cache_and_a_reset_drops_it():
    ref, kv = _manager(0), _manager(1)
    for m in (ref, kv):
        for rid, n in (("a", 5), ("b", 6), ("c", 9)):
            _prefill(m, rid, n)
        _verify(m, ("a", "b"), 0, 3)
        # c's rewind names no cached row; a's patches its row
        m.rewind(VerdictBatch(["c", "a"], ["main", "main"], [2, 1]))
    assert _flat(_verify(kv, ("a", "b"), 1, 3)[0]) == _flat(_verify(ref, ("a", "b"), 1, 3)[0])
    assert kv.plan_cache_stats == {"hit": 1, "miss": 0, "mismatch": 0}
    for m in (ref, kv):
        m.reset_request("b")
        _prefill(m, "b", 3)
    assert _flat(_verify(kv, ("a", "b"), 0, 3)[0]) == _flat(_verify(ref, ("a", "b"), 0, 3)[0])
    assert kv.plan_cache_stats["miss"] == 1


def test_a_step_declared_again_for_its_forward_keeps_the_cache():
    """A pre-plan declares its step over the bucket's template rows, and a submodule whose step
    isn't input-free declares it again over the real inputs: an equal step, another object. Its
    commit is still the commit the cache planned for."""
    kv = _manager(1)
    for rid, n in (("a", 5), ("b", 6)):
        _prefill(kv, rid, n)
    for t in range(4):
        padded = ("a", "b")
        staged = KVStep(segments=tuple(Segment(r, "main", 4) for r in padded))
        ctx = _ctx(padded, padded, t % 2, 4, preplan=True)
        assert kv.admit(staged, ctx).ok
        kv.plan(staged, ctx)
        again = KVStep(segments=tuple(Segment(r, "main", 4) for r in padded))
        assert again.segments is not staged.segments
        ctx = _ctx(padded, padded, t % 2, 4)
        assert kv.admit(again, ctx).ok
        kv.plan(again, ctx)
        kv.commit(again, ctx)
    assert kv.plan_cache_stats == {"hit": 3, "miss": 0, "mismatch": 0}


@pytest.mark.parametrize("pad", [0, 2])
def test_host_write_slots_match_the_device_ones(pad):
    """A verify step planned from the streams computes its write slots on the host; they are
    the slots the device path unpacks from the same index arrays, padding rows included."""
    plans = {}
    for verify in (False, True):
        kv = _manager(0)
        for rid, n in (("a", 5), ("b", 7), ("c", 4)):
            _prefill(kv, rid, n)
        rng = random.Random(3)
        plans[verify] = []
        for t in range(8):
            padded = ("a", "b", "c") + tuple(f"pad{i}" for i in range(pad))
            res, _ = _verify(kv, ("a", "b", "c"), t % 2, 4, pad, verify=verify)
            state = kv._static_plan_states[(t % 2, "main")]
            n = len(padded) * 4
            plans[verify].append((_flat(res), state.token_to_page[:n].tolist(), state.token_to_cache[:n].tolist()))
            assert (res["main"].token_pages is not None) == verify
            kv.rewind(VerdictBatch(["a", "c"], ["main", "main"], [rng.randint(1, 3), rng.randint(1, 3)]))
    assert plans[True] == plans[False]


def test_decode_steps_still_cache_lazily():
    """A decode step's cache keeps its views and derives its rows at the first hit, as
    before; spans past the bound aren't cached."""
    kv = _manager(1)
    _prefill(kv, "a", 5)
    _verify(kv, ("a",), 0, 1)
    assert kv._decode_plan_cache is not None and kv._decode_plan_cache.derived is None
    _verify(kv, ("a",), 1, 17)
    assert kv._decode_plan_cache is None or kv._decode_plan_cache.span != 17


def _ahead(kv: KVManager, rids, slot: int, span: int, rejected: dict, pad: int = 0):
    """A verify step planned ahead of the last step's verdict, as the runner drives it: pre-admit
    and pre-plan from the committed lengths, then the rewind and `correct_preplan`, then the
    promotion and the commit. Also returns the plan the streams give for the step once the
    rewind has landed, and its write slots."""
    padded = tuple(rids) + tuple(f"pad{i}" for i in range(pad))
    step = KVStep(segments=tuple(Segment(r, "main", span) for r in padded), verify=True)
    ctx = _ctx(tuple(rids), padded, slot, span, preplan=True)
    assert kv.admit(step, ctx).ok
    ctx.plan_results["kv"] = kv.plan(step, ctx)
    kept = [(rid, r) for rid, r in rejected.items() if r]
    if kept:
        verdicts = VerdictBatch([rid for rid, _ in kept], ["main"] * len(kept), [r for _, r in kept])
        kv.rewind(verdicts)
        assert kv.correct_preplan(verdicts, step, ctx)
    ctx = _ctx(tuple(rids), padded, slot, span)
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    slots = _slots(kv, len(rids) * span)
    views = kv._sequence_views(list(step.segments), {}, ctx, step.sink)
    full = {"main": kv._plan_output(views)}
    want = ([], [])
    for v in views[:len(rids)]:
        for pos in range(v.length - v.to_compute, v.length):
            want[0].append(v.page_idxs[pos // PAGE])
            want[1].append(pos % PAGE)
    kv.commit(step, ctx)
    return res, slots, full, want


def _lengths(res):
    o = res["main"]
    return [(v.request_id, v.length, v.to_compute) for v in o.views], o.cpu_indptrs.kv_lens.tolist()


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("pad", [0, 2])
def test_a_step_planned_ahead_and_corrected_is_the_step_planned_after_the_rewind(mode, pad):
    """Planned from the committed lengths and corrected for the verdict, a step's plan and write
    slots are the ones its streams give once the rewind has landed, off the cache or from the
    streams alike; and its lengths are those of the step planned after the rewind (the lag-1
    order). Its pages may differ from that step's: it reserved for the longer rows."""
    k = 3
    rng = random.Random(7)
    ref, kv = _manager(0), _manager(mode)
    rids = ("a", "b", "c")
    for m in (ref, kv):
        for rid, n in zip(rids, (5, 7, 4), strict=True):
            _prefill(m, rid, n)
    rejected = dict.fromkeys(rids, 0)
    for t in range(16):
        kept = [(rid, r) for rid, r in rejected.items() if r]
        if kept:
            ref.rewind(VerdictBatch([rid for rid, _ in kept], ["main"] * len(kept), [r for _, r in kept]))
        want, _ = _verify(ref, rids, t % 2, k + 1, pad)
        res, slots, full, full_slots = _ahead(kv, rids, t % 2, k + 1, rejected, pad)
        assert _flat(res) == _flat(full), t
        assert slots == full_slots, t
        assert _lengths(res) == _lengths(want), t
        rejected = {rid: rng.randint(0, k) for rid in rids}
    for rid in rids:
        assert kv._streams[rid]["main"].stored_len == ref._streams[rid]["main"].stored_len
    assert kv.plan_cache_stats["mismatch"] == 0
    if mode:
        assert kv.plan_cache_stats["hit"] == 15


def test_a_staged_rewind_refused_where_pages_moved():
    """A rewind that would move pages under the staged step (a sealed tail) can't be corrected."""
    kv = _manager(1)
    _prefill(kv, "a", 7)
    _verify(kv, ("a",), 0, 4)
    padded = ("a",)
    step = KVStep(segments=(Segment("a", "main", 4),))
    ctx = _ctx(padded, padded, 1, 4, preplan=True)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    stream = kv._streams["a"]["main"]
    kv._arena.seal(stream.page_indices[2:3])
    verdicts = VerdictBatch(["a"], ["main"], [3])
    kv.rewind(verdicts)
    assert not kv.correct_preplan(verdicts, step, ctx)
