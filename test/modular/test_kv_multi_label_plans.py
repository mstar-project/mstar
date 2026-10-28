"""KV plans of more than one label (BAGEL's CFG: main, cfg_text, cfg_img)
under the opt-in host paths: the index ring (MSTAR_KV_PINNED_INDPTRS) keeps
every label's arrays for the whole plan, and the plan cache
(MSTAR_KV_PLAN_CACHE) gives each label its own rows. CPU only: the ring runs
on plain tensors."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from mstar.engine.resources.kv.config import KVReqConfig, KVStep  # noqa: E402
from mstar.engine.resources.kv.plan import PinnedIndexRing  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kv_decode_plan_cache_helpers", pathlib.Path(__file__).with_name("test_kv_decode_plan_cache.py"),
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

LABELS = ("main", "cfg_text", "cfg_img")


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(P.H.manager_mod, "KVTransferManager", P.H._StubTransfer)


def _manager(mode: int, ring: bool):
    kv = P._manager(mode)
    if ring:
        kv._indptr_ring = PinnedIndexRing(numel=8)
    return kv


def _segments(rid_labels: dict[str, tuple[str, ...]], span: int | dict[str, int]):
    """Label-major, as BAGEL declares them."""
    return tuple(
        Segment(rid, label, span if isinstance(span, int) else span[rid])
        for label in LABELS
        for rid, labels in rid_labels.items() if label in labels
    )


def _prefill(kv, rid_labels: dict[str, tuple[str, ...]], lens: dict[str, int]):
    for rid in rid_labels:
        kv.ingest_request(rid, KVReqConfig())
    step = KVStep(segments=_segments(rid_labels, lens))
    ctx = StepContext(request_ids=tuple(rid_labels), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    kv.commit(step, ctx)
    return res


def _decode(kv, rid_labels: dict[str, tuple[str, ...]], slot: int, pad: int = 0):
    rids = tuple(rid_labels)
    padded = dict(rid_labels)
    labelled = {lab for labels in rid_labels.values() for lab in labels}
    padded.update((f"pad{i}", tuple(lab for lab in LABELS if lab in labelled)) for i in range(pad))
    step = KVStep(segments=_segments(padded, 1))
    bucket = BucketKey(graph_walk=P.WALK, bs=len(padded), num_tokens=len(step.segments))
    ctx = StepContext(
        request_ids=rids, graph_walk=P.WALK, slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=bucket),
    )
    ctx.set_padded_rids(tuple(padded))
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    flat = P._flat(res)
    kv.commit(step, ctx)
    return res, flat


SETS = {
    "one_label": {"a": ("main",), "b": ("main",), "c": ("main",)},
    "two_labels": {"a": ("main", "cfg_img"), "b": ("main", "cfg_img")},
    "three_labels": {"a": LABELS, "b": LABELS},
    "mixed": {"a": ("main",), "b": ("main", "cfg_img"), "c": LABELS},
}
LENS = {"a": 5, "b": 7, "c": 4}


def _run(mode: int, rid_labels, ring: bool, pad: int, steps: int = 10):
    kv = _manager(mode, ring)
    _prefill(kv, rid_labels, LENS)
    return kv, [_decode(kv, rid_labels, slot=t % 2, pad=pad)[1] for t in range(steps)]


def test_cached_multilabel_decode_has_one_query_row_per_label():
    """One request on main and cfg_img: each label's plan has one row, so
    qo_indptr is [0, 1], on the cached steps too."""
    kv = _manager(1, ring=False)
    _prefill(kv, {"r": ("main", "cfg_img")}, {"r": 1})
    for t in range(3):
        res, _ = _decode(kv, {"r": ("main", "cfg_img")}, slot=t % 2)
        for label, out in res.items():
            assert out.cpu_indptrs.qo_indptr.tolist() == [0, 1], (t, label)
    assert kv.plan_cache_stats == {"hit": 2, "miss": 0, "mismatch": 0}


def test_ring_keeps_every_label_until_the_plan_is_consumed():
    """Three labels take twelve arrays; none may overwrite another of the
    same plan (the setup of the plan states reads them all)."""
    kv = _manager(0, ring=True)
    lens = {"main": 5, "cfg_text": 9, "cfg_img": 13}
    kv.ingest_request("r", KVReqConfig())
    step = KVStep(segments=tuple(Segment("r", lab, lens[lab]) for lab in LABELS))
    ctx = StepContext(request_ids=("r",), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    for label in LABELS:
        out = res[label]
        assert out.cpu_indptrs.qo_indptr.tolist() == [0, lens[label]]
        assert out.cpu_indptrs.paged_kv_indices.tolist() == out.views[0].page_idxs


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("name", list(SETS))
def test_cached_plans_match_the_full_ones(name, ring):
    rid_labels = SETS[name]
    _, full = _run(0, rid_labels, ring=False, pad=0)
    kv, fast = _run(1, rid_labels, ring=ring, pad=0)
    assert fast == full
    assert kv.plan_cache_stats == {"hit": 9, "miss": 0, "mismatch": 0}
    kv, _ = _run(2, rid_labels, ring=ring, pad=0)
    assert kv.plan_cache_stats == {"hit": 9, "miss": 0, "mismatch": 0}


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize("name", list(SETS))
def test_padded_multilabel_steps_plan_what_the_streams_say(name, ring):
    """Padding rows carry every label of the batch; whether or not the cache
    serves the step, the plan is the full plan's."""
    rid_labels = SETS[name]
    _, full = _run(0, rid_labels, ring=False, pad=2)
    kv, fast = _run(2, rid_labels, ring=ring, pad=2)
    assert fast == full
    assert kv.plan_cache_stats["mismatch"] == 0


def test_padding_under_some_labels_only_plans_from_the_streams():
    """Real rows first, padding under main alone: a hit would hand cfg_img
    main's padding rows, so the step plans from the streams."""
    layout = (("a", "main"), ("a", "cfg_img"), ("pad0", "main"))
    plans = {}
    for mode in (0, 2):
        kv = _manager(mode, ring=False)
        _prefill(kv, {"a": ("main", "cfg_img")}, {"a": 5})
        plans[mode] = []
        for t in range(4):
            step = KVStep(segments=tuple(Segment(r, lab, 1) for r, lab in layout))
            ctx = StepContext(
                request_ids=("a",), graph_walk=P.WALK, slot=t % 2, capture=False,
                slot_lease=SlotLease(slot=t % 2, bucket=BucketKey(graph_walk=P.WALK, bs=2, num_tokens=3)),
            )
            ctx.set_padded_rids(("a", "pad0"))
            assert kv.admit(step, ctx).ok
            plans[mode].append(P._flat(kv.plan(step, ctx)))
            kv.commit(step, ctx)
    assert plans[2] == plans[0]
    assert kv.plan_cache_stats["mismatch"] == 0


def test_padded_combined_labels_keep_their_real_rows():
    """Real rows first, then one padding row on every label, and two combined
    plan labels: in AB, A's padding comes before B's real rows, which a hit
    must not take for padding (the counts alone match the step's padding)."""
    real = (("a", "A"), ("a", "B"), ("b", "B"), ("a", "C"), ("a", "D"), ("b", "D"))
    layout = real + tuple(("pad0", lab) for lab in "ABCD")
    combined = {("A", "B"): "AB", ("C", "D"): "CD"}
    plans = {}
    for mode in (0, 1):
        kv = _manager(mode, ring=False)
        for rid in ("a", "b"):
            kv.ingest_request(rid, KVReqConfig())
        step = KVStep(segments=tuple(Segment(r, lab, 5) for r, lab in real))
        ctx = StepContext(request_ids=("a", "b"), graph_walk="prefill", slot=0, capture=False)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)
        plans[mode] = []
        for t in range(4):
            step = KVStep(segments=tuple(Segment(r, lab, 1) for r, lab in layout), combined_labels=combined)
            ctx = StepContext(
                request_ids=("a", "b"), graph_walk=P.WALK, slot=t % 2, capture=False,
                slot_lease=SlotLease(slot=t % 2, bucket=BucketKey(graph_walk=P.WALK, bs=3, num_tokens=10)),
            )
            ctx.set_padded_rids(("a", "b", "pad0"))
            assert kv.admit(step, ctx).ok
            plans[mode].append(P._flat(kv.plan(step, ctx)))
            kv.commit(step, ctx)
    assert plans[1] == plans[0]


def _layout_decodes(mode: int, layouts):
    """Decode steps of rid "a" on labels A and B, one per (segments, combined
    labels) in ``layouts``; the flat plans and the cache stats."""
    kv = _manager(mode, ring=False)
    kv.ingest_request("a", KVReqConfig())
    step = KVStep(segments=(Segment("a", "A", 5), Segment("a", "B", 6)))
    ctx = StepContext(request_ids=("a",), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    plans = []
    for t, (layout, combined) in enumerate(layouts):
        step = KVStep(segments=tuple(Segment(r, lab, 1) for r, lab in layout), combined_labels=combined)
        rids = tuple(dict.fromkeys(r for r, _ in layout))
        ctx = StepContext(
            request_ids=("a",), graph_walk=P.WALK, slot=t % 2, capture=False,
            slot_lease=SlotLease(slot=t % 2, bucket=BucketKey(graph_walk=P.WALK, bs=len(rids), num_tokens=len(layout))),
        )
        ctx.set_padded_rids(rids)
        assert kv.admit(step, ctx).ok
        plans.append(P._flat(kv.plan(step, ctx)))
        kv.commit(step, ctx)
    return plans, kv.plan_cache_stats


def test_padding_that_changes_label_is_planned_from_the_streams():
    """In a combined label the padding's labels place the real rows: padding
    under B, then under A, puts A's padding before B's real row."""
    real = (("a", "A"), ("a", "B"))
    layouts = [(real + (("pad0", "B"),), {("A", "B"): "AB"}), (real + (("pad0", "A"),), {("A", "B"): "AB"})]
    full, _ = _layout_decodes(0, layouts)
    cached, stats = _layout_decodes(1, layouts)
    assert cached == full
    assert stats["hit"] == 0


def test_a_step_that_combines_labels_is_planned_from_the_streams():
    """The same real rows with and without combined labels are different plans."""
    real = (("a", "A"), ("a", "B"))
    layouts = [(real, {}), (real, {("A", "B"): "AB"}), (real, {("A", "B"): "AB"})]
    full, _ = _layout_decodes(0, layouts)
    cached, stats = _layout_decodes(1, layouts)
    assert cached == full
    assert stats == {"hit": 1, "miss": 1, "mismatch": 0}


def test_a_plans_arrays_outlive_the_next_plan():
    """The ring hands a set back after two plans, whatever the label count:
    one plan's arrays are intact while the next plan is built."""
    kv = _manager(0, ring=True)
    rid_labels = SETS["three_labels"]
    _prefill(kv, rid_labels, LENS)
    prev, prev_flat = _decode(kv, rid_labels, slot=0)
    _decode(kv, rid_labels, slot=1)
    assert P._flat(prev) == prev_flat
