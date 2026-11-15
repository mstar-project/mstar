"""KV host plans: the index ring (MSTAR_KV_PINNED_INDPTRS) keeps every label's
arrays for the whole plan (BAGEL's CFG: main, cfg_text, cfg_img), a plan and a
removed request's streams go by refcount, and the sink pages grow under the
manager's lock. CPU only: the ring runs on plain tensors."""
from __future__ import annotations

import gc
import importlib.util
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.kv import arena as arena_mod  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import CacheStream, KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import KVPlanOutputs, PinnedIndexRing  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kv_prefix_decode_helpers", pathlib.Path(__file__).with_name("test_kv_prefix_decode.py"),
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

PAGE = 4
WALK = "decode"
LABELS = ("main", "cfg_text", "cfg_img")


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", H._StubTransfer)


def _manager(ring: bool = False) -> KVManager:
    cfg = PagedKVConfig(num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=256, num_layers=1, page_size=PAGE)
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    if ring:
        kv._indptr_ring = PinnedIndexRing(numel=8)
    return kv


def _flat(res):
    """Each (label, group)'s plan, as lists."""
    out = {}
    for g, outputs in enumerate(res.groups):
        for label, o in outputs.items():
            ind = o.cpu_indptrs
            out[label, g] = (
                [tuple(v[:5]) for v in o.views],
                ind.qo_indptr.tolist(), ind.paged_kv_indptr.tolist(), ind.paged_kv_indices.tolist(),
                ind.paged_kv_last_page_len.tolist(), ind.kv_lens.tolist(),
            )
    return out


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


def _decode(kv, rid_labels: dict[str, tuple[str, ...]], slot: int):
    rids = tuple(rid_labels)
    step = KVStep(segments=_segments(rid_labels, 1))
    bucket = BucketKey(graph_walk=WALK, bs=len(rids), num_tokens=len(step.segments))
    ctx = StepContext(
        request_ids=rids, graph_walk=WALK, slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=bucket),
    )
    ctx.set_padded_rids(rids)
    assert kv.admit(step, ctx).ok
    res = kv.plan(step, ctx)
    flat = _flat(res)
    kv.commit(step, ctx)
    return res, flat


def test_ring_keeps_every_label_until_the_plan_is_consumed():
    """Three labels take twelve arrays; none may overwrite another of the
    same plan (the setup of the plan states reads them all)."""
    kv = _manager(ring=True)
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


def test_a_plans_arrays_outlive_the_next_plan():
    """The ring hands a set back after two plans, whatever the label count:
    one plan's arrays are intact while the next plan is built."""
    kv = _manager(ring=True)
    rid_labels = {"a": LABELS, "b": LABELS}
    _prefill(kv, rid_labels, {"a": 5, "b": 7})
    prev, prev_flat = _decode(kv, rid_labels, slot=0)
    _decode(kv, rid_labels, slot=1)
    assert _flat(prev) == prev_flat


def test_plans_and_streams_are_freed_by_refcount():
    """A step's plan, with its pinned and device indptrs, and a removed request's streams go
    when the last reference does, not at the cyclic GC's next pass."""
    kv = _manager()
    gc.collect()
    # another test's module may keep its managers, and their streams, for the session
    before = {id(o) for o in gc.get_objects() if isinstance(o, (KVPlanOutputs, CacheStream))}
    gc.disable()
    try:
        _prefill(kv, {"a": ("main",)}, {"a": 5})
        for t in range(4):
            _decode(kv, {"a": ("main",)}, slot=t % 2)
        kv.remove_request("a")
        alive = [
            type(o).__name__ for o in gc.get_objects()
            if isinstance(o, (KVPlanOutputs, CacheStream)) and id(o) not in before
        ]
    finally:
        gc.enable()
    assert alive == []


def test_sink_pages_grow_under_the_managers_lock(monkeypatch):
    """The sink run is a page list the main thread changes under the lock, and
    a plan may run on the plan thread: growing it takes the lock too. Capture
    sizes it for the longest captured step. On the Python arena, whose list a
    spy can wrap."""
    monkeypatch.setattr(manager_mod, "new_page_arena", lambda n: arena_mod.new_page_arena(n, "python"))
    kv = _manager()
    owned = []
    extend = arena_mod._ListStreamPages.extend

    def spy(pages, new):
        if pages is kv._sink_pages:
            owned.append(kv._lock._is_owned())
        extend(pages, new)

    monkeypatch.setattr(arena_mod._ListStreamPages, "extend", spy)
    kv.ingest_request("b", KVReqConfig())
    step = KVStep(segments=(Segment("b", "main", 200),), sink=frozenset({("b", "main")}))
    ctx = StepContext(request_ids=("b",), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    owned.clear()
    kv.plan(step, ctx)
    assert owned and all(owned)
    assert len(kv._sink_pages) >= -(-200 // PAGE)
    kv.build_accelerator_graph_buffers([], 4, 512)
    assert len(kv._sink_pages) == 512 // PAGE
