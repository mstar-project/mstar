"""The KV page table: a stream's ``StreamPages`` behaves as the list it
replaces, and a plan built from columns (``KVManager._plan_label``) equals the
one built from per-row views, whatever the streams went through."""
from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.page_table import PageTable  # noqa: E402
from mstar.engine.resources.kv.plan import group_by_plan_label  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext  # noqa: E402

PAGE = 4
LABELS = ("main", "aux")


@pytest.mark.parametrize("seed", range(20))
def test_stream_pages_behave_as_lists(seed):
    """Several streams on one table, so extents move, are freed and are reused."""
    rng = random.Random(seed)
    table = PageTable(capacity=8)
    streams = [table.new() for _ in range(4)]
    refs: list[list[int]] = [[] for _ in streams]
    for _ in range(400):
        i = rng.randrange(len(streams))
        s, ref = streams[i], refs[i]
        op = rng.random()
        if op < 0.4:
            new = [rng.randrange(1000) for _ in range(rng.randrange(1, 20))]
            s.extend(new)
            ref.extend(new)
        elif op < 0.55 and ref:
            a = rng.randrange(len(ref))
            b = rng.randrange(a, len(ref) + 1)
            del s[a:b]
            del ref[a:b]
        elif op < 0.65 and ref:
            k = rng.randrange(1, len(ref) + 1)
            del s[-k:]
            del ref[-k:]
        elif op < 0.75:
            new = [rng.randrange(1000) for _ in range(rng.randrange(0, 40))]
            s.assign(new)
            ref[:] = new
        elif op < 0.8:
            s.clear()
            ref.clear()
        elif op < 0.85:
            s.free()
            ref.clear()
        for got, want in zip(streams, refs, strict=True):
            assert got == want and len(got) == len(want) and list(got) == want
        if ref:
            j = rng.randrange(-len(ref), len(ref))
            assert s[j] == ref[j]
            a, b = sorted(rng.randrange(-len(ref) - 2, len(ref) + 2) for _ in range(2))
            assert s[a:b] == ref[a:b]


def test_gather_reads_each_rows_prefix():
    table = PageTable(capacity=8)
    a, b = table.new(), table.new()
    a.extend([1, 2, 3, 4, 5])
    b.extend([9, 8])
    offs = np.array([a.off, b.off, a.off], np.int64)
    counts = np.array([3, 2, 0], np.int64)
    indptr = np.array([0, 3, 5, 5], np.int64)
    assert table.gather(offs, counts, indptr).tolist() == [1, 2, 3, 9, 8]


class _StubTransfer:
    def __init__(self, *a, **k):
        pass

    def get_kv_transfer_info(self, **k):
        pass

    def owns_transfer_info(self, *a, **k):
        return False

    def cleanup(self):
        pass

    def remove_request(self, rid):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager() -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512,
            max_num_pages=2048, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    return kv


def _run(kv, segments, rids, padded=None, lease=False):
    step = KVStep(segments=tuple(segments))
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=BucketKey("decode", len(segments), len(segments))) if lease else None,
    )
    if padded is not None:
        ctx.set_padded_rids(padded)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _same(a, b):
    for f in ("qo_indptr", "paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len", "kv_lens"):
        x, y = getattr(a.cpu_indptrs, f), getattr(b.cpu_indptrs, f)
        assert x.dtype == y.dtype, f
        assert torch.equal(x.as_subclass(torch.Tensor), y.as_subclass(torch.Tensor)), f
    assert a.views == b.views
    assert a.is_decode == b.is_decode
    if a.decode_pages is not None:
        want = [v.page_idxs[-1] for v in b.views]
        assert [int(p) for p in a.decode_pages] == want
        assert [int(o) for o in a.decode_offsets] == [(v.last_page_len(PAGE) or PAGE) - 1 for v in b.views]


@pytest.mark.parametrize("seed", range(12))
def test_a_columnar_plan_equals_the_per_row_plan(seed):
    rng = random.Random(seed)
    kv = _manager()
    live: list[str] = []
    for step_no in range(60):
        # change the streams: new requests, prefills, decodes, front releases, resets, removals
        op = rng.random()
        if op < 0.25 or not live:
            rid = f"r{step_no}"
            kv.ingest_request(rid, KVReqConfig())
            _run(kv, [Segment(rid, label, rng.randrange(1, 40)) for label in LABELS], [rid])
            live.append(rid)
        elif op < 0.55:
            rows = rng.sample(live, rng.randrange(1, len(live) + 1))
            _run(kv, [Segment(r, label, 1) for r in rows for label in LABELS], rows)
        elif op < 0.65:
            r = rng.choice(live)
            kv.release_oldest(r, rng.randrange(PAGE, 4 * PAGE))
        elif op < 0.72:
            r = rng.choice(live)
            kv.remove_request(r)
            live.remove(r)
        # then plan a step over the live rows both ways: decode or prefill spans,
        # in any order, with padding rows and a sink row
        if not live:
            continue
        rows = rng.sample(live, rng.randrange(1, len(live) + 1))
        prefill = rng.random() < 0.3
        segs = [Segment(r, label, rng.randrange(0, 9) if prefill else 1) for r in rows for label in LABELS]
        pads = [f"pad{i}" for i in range(rng.randrange(0, 3))]
        segs += [Segment(p, label, 1) for p in pads for label in LABELS]
        ctx = StepContext(request_ids=tuple(rows), graph_walk="decode", slot=0, capture=False)
        ctx.set_padded_rids(tuple(rows) + tuple(pads))
        sink = frozenset({(rows[0], "main")}) if rng.random() < 0.2 else frozenset()
        # the two labels as one plan label (a combined CFG plan) or each its own
        combined = {LABELS: "both"} if rng.random() < 0.5 else {}
        for _label, group in group_by_plan_label(tuple(segs), combined).items():
            columns = kv._plan_label(group, {}, ctx, sink)
            per_row = kv._plan_output(kv._sequence_views(group, {}, ctx, sink))
            _same(columns, per_row)
