"""A verify step adds a block of draft entries per row, and its settle keeps only what the target model accepted.

The accepted count is known on the device once the step has run, and the host reads each stream's length again at
the next admission. So `settle` copies the counts to a table on the host, and the KV and position resources each move
their own stream back by them before anything reads its length: at the next admit, before an offload or a drop takes
the pages, and when the model asks a position. A count applied twice, to the wrong row, or after a later step reused
its table would leave the pages and the positions apart, and no kernel would notice.
"""

from __future__ import annotations

import gc
import re
import sys
from dataclasses import dataclass
from typing import Any, NamedTuple

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import test_kv_host_index as host
import torch
from test_kv_preemption import _offload, _preempt, _running
from test_kv_preemption import _pool as _preemption_pool
from test_kv_preemption import _step as _pool_step
from test_kv_prefix_lease import NODE, PAGE_SIZE
from test_resident_delta import _admits, _ingest, _pool_rank, _rebuilds_none
from test_state_declaration_checks import _StubModel

from mstar.engine.resources import Residency, Segment, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionConfig, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import BucketKey, SlotLease, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Bidirectional,
    Block,
    Causal,
    Entry,
    Fork,
    FrameWindow,
    KVEntry,
    LatentAttend,
    LatentEntry,
    Positions,
    Read,
    Selected,
    TensorType,
    Value,
    Window,
    adapters,
    settle,
)
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle, build_specs, state_steps
from mstar.engine.state.analysis import state_declaration_problems
from mstar.engine.state.calls import VerifyRun
from mstar.engine.state.contract import Op
from mstar.model.base import KVLabel, PrefixKeys

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a count event in a graph needs a GPU")

LAYERS, SLOTS, HEADS, DIM, SPAN = 2, 64, 2, 8, 4
ATTEND = Attend(q_heads=HEADS)


class _StubKV:
    """Holds pages per layer and records the counts it is handed instead of applying them."""

    tp_group = None

    def __init__(self):
        self.pages = torch.zeros(LAYERS, SLOTS, 2, HEADS, DIM)
        self.corrections = []

    def layer_view(self, i):
        return self.pages[i]

    def write_kv(self, k, v, layer_idx=None, label=None, tokens=None):
        self.pages[layer_idx, :k.shape[0], 0] = k

    def opens_label(self, rid, label):
        return True

    def defer_correct_len(self, correction):
        self.corrections.append(correction)

    @property
    def deferred(self):
        """Each request's rid, label and count, in the order the steps deferred them."""
        return [(rid, step.label, _Row(step, rid)) for step in self.corrections for rid in step.rids]


class _StubAttn:
    requires_kv_write = True

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        return q

    def reads_step_kv(self, label=None):
        return False


class _StubPositions(_StubKV):
    """Records the counts it is handed, as the KV stub does."""


class _Row(NamedTuple):
    """One request's count of a step's correction."""

    step: Any
    rid: Any

    def delta(self):
        return self.step.delta(self.rid)


class _Count:
    """A correction of ``rids`` whose step has run: ``delta`` tokens each, or what the row ``delta`` reads."""

    label = "main"

    def __init__(self, delta, *rids):
        self._delta, self.rids = delta, rids

    def delta(self, rid):
        return self._delta.delta() if isinstance(self._delta, _Row) else self._delta


def _array(span=SPAN, **overrides) -> Array:
    return Array("kv", **{
        "nodes": {"LLM"}, "layers": frozenset(range(LAYERS)), "entry": KVEntry(heads=HEADS, head_dim=DIM),
        "max_len": SLOTS, "reads": {"attn": Read(mask=Causal(), op=ATTEND)}, "positions": Positions(),
        "verify_span": span, **overrides,
    })


def _handle(array=None, kv=None, positions=None):
    array = array or _array()
    resources = {"kv": kv or _StubKV(), "attn": _StubAttn(), "rope": positions or _StubPositions()}
    return build_handle(array, resources, _resolve([array], StateSetup()))


def _ctx(rows=(1, 2), lease=None, capture=False) -> StepContext:
    return StepContext(request_ids=rows, graph_walk="decode", slot=0, capture=capture, slot_lease=lease)


def _step(h, spans=(SPAN, SPAN), ctx=None, labels=None, **declared) -> SubmoduleStep:
    labels = labels or ("main",) * len(spans)
    step = SubmoduleStep(
        segments=[Segment(row, label, span) for row, (label, span) in enumerate(zip(labels, spans, strict=True), 1)],
        state={h.name: ArrayStep(**{"verify": True, **declared})},
    )
    step.set_ctx(ctx or _ctx(tuple(range(1, len(spans) + 1))))
    return step


def _verify(h, kept, step=None):
    """Run one verify step on ``h`` through its commit, adding every row's block and keeping ``kept`` per row."""
    step = step or _step(h, spans=(SPAN,) * len(kept))
    h.begin_step(step)
    if not step.ctx.replays:
        k = torch.randn(sum(segment.span for segment in step.segments), HEADS, DIM)
        for layer in range(LAYERS):
            h.add(layer, k=k, v=k)
        settle(h, torch.tensor(kept))
    h.check_step(step)
    h.commit_step(step)
    h.end_step(step)
    return step


def _refused(pattern: str, call) -> None:
    with pytest.raises(ValueError, match=pattern):
        call()


# Load refusals


def test_a_verify_span_needs_kept_entries():
    _refused("verify_span must be a positive int", lambda: _array(span=0))
    _refused("verify_span must be a positive int", lambda: _array(span="2"))
    _refused("max_len 0 keeps nothing for a verify step to settle",
             lambda: _array(max_len=0, positions=None, reads={"v": Read(mask=Block(), op=ATTEND)}))


def test_a_verify_span_refuses_a_bidirectional_read():
    array = _array(reads={"attn": Read(mask=Bidirectional(kept_window=True), op=ATTEND)})
    assert state_declaration_problems(_StubModel(array)) == [
        "Array 'kv'.verify_span: 4 beside reads['attn']. Bidirectional sees each step's adds as a whole, and a settle "
        "keeps only part of a verify step's adds"
    ], "keeping some of a step's adds changes what a read that saw the step whole saw"


def test_a_verify_span_refuses_prefix_keys():
    labels = {"kv": {"main": KVLabel(keys=PrefixKeys("text_inputs", ("prefill",)))}}
    assert state_declaration_problems(_StubModel(_array(), kv_labels=labels)) == [
        "Array 'kv'.verify_span: 4, and kv_labels()['kv']['main'] keys for prefix reuse, which would key the entries "
        "a settle drops"
    ]


def _host_value(**entry) -> Array:
    return Array("kv", nodes={"LLM"}, layers={0}, entry=Entry(**entry), max_len=1, rewritable=True, verify_span=2)


@pytest.mark.parametrize("arrays, storage", [
    ([_array(span=2, reads={"attn": Read(mask=Causal(), op=ATTEND, walks={"decode"})})], "PagedByWalkStorage"),
    ([Array("q", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=SLOTS,
            reads={"self": Read(mask=Causal(), op=ATTEND)}),
      _array(span=2, positions=None, reads={"cross": Read(mask=Bidirectional(), op=ATTEND, queries="q")})],
     "PagedCrossStorage"),
    ([_array(span=2, positions=None, reads={"attn": Read(mask=FrameWindow(16, frames=2), op=ATTEND, packed=False)})],
     "RingStorage"),
    ([_host_value(tensors={"x": TensorType((4,), dtype=torch.float32)}, host={"n": int})], "ValueWithHostStorage"),
    ([_host_value(tensors={}, host={"n": int})], "HostValueStorage"),
])
def test_a_verify_span_where_nothing_settles_has_no_layout(arrays, storage):
    _refused(f"no layout yet: verify_span 2, which today's resources settle on no {storage}",
             lambda: build_specs(arrays))


def test_a_rewound_pool_with_kv_groups_has_no_verify_layout():
    reads = {"local": Read(mask=Window(8), op=ATTEND, layers={0}),
             "global": Read(mask=Causal(), op=ATTEND, layers={1})}
    _refused("beside Window reads \\['local'\\], whose pool keeps KV groups",
             lambda: build_specs([_array(reads=reads)]))


class _Unbound(Op):
    reads = ("k",)


def test_a_verify_entry_builds_no_spec_or_step_and_lists_no_op():
    assert build_specs([_array(span=SPAN)]) == build_specs([_array(span=1)]), "a verify span builds no resource"
    steps = []
    for span in (SPAN, 1):
        h = _handle(_array(span=span))
        step = SubmoduleStep(segments=[Segment(1, "main", 1)], state={"kv": ArrayStep()})
        steps.append(state_steps(step, {"kv": h}).steps)
    assert steps[0] == steps[1], "a verify span adds no step to a step that doesn't verify"
    with pytest.raises(ValueError, match="must be Attend, the reads") as refused:
        build_specs([_array(reads={"x": Read(mask=Causal(), op=_Unbound())})])
    assert "Verify" not in str(refused.value), "the verify entry is no op a read may name"


# Step refusals


def test_a_verify_step_fits_its_array_s_span():
    h = _handle(_array(span=1))
    _refused("a verify step, and the array declares no verify span", lambda: h.begin_step(_step(h)))
    for declared, cause in (
        ({"keep": False}, "keep=False, forks or placeholders"),
        ({"forks": (Fork(src="main", dst="main", at="end"),)}, "keep=False, forks or placeholders"),
        ({"placeholders": frozenset({(1, "main")})}, "keep=False, forks or placeholders"),
    ):
        h = _handle()
        _refused(cause, lambda h=h, declared=declared: h.begin_step(_step(h, **declared)))
    h = _handle()
    _refused("adds 5 entries for row 2, and a verify block holds 1 to verify_span 4",
             lambda: h.begin_step(_step(h, spans=(SPAN, SPAN + 1))))
    h = _handle(_array(labels={"main", "draft"}))
    _refused("over several labels", lambda: h.begin_step(_step(h, labels=("main", "draft"))))


def test_a_verify_step_settles_once():
    h = _handle()
    step = _step(h, verify=False)
    h.begin_step(step)
    _refused("settle on a step that doesn't verify", lambda: settle(h, torch.tensor([1, 1])))
    h.end_step(step)

    step = _step(h)
    h.begin_step(step)
    _refused("a verify step ends with no settle", lambda: h.check_step(step))
    settle(h, torch.tensor([1, 1]))
    _refused("a second settle in one verify step", lambda: settle(h, torch.tensor([1, 1])))
    h.end_step(step)

    _refused("settle in a step that doesn't list the array", lambda: settle(h, torch.tensor([1, 1])))
    plain = _handle(_array(span=1))
    plain.begin_step(_step(plain, verify=False))
    _refused("settle, and the array declares no verify span", lambda: settle(plain, torch.tensor([1, 1])))


def test_settle_refuses_a_count_outside_the_block(monkeypatch):
    monkeypatch.setattr(adapters, "_DEBUG_ASSERTS", True)
    for kept, cause in (([0, 2], "keeps 0 entries of row 1"), ([2, 5], "keeps 5 entries of row 2")):
        h = _handle()
        h.begin_step(_step(h))
        _refused(f"{cause}, whose verify block holds 4", lambda h=h, kept=kept: settle(h, torch.tensor(kept)))


def test_a_verify_step_whose_capture_repeats_is_refused():
    h = _handle()
    ctx = _ctx()
    ctx.repeats = 2
    _refused("replays 2 times", lambda: h.begin_step(_step(h, ctx=ctx)))


# The counts


def test_the_rewind_hands_each_real_row_its_settled_count():
    kv, positions = _StubKV(), _StubPositions()
    h = _handle(kv=kv, positions=positions)
    step = SubmoduleStep(segments=[Segment(1, "main", 4), Segment(2, "main", 3), Segment(-1, "main", 4)],
                         state={"kv": ArrayStep(verify=True)})
    step.set_ctx(_ctx(rows=(1, 2)))
    _verify(h, [2, 3, 1], step)
    assert [(rid, label, count.delta()) for rid, label, count in kv.deferred] == [(1, "main", -2), (2, "main", 0)], (
        "each real row moves back by its block less what it kept, and a padding row moves nothing"
    )
    assert [count for *_, count in positions.deferred] == [count for *_, count in kv.deferred], (
        "the positions move back by the same counts"
    )


def _left_alive(call) -> int:
    """The container objects ``call`` leaves alive, by the collector's own count of its youngest generation."""
    gc.collect()
    gc.disable()
    try:
        before = gc.get_count()[0]
        call()
        return gc.get_count()[0] - before
    finally:
        gc.enable()


def test_a_32_row_verify_step_defers_one_correction():
    """CPython collects its youngest generation once 700 more containers are alive than at its last collection. A
    count object per row, held in a dict per row on each of the two resources until the next admit, took a 32-row
    GLM-5.2 step past 700 on most steps, and every collection ran on the worker's main thread."""
    kv, positions = _StubKV(), _StubPositions()
    h = _handle(kv=kv, positions=positions)
    step = _step(h, spans=(1,) * 32)
    h.begin_step(step)
    k = torch.randn(32, HEADS, DIM)
    for layer in range(LAYERS):
        h.add(layer, k=k, v=k)
    settle(h, torch.ones(32, dtype=torch.long))
    h.check_step(step)
    alive = _left_alive(lambda: h.commit_step(step))
    h.end_step(step)
    assert len(kv.corrections) == 1 and positions.corrections == kv.corrections, (
        "the two resources each hold a correction of their own"
    )
    assert alive < 16, f"the commit left {alive} objects alive, and the step's rows are 32"


def test_a_verify_step_takes_one_row_per_request():
    h = _handle()
    step = SubmoduleStep(segments=[Segment(1, "main", SPAN), Segment(1, "main", SPAN)],
                         state={"kv": ArrayStep(verify=True)})
    step.set_ctx(_ctx(rows=(1,)))
    with pytest.raises(AssertionError, match="two rows of request 1"):
        _verify(h, [2, 3], step)


def test_two_eager_steps_keep_two_tables():
    kv = _StubKV()
    h = _handle(kv=kv)
    _verify(h, [3, 1])
    _verify(h, [2, 4])
    assert [count.delta() for *_, count in kv.deferred] == [-1, -3, -2, 0], (
        "step 2 must not overwrite the table step 1's counts are read from"
    )


class _StubEvent:
    """Notes when it is recorded and waited on, in place of a CUDA event."""

    def __init__(self, log):
        self.log = log

    def record(self):
        self.log.append("record")

    def synchronize(self):
        self.log.append("synchronize")


def test_a_count_waits_for_its_settle_s_copy_and_not_the_rest_of_the_step():
    """GLM-5.2 settles before its MTP seed pass. A count that waited for the whole forward would hold the next step's
    admit until the seed pass ends, where ``main`` admitted and planned that step while the seed pass ran."""
    kv, log = _StubKV(), []
    h = _handle(kv=kv)
    step = _step(h)
    h.begin_step(step)
    h.verify._table.landed = _StubEvent(log)
    k = torch.randn(2 * SPAN, HEADS, DIM)
    for layer in range(LAYERS):
        h.add(layer, k=k, v=k)
    settle(h, torch.tensor([3, 1]))
    log.append("later passes")
    h.check_step(step)
    h.commit_step(step)
    h.end_step(step)
    assert [count.delta() for *_, count in kv.deferred] == [-1, -3]
    assert log == ["record", "later passes", "synchronize"], "the count would wait for the passes after its copy"


def _overwritten_count():
    """Row 0's count of verify step 1, whose table step 3 reused."""
    kv = _StubKV()
    h = _handle(kv=kv)
    for kept in ([3, 1], [2, 2], [1, 1]):
        _verify(h, kept)
    return kv.deferred[0][2]


def test_an_overwritten_count_table_raises():
    with pytest.raises(RuntimeError, match="the count table of verify step 1 was overwritten by step 3"):
        _overwritten_count().delta()


def test_a_captured_step_numbers_its_table_on_replay_only():
    kv = _StubKV()
    h = _handle(kv=kv)
    lease = SlotLease(slot=0, bucket=BucketKey("decode", 2, 2 * SPAN))
    _verify(h, [3, 2], _step(h, ctx=_ctx(lease=lease, capture=True)))
    (table,) = h.verify._tables.values()
    assert kv.deferred == [] and table.host[0] == 0, "a capture defers no count and leaves the number to its replays"
    _verify(h, [], _step(h, ctx=_ctx(lease=lease)))
    assert table.host[0] == 2, "the host numbers the slot's table before each replay"
    assert [count.delta() for *_, count in kv.deferred] == [-1, -2], "a replay's counts are what its graph copied"


@requires_cuda
def test_a_replay_s_counts_are_read_after_its_own_copy_lands():
    """A replay records the count event inside its graph, and the host numbers the table before the launch. A read
    that got ahead of the replay's copy would pass the number check and take the counts the table held from the
    replay before. So each replay here settles counts that differ from the table's last, behind enough work that the
    copy has not landed when the launch returns."""
    kv = _StubKV()
    kv.pages = kv.pages.cuda()
    h = _handle(kv=kv)
    lease = SlotLease(slot=0, bucket=BucketKey("decode", 2, 2 * SPAN))
    kept = torch.zeros(2, dtype=torch.int64, device="cuda")
    k = torch.randn(2 * SPAN, HEADS, DIM, device="cuda")
    work = torch.randn(4096, 4096, device="cuda")
    step = _step(h, ctx=_ctx(lease=lease, capture=True))
    h.begin_step(step)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            work = torch.mm(work, work)
        for layer in range(LAYERS):
            h.add(layer, k=k, v=k)
        settle(h, kept)
    h.check_step(step)
    h.commit_step(step)
    h.end_step(step)
    (table,) = h.verify._tables.values()
    for counts in ([3, 2], [1, 4], [4, 1], [2, 3]):
        assert table.host[1:3].tolist() != counts, "the table already holds these counts, so a stale read would pass"
        kept.copy_(torch.tensor(counts))
        replay = _step(h, ctx=_ctx(lease=lease))
        h.begin_step(replay)
        graph.replay()
        h.check_step(replay)
        h.commit_step(replay)
        h.end_step(replay)
        assert [count.delta() for *_, count in kv.deferred[-2:]] == [count - SPAN for count in counts], (
            "the read took the table's previous counts, ahead of this replay's copy"
        )


def test_the_eager_tables_grow_outside_capture():
    h = _handle()
    _verify(h, [1])
    _verify(h, [1])
    _verify(h, [1, 2, 3], _step(h, spans=(SPAN,) * 3))
    assert max(table.rows for table in h.verify._tables.values()) == 3, "an eager table grows with the step's rows"
    lease = SlotLease(slot=0, bucket=BucketKey("decode", 1, SPAN))
    _verify(h, [1], _step(h, spans=(SPAN,), ctx=_ctx(rows=(1,), lease=lease, capture=True)))
    _refused("its slot's count table holds 1",
             lambda: h.begin_step(_step(h, ctx=_ctx(lease=lease, capture=True))))


class _StubGroup:
    """A second TP rank, which settled ``other``."""

    world_size = 2

    def __init__(self, other):
        self.other = torch.tensor(other)

    def all_gather(self, counts, dim=0):
        return torch.cat([counts, self.other], dim)


def _settled_on_two_ranks(other, monkeypatch):
    monkeypatch.setattr(adapters, "_DEBUG_ASSERTS", True)
    kv = _StubKV()
    kv.tp_group = _StubGroup(other)
    _verify(_handle(kv=kv), [3, 2])
    return kv.deferred[0][2]


def test_tp_ranks_that_settle_alike_apply_their_count(monkeypatch):
    """The control for `test_the_tp_compare_refuses_counts_that_differ`: it passes without the compare too."""
    assert _settled_on_two_ranks([3, 2], monkeypatch).delta() == -1, "ranks that agree apply the count"


def test_the_tp_compare_refuses_counts_that_differ(monkeypatch):
    count = _settled_on_two_ranks([3, 1], monkeypatch)
    with pytest.raises(RuntimeError, match=r"settled verify step 1 differently, \[\[3, 2\], \[3, 1\]\]"):
        count.delta()


# The resources


class _StubTransferManager:
    """No engine, no bytes moved."""

    def __init__(self, *args, **kwargs):
        pass

    def cleanup(self):
        pass

    def remove_request(self, request_id):
        pass


@pytest.fixture
def _kv(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)
    cfg = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=64, max_num_pages=16, page_size=4)
    kv = KVManager(cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
                   device=torch.device("cpu"), dtype=torch.float32)
    for rid in ("a", "b"):
        kv.ingest_request(rid)
    return kv


def _grow(kv, *spans):
    step = KVStep(segments=tuple(Segment(rid, "main", span) for rid, span in zip("ab", spans, strict=False)))
    ctx = StepContext(request_ids=tuple("ab"[:len(spans)]), graph_walk="decode", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _stored(kv, rid="a"):
    return kv._streams[rid]["main"].stored_len


def _hosted(kv, rid="a"):
    (group,) = kv._cpu_pool.offloaded[rid]["main"].groups
    return group.stored_len


def test_a_count_is_applied_at_the_next_admit(_kv):
    _grow(_kv, SPAN)
    _kv.defer_correct_len(_Count(-2, "a"))
    assert _stored(_kv) == SPAN, "a held count waits for the next admit"
    _grow(_kv, 1)
    assert _stored(_kv) == SPAN - 2 + 1, "the admit applies the count before it plans the step"


def test_a_request_left_out_of_the_next_step_still_gets_its_count(_kv):
    _grow(_kv, SPAN, SPAN)
    _kv.defer_correct_len(_Count(-1, "a"))
    _kv.defer_correct_len(_Count(-3, "b"))
    _grow(_kv, 1)
    assert (_stored(_kv, "a"), _stored(_kv, "b")) == (SPAN, 1), (
        "every admit applies every held count, so b's can't wait until a later step reuses its table"
    )
    assert _kv._deferred == {}


@pytest.mark.parametrize("unreadable", ["overwritten", "tp_ranks_apart"])
def test_a_count_read_that_raises_keeps_the_count_held(_kv, monkeypatch, unreadable):
    """The pre-plan admit is the first to read a count, and a raise there is logged and the step planned again
    inline. The inline admit has to find the count still held and raise too, or it plans the step past the tokens
    the verify step rejected."""
    count = _overwritten_count() if unreadable == "overwritten" else _settled_on_two_ranks([3, 1], monkeypatch)
    rope = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"))
    rope.ingest_request("a")
    rope.correct_len("a", "main", SPAN)
    _grow(_kv, SPAN)
    position_step = PositionStep(segments=(Segment("a", "main", 1),))
    ctx = StepContext(request_ids=("a",), graph_walk="decode", slot=0, capture=False)
    admits = ((_kv, lambda: _grow(_kv, 1)), (rope, lambda: rope.admit(position_step, ctx)))
    held = _Count(count, "a")
    for resource, _ in admits:
        resource.defer_correct_len(held)
    for resource, admit in admits:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                admit()
            assert resource._deferred == {"a": held}, "a count read that raises would drop the count"
    assert (_stored(_kv), rope._counters["a"]["main"]) == (SPAN, SPAN), "the pages and the positions would move apart"


def test_a_victim_s_count_is_applied_before_its_offload_or_drop(_kv):
    _grow(_kv, SPAN + 2, SPAN + 2)
    _kv.defer_correct_len(_Count(-2, "a"))
    _kv.defer_correct_len(_Count(-1, "b"))
    _kv.offload("a")
    assert _stored(_kv, "a") == SPAN, "an offload copies the stream at its settled length"
    _kv.check_drop("b", SPAN + 1)
    assert _stored(_kv, "b") == SPAN + 1, "a drop is checked against the settled length"


def test_a_32_row_correction_is_held_with_no_object_per_row(_kv):
    rope = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"))
    count = _Count(-1, *(f"r{i}" for i in range(32)))
    for resource in (_kv, rope):
        alive = _left_alive(lambda resource=resource: resource.defer_correct_len(count))
        assert alive < 4, f"holding one step's correction left {alive} objects alive"
        assert set(resource._deferred.values()) == {count} and len(resource._deferred) == 32


def test_one_request_s_count_is_applied_without_the_rest_of_its_step_s(_kv):
    """A drop check or an offload of one request waits for its own verify step's counts and applies its own."""
    _grow(_kv, SPAN + 2, SPAN + 2)
    count = _Count(-2, "a", "b")
    _kv.defer_correct_len(count)
    _kv.check_drop("a", SPAN)
    assert (_stored(_kv, "a"), _stored(_kv, "b")) == (SPAN, SPAN + 2), "the drop check moved a stream it wasn't asked"
    assert _kv._deferred == {"b": count}
    _grow(_kv, 1)
    assert _stored(_kv, "b") == SPAN, "the next admit applies what the step still holds"


def _runner_step(*spans, preplan=False) -> SubmoduleStep:
    segments = tuple(Segment(rid, "main", span) for rid, span in zip("ab", spans, strict=False))
    step = SubmoduleStep(
        segments=list(segments), steps={"kv": KVStep(segments=segments), "rope": PositionStep(segments=segments)},
    )
    step.set_ctx(StepContext(
        request_ids=tuple("ab"[:len(spans)]), graph_walk="decode", slot=0, capture=False, is_preplan=preplan,
    ))
    return step


@pytest.mark.parametrize("admit", ["admit", "pre_admit"])
def test_a_count_is_applied_at_the_runner_s_admit_and_pre_admit(_kv, admit):
    rope = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"))
    runner = StepRunner({"kv": _kv, "rope": rope})
    for rid in "ab":
        rope.ingest_request(rid)
    step = _runner_step(SPAN, SPAN)
    assert runner.admit(step).ok
    runner.plan(step)
    runner.commit(step)
    for resource in (_kv, rope):
        resource.defer_correct_len(_Count(-2, "a"))
    assert getattr(runner, admit)(_runner_step(1, preplan=admit == "pre_admit")).ok
    assert (_stored(_kv), rope._counters["a"]["main"]) == (SPAN - 2, SPAN - 2), (
        "the next step would be planned past the tokens the verify step rejected"
    )


def test_a_tp_follower_applies_the_count_before_it_replays_rank_0_s_offload(monkeypatch):
    """A follower moves a request to the host only when it replays rank 0's journal, in `apply_resident_delta`."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    leader._cpu_pool, follower._cpu_pool = host._HostPool(64), host._HostPool(64)
    _ingest((leader, follower), "a", 2, arrival=1)
    assert _admits(leader, "a")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for kv in (leader, follower):
        assert _pool_step(kv, "a", 2 * PAGE_SIZE).ok and _pool_step(kv, "a", SPAN).ok
        kv.defer_correct_len(_Count(-2, "a"))
    lead.offload_request(NODE, "a")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    assert follower.residency("a") is Residency.ON_HOST
    assert _hosted(follower) == _hosted(leader) == 2 * PAGE_SIZE + SPAN - 2, (
        "the follower's host copy would keep the rejected tokens rank 0's dropped"
    )


def test_a_guaranteed_request_s_victim_gets_its_count_before_it_moves_to_the_host(monkeypatch):
    """A guaranteed request takes back the pages it lent by moving the preemptible request that borrowed them, older
    or not."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)
    kv = _preemption_pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "o", 3, arrival=0)
    _running(kv, "g", 1, arrival=1, guaranteed=True)
    assert _pool_step(kv, "o", 3 * PAGE_SIZE).ok
    short = _pool_step(kv, "g", 4 * PAGE_SIZE).reason.pages_short
    kv.defer_correct_len(_Count(-2, "o"))
    assert _preempt(kv, "g", short, droppable=lambda rid: True).to_host == ["o"]
    _offload(kv, "o")
    assert _hosted(kv, "o") == 6 * PAGE_SIZE - 2, "the take-back moved o to the host with its rejected tokens"


def test_a_removed_or_reset_request_leaves_no_count(_kv):
    rope = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"))
    _grow(_kv, SPAN, SPAN)
    for rid in "ab":
        rope.ingest_request(rid)
        for resource in (_kv, rope):
            resource.defer_correct_len(_Count(-1, rid))
    for resource in (_kv, rope):
        resource.remove_request("a")
        resource.reset_request("b")
        assert resource._deferred == {}, "a recycled handle must not inherit a count"


def test_position_waits_for_a_pending_count():
    rope = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"))
    for rid in "ab":
        rope.ingest_request(rid)
        rope.correct_len(rid, "main", 8)
        rope.defer_correct_len(_Count(-3, rid))
    assert rope.position("a", "main") == 5, "a position read after a verify step waits for its count"
    rope.admit(PositionStep(segments=(Segment("b", "main", 1),)), StepContext(
        request_ids=("b",), graph_walk="decode", slot=0, capture=False))
    assert rope._counters["b"]["main"] == 5 and rope._deferred == {}, "the next admit applies every held count"


# Verify on a Value


@dataclass(frozen=True)
class _Recurrence(Op):
    """An update of a Value's field, bound to keep a verify block."""

    reads = writes = ("x",)


@dataclass(frozen=True)
class _OtherRecurrence(_Recurrence):
    """A second update type, since a Value holds one update of each type."""


class _KeepsBlocks(adapters.OpBinding):
    """Keeps one verify block above span 1, and records each settle instead of running a kernel."""

    def __init__(self):
        self.settled = []

    def verify_blocks(self, array):
        return {"spec_x": RecurrentBlockConfig(shape=(1,), dtype=torch.int32, per_layer=False)} if (
            array.verify_span > 1) else {}

    def settle(self, run, kept, label):
        self.settled.append((run, kept, label))


def _keeps_blocks(monkeypatch) -> _KeepsBlocks:
    binding = _KeepsBlocks()
    for kind in (_Recurrence, _OtherRecurrence):
        monkeypatch.setitem(adapters.BINDINGS, (adapters.VALUE, kind), binding)
    return binding


def _value(updates, span=2) -> Array:
    return Value("v", nodes={"LLM"}, layers={0, 1}, entry=Entry({"x": TensorType((4,), dtype=torch.float32)}),
                 updates=updates, verify_span=span)


def _value_handle(span=2):
    """A Value whose one update keeps a verify block, and the run its update calls."""
    arrays = [_array(span=1), _value({"a": _Recurrence()}, span)]
    (config,) = (spec.config for spec in build_specs(arrays) if spec.resource_key == "v")
    run = object()
    resources = {"v": RecurrentStatePool(torch.device("cpu"), config, kv_dtype=torch.float32), "a": run}
    return build_handle(arrays[1], resources, _resolve(arrays, StateSetup())), run


def _value_step(h, verify=True) -> SubmoduleStep:
    step = SubmoduleStep(segments=[Segment(1, "main", 2)], state={h.name: ArrayStep(verify=verify)})
    step.set_ctx(_ctx((1,)))
    return step


@pytest.mark.parametrize("updates, found", [
    ({}, "[]"), ({"a": _Recurrence(), "b": _OtherRecurrence()}, "['a', 'b']"),
], ids=["none", "two"])
def test_a_value_verifies_only_through_an_update_with_verify_blocks(monkeypatch, updates, found):
    _keeps_blocks(monkeypatch)
    _refused(re.escape(f"a Value verifies through one update that keeps verify blocks, and the array has {found}"),
             lambda: build_specs([_array(span=1), _value(updates)]))


def test_a_value_s_pool_holds_its_update_s_verify_blocks(monkeypatch):
    _keeps_blocks(monkeypatch)
    (config,) = (spec.config for spec in build_specs([_array(span=1), _value({"a": _Recurrence()})])
                 if spec.resource_key == "v")
    assert set(config.blocks) == {"x", "spec_x"}, "the verify block sits in the Value's pool beside its field"


def test_a_value_handle_attaches_only_its_verify_entry(monkeypatch):
    _keeps_blocks(monkeypatch)
    h, _ = _value_handle()
    assert set(h._runners) == {_Recurrence}, "the verify entry is no update the handle runs"
    assert isinstance(h.verify, adapters.UpdateSettleRun), "the verify entry gives the handle its verify steps' run"
    assert type(_value_handle(span=1)[0].verify) is VerifyRun, "a Value at span 1 refuses verify steps"


def test_a_value_settles_after_its_last_update(monkeypatch):
    binding = _keeps_blocks(monkeypatch)
    h, run = _value_handle()
    kept = torch.tensor([2])
    step = _value_step(h)
    h.begin_step(step)
    h._update(_Recurrence, 0)
    _refused(re.escape("settle before the updates of [('x', 1)]"), lambda: settle(h, kept))
    _refused("a verify step ends with no settle", lambda: h.check_step(step))
    h._update(_Recurrence, 1)
    settle(h, kept)
    _refused("an update after this verify step's settle", lambda: h._update(_Recurrence, 0))
    assert binding.settled == [(run, kept, "main")], "the update's binding settles the step on the update's run"


def test_a_value_that_verifies_is_not_read_whole(monkeypatch):
    _keeps_blocks(monkeypatch)
    h, _ = _value_handle()
    h.begin_step(_value_step(h, verify=False))
    _refused("no layout yet: a read of a Value that verifies", lambda: h.read(0))


# Holes


RANK, HOLE = 16, adapters.HOLE


class _StubHoledKV:
    """MLA pages of one row each, written from the first, so a step's slots are its first rows; records counts."""

    def __init__(self):
        self.pages = torch.zeros(LAYERS, SLOTS, 1, RANK + adapters.HOLE_SLOT)
        self.written = 0
        self.deferred = []

    def layer_view(self, i):
        return self.pages[i]

    def write_kv(self, k, v=None, layer_idx=None, label=None, tokens=None):
        self.pages[layer_idx, :k.shape[0], 0] = k
        self.written = k.shape[0]

    def write_slots(self, label):
        return torch.arange(self.written), torch.zeros(self.written, dtype=torch.long)

    def opens_label(self, rid, label):
        return True

    def defer_correct_len(self, correction):
        self.deferred.append(correction)


class _StubHoledMLA:
    """Records the rope query each read passes instead of attending."""

    requires_kv_write = True

    def __init__(self):
        self.q_pe = []

    def run(self, q, label=None, kv_cache_layer=None, q_pe=None):
        self.q_pe.append(q_pe)
        return q


def _holed(span=SPAN, **overrides) -> Array:
    return Array("kv", **{
        "nodes": {"LLM"}, "layers": frozenset(range(LAYERS)), "entry": LatentEntry(rank=RANK, rope_dim=0),
        "max_len": SLOTS // SPAN, "reads": {"attn": Read(mask=Causal(), op=LatentAttend(HEADS, scale=0.125))},
        "verify_span": span, **overrides,
    })


def _holed_handle(kv=None, attn=None):
    array = _holed()
    return build_handle(array, {"kv": kv or _StubHoledKV(), "attn": attn or _StubHoledMLA()},
                        _resolve([array], StateSetup()))


def _add_all(h, tokens, layers=range(LAYERS)) -> torch.Tensor:
    ckv = torch.randn(tokens, RANK)
    for layer in layers:
        h.add(layer, ckv=ckv)
    return ckv


def test_the_holed_storage_binds_its_read_and_verify_entries():
    assert adapters.storage_of(_holed()) is adapters.HOLED, "a latent with no kpe that verifies keeps holes"
    assert adapters.storage_of(_holed(span=1)) is adapters.LATENT, "one that doesn't verify has none to keep"
    kpe = _holed(entry=LatentEntry(rank=RANK, rope_dim=8))
    assert adapters.storage_of(kpe) is adapters.LATENT, "a latent whose rope key fills the slot rewinds instead"
    bound = {type(each.binding) for each in adapters._bindings(adapters.HOLED, _holed())}
    assert bound == {adapters.HoledLatentAttend, adapters.Holes}, f"the holed storage binds {bound}"
    _refused("positions", lambda: build_specs([_holed(positions=Positions())]))
    _refused("must be LatentAttend", lambda: build_specs([_holed(reads={"attn": Read(mask=Causal())})]))


@pytest.mark.parametrize("mask", [Window(8), Selected(of="keys")], ids=["window", "selected"])
def test_a_holed_array_reads_causally(mask):
    reads = {"attn": Read(mask=mask, op=LatentAttend(HEADS, scale=0.125))}
    _refused("mask must be Causal, the one mask a holed array's marks are written for",
             lambda: build_specs([_holed(reads=reads)]))


def test_a_holed_array_keeps_no_positions():
    _refused("positions, which today's resources don't keep on HoledStorage",
             lambda: build_specs([_holed(positions=Positions())]))


def test_the_holed_pool_holds_max_len_times_the_span():
    (pool,) = (spec for spec in build_specs([_holed()]) if spec.resource_key == "kv")
    config = pool.config
    assert (config.max_seq_len, config.kv_lora_rank) == (SLOTS, RANK), "a row keeps every verify block whole"
    assert (config.head_dim, config.qk_rope_head_dim) == (RANK + 64, 64), "beside a 64-wide slot for the marks"


def test_a_holed_array_keeps_every_verify_row_and_marks_the_rejects():
    kv = _StubHoledKV()
    h = _holed_handle(kv)
    step = _step(h)
    h.begin_step(step)
    ckv = _add_all(h, 2 * SPAN, layers=(0,))
    settle(h, torch.tensor([1, 3]))
    h.add(1, ckv=ckv)
    h.check_step(step)
    h.end_step(step)
    marks = [0.0, HOLE, HOLE, HOLE, 0.0, 0.0, 0.0, HOLE]
    for layer in range(LAYERS):
        rows = kv.pages[layer, :2 * SPAN, 0]
        assert torch.equal(rows[:, :RANK], ckv), f"layer {layer} keeps every row of each verify block"
        assert rows[:, RANK:].eq(torch.tensor(marks)[:, None]).all(), f"layer {layer}'s rejected rows are holes"


def test_a_holed_verify_step_defers_no_count():
    kv = _StubHoledKV()
    h = _holed_handle(kv)
    assert isinstance(h.verify, adapters.HolesRun), "the holed array settles by marks, not by a count table"
    step = _step(h)
    h.begin_step(step)
    _add_all(h, 2 * SPAN)
    settle(h, torch.tensor([2, 2]))
    h.check_step(step)
    h.commit_step(step)
    h.end_step(step)
    assert kv.deferred == [], "a holed array is never rewound"


def test_a_holed_verify_step_fits_its_span_and_settles_once():
    h = _holed_handle()
    _refused(re.escape("a verify step whose rows add [2, 4] entries"), lambda: h.begin_step(_step(h, spans=(SPAN, 2))))
    h = _holed_handle()
    step = _step(h)
    h.begin_step(step)
    _add_all(h, 2 * SPAN)
    _refused("a verify step ends with no settle", lambda: h.check_step(step))
    settle(h, torch.tensor([1, 1]))
    _refused("a second settle in one verify step", lambda: settle(h, torch.tensor([1, 1])))


def test_no_state_op_addresses_a_holed_array():
    assert _holed_handle().takes_fused(None) is False, "no format names the slot a kept entry keeps at zero"


def test_a_holed_read_passes_ones_as_the_rope_query():
    attn = _StubHoledMLA()
    h = _holed_handle(attn=attn)
    h.begin_step(_step(h, verify=False, spans=(1, 1)))
    h.read(0, torch.randn(2, HEADS, RANK))
    (q_pe,) = attn.q_pe
    assert q_pe.shape == (2, HEADS, 64) and q_pe.eq(1).all(), "a hole's mark scores 64 times HOLE"


def _capture_step(h, tokens, capture):
    step = _step(h, verify=False, spans=(tokens,), ctx=_ctx((1,), capture=capture))
    h.begin_step(step)
    return step


def test_the_staging_tail_stays_zero():
    kv = _StubHoledKV()
    h = _holed_handle(kv)
    for tokens, capture in ((2, False), (6, False), (4, True)):
        step = _capture_step(h, tokens, capture)
        _add_all(h, tokens)
        h.end_step(step)
        assert kv.pages[:, :tokens, 0, RANK:].eq(0).all(), f"an add of {tokens} rows (capture={capture}) marks none"


def test_an_eager_step_past_every_bucket_leaves_the_captured_buffers():
    h = _holed_handle()
    step = _capture_step(h, 4, capture=True)
    _add_all(h, 4)
    h.read(0, torch.randn(4, HEADS, RANK))
    h.end_step(step)
    captured = (h._staging[True], h._ones[True])
    step = _capture_step(h, 8, capture=False)
    _add_all(h, 8)
    h.read(0, torch.randn(8, HEADS, RANK))
    h.end_step(step)
    assert h._staging[True] is captured[0] and h._ones[True] is captured[1], "an eager step moves no captured buffer"


def test_the_captured_holed_buffers_never_grow_in_a_recording(monkeypatch):
    h = _holed_handle()
    step = _capture_step(h, 2, capture=True)
    _add_all(h, 2)
    h.end_step(step)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _capture_step(h, 4, capture=True)
    _refused("a recording of 4 rows past the captured buffers' 2", lambda: h.add(0, ckv=torch.randn(4, RANK)))


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9),
                    reason="needs sm90 (FlashInfer MLA)")
@torch.no_grad()
def test_holes_move_flashinfer_mla_by_at_most_one_bf16_ulp():
    """A verify block's queries over a cache whose rejected rows carry HOLE in the rope
    slot, against the same cache without those rows: the query rope part of ones scores a
    real key +0 and a hole about -6.4e5, so only the kernel's split of the keys moves. That
    moves about half the outputs by one bf16 ulp of their row's largest value, as a second
    request in the batch (another split) does too."""
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAWrapper
    from mstar.engine.resources.attn.workspace import WorkspacePool

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    heads, latent, rope, page, block = 8, 512, adapters.HOLE_SLOT, 128, 4
    real = 700
    keep = torch.rand(real + 400) < real / (real + 400)  # holes spread through the context
    keep[-block:] = True  # the query block's own rows are real
    n_all = int(keep.numel())
    ckv = torch.randn(n_all, latent, device=dev, dtype=torch.bfloat16)
    kpe = torch.zeros(n_all, rope, device=dev, dtype=torch.bfloat16)
    kpe[~keep.to(dev)] = HOLE
    q_nope = torch.randn(block, heads, latent, device=dev, dtype=torch.bfloat16)
    ones = torch.ones(block, heads, rope, device=dev, dtype=torch.bfloat16)

    def attend(rows, q_pe):
        n = rows.shape[0]
        pages = -(-n // page)
        cache = torch.zeros(pages, page, latent + rope, device=dev, dtype=torch.bfloat16)
        cache.view(-1, latent + rope)[:n] = torch.cat([ckv, kpe], -1)[rows]
        # main's wrappers take a pool's Workspace (#1142), not a raw buffer
        workspace = WorkspacePool(torch.device(dev), size_mb=128).get("main")
        wrapper = FlashInferMLAWrapper(
            workspace, num_qo_heads=heads, kv_lora_rank=latent, qk_rope_head_dim=rope,
            page_size=page, sm_scale=192 ** -0.5, device=dev)
        assert not wrapper.fallback
        wrapper.plan(
            qo_indptr=torch.tensor([0, block], dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, pages], dtype=torch.int32),
            paged_kv_indices=torch.arange(pages, dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([n - (pages - 1) * page], dtype=torch.int32),
            causal=True, dtype=torch.bfloat16,
        )
        return wrapper.run(q_nope, q_pe, cache).float()

    everything = torch.arange(n_all, device=dev)
    only_real = everything[keep.to(dev)]
    base = attend(only_real, ones)
    # a real key's rope slot is 0, so the rope part of the query is exact
    assert torch.equal(attend(only_real, torch.zeros_like(ones)), base)
    holes = attend(everything, ones)
    # one bf16 ulp of each (query, head) row's largest output
    ulp = torch.exp2(torch.floor(torch.log2(base.abs().amax(-1, keepdim=True))) - 7)
    diff = (holes - base).abs()
    assert (diff <= ulp).all(), (diff / ulp).max()
