"""A mixed step planned a step ahead: the plan thread leases its packed capture
by the spans the step was scheduled with and stages it over rows of those
spans; the GPU thread promotes the stage when the prepared step is the same,
and drops it (and the lease) when the composition changed in between.

Driven through the real ``Engine`` methods and a real ``StepRunner`` over
recording resources; the graph runner and the submodule are fakes that answer
only what the engine asks.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.accelerator_graph_config import (
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AdmitOutcome,
    AllocationFailed,
    KVStep,
    Resource,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.step import BucketKey, SlotLease
from mstar.model.submodule_base import InputSeqLenInfo
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.worker import make_worker
from mstar.worker.scheduled_batch import ScheduledBatch
from mstar.worker.worker import Worker

MIXED = "mixed"
PAD = (-1, -2, -3, -4, -5, -6, -7, -8)


class _Res(Resource):
    """A pre-planning resource that records what each sweep saw."""

    def __init__(self, refuse_pre_admit: bool = False):
        self.calls: list[tuple] = []
        self.cleared = 0
        self._refuse = refuse_pre_admit

    @classmethod
    def build(cls, spec, device, comm_group, **engine_kwargs):
        raise NotImplementedError

    @property
    def supports_preplan(self):
        return True

    def admit(self, step, ctx):
        spans = tuple(seg.span for seg in step.segments)
        self.calls.append(("pre_admit" if ctx.is_preplan else "admit", spans, dict(ctx.chunk_tokens_after)))
        if self._refuse and ctx.is_preplan:
            return AdmitOutcome(ok=False, reason=AllocationFailed("full", 1, "main", 3))
        return AdmitOutcome(ok=True)

    def plan(self, step, ctx):
        self.calls.append(("pre_plan" if ctx.is_preplan else "plan", tuple(seg.span for seg in step.segments)))

    def commit(self, step, ctx):
        self.calls.append(("commit",))

    def clear_preplan(self):
        self.cleared += 1


class _Row(SimpleNamespace):
    """A node input as the engine reads it."""

    def __init__(self, n, walk=None, start=0, total=None):
        super().__init__(
            input_seq_len=n, graph_walk=walk, chunk_start=start,
            chunk_total=n if total is None else total, input_ids=None,
        )

    @property
    def is_final_chunk(self):
        return self.chunk_start + self.input_seq_len >= self.chunk_total


class _Submodule:
    """Declares one segment per row, as a decoder's ``declare_step`` does."""

    def __init__(self):
        self.declared: list[tuple[int, ...]] = []

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, per_request_info=None, **kw):
        spans = tuple(inp.input_seq_len for inp in inputs)
        self.declared.append(spans)
        segments = [Segment(rid, "main", n) for rid, n in zip(request_ids, spans, strict=True)]
        return SubmoduleStep(segments=segments, steps={"kv": KVStep()}, cg_key_info=None)

    def step_is_input_free(self, graph_walk):
        return graph_walk == "decode"

    def cg_key_info(self, graph_walk, per_request_info, per_request_input_metadata=None):
        return None

    def speculative_input_seq_len(self, graph_walk, fwd_info, prev_seq_len):
        return 1 if graph_walk == "decode" else (prev_seq_len if prev_seq_len >= 0 else None)

    def get_input_sequence_len(self, graph_walk, fwd_info, inputs, **kw):
        ids = inputs.get("text_inputs")
        return InputSeqLenInfo(ids[0].shape[0]) if ids else None


class _CG:
    """A graph runner with one packed (mixed) capture and one batched (decode) one."""

    def __init__(self):
        self.leased: list[tuple] = []
        self.released: list[tuple] = []

    def lease_slot(self, graph_walk, bs, slot, num_tokens=None, cg_key_info=None):
        self.leased.append((graph_walk, bs, num_tokens))
        if graph_walk == MIXED:
            if num_tokens is None:
                return None
            bucket = BucketKey(MIXED, 8, 2048 if num_tokens <= 2048 else 4096)
        else:
            bucket = BucketKey("decode", 8, 8)
        return SlotLease(slot=slot % 2, bucket=bucket)

    def is_packed(self, lease):
        return lease.bucket.graph_walk == MIXED

    def declare_rows_for(self, lease, spans):
        return [_Row(n) for n in (*spans, *[0] * (lease.bucket.bs - len(spans)))]

    def declare_inputs_for(self, lease):
        return [_Row(1)] * lease.bucket.bs

    def pad_inputs(self, lease, inputs, uniform=None):
        return [*inputs, *[_Row(0)] * (lease.bucket.bs - len(inputs))]

    def step_ids(self, lease, request_ids):
        return [*request_ids, *PAD[: lease.bucket.bs - len(request_ids)]]

    def plan_stream(self):
        return None

    def release(self, lease, real_bs):
        self.released.append((lease.bucket.num_tokens, real_bs))

    def captures_walk(self, walk):
        return walk in (MIXED, "decode")

    def batched_capture_for(self, walk):
        return walk == "decode"


class _Event:
    def __init__(self, device=None):
        pass

    def record(self, stream=None):
        pass


@pytest.fixture(autouse=True)
def _no_device_events(monkeypatch):
    monkeypatch.setattr(torch, "Event", _Event)


def _engine(resource: _Res | None = None):
    resource = resource or _Res()
    submodule = _Submodule()
    cg = _CG()
    runner = StepRunner({"kv": resource})
    mgmt = make_mgmt(submodule, accelerator_graph_runner=cg, resources={"kv": resource})
    mgmt.lease_slot = lambda: 1
    engine = make_engine(_submodules={"LLM": mgmt}, _runner=runner, _order_after_pre_plans=lambda: None)
    return engine, resource, submodule, cg


def _batch(rows: dict[int, str], chunks=None, tensors=None) -> ExecutingBatch:
    """rid -> the row's own walk, under the mixed walk."""
    ctx = StepContext(request_ids=list(rows), graph_walk=MIXED, slot=1, capture=False, request_walks=dict(rows))
    return ExecutingBatch(
        node_name="LLM",
        per_request_info={rid: SimpleNamespace(rid=rid) for rid in rows},
        step_context=ctx,
        per_request_input_tensors=tensors or {},
        chunk_ranges=chunks or {},
    )


def _prompt(n):
    return {"text_inputs": [torch.zeros(n, dtype=torch.long)]}


def _stage(engine, batch):
    """What the plan thread does for a speculated step: lease, then pre-plan."""
    assert engine.reserve_replay_slot(batch) is not None
    return engine.pre_plan_for_batch(batch)


def _prepare(batch, rows: list[_Row]):
    """What the GPU thread's ``prepare_inputs`` leaves on the batch."""
    batch.register_prepare_batch(rows)
    batch.prepared_rids = tuple(batch.request_ids)


def _exec_admit_and_plan(engine, batch):
    """The GPU thread from the stale check through exec's declare/admit/plan."""
    if engine.preplan_is_stale(batch):
        engine.reset_pre_plan_for_batch(batch)
    if engine._spans_moved(batch):
        engine._drop_span_lease(batch)
    lease = batch.step_context.slot_lease or engine.reserve_replay_slot(batch)
    cg = engine._submodules["LLM"].accelerator_graph_runner
    inputs = cg.pad_inputs(lease, batch.inputs)
    batch.step_context.set_padded_rids(cg.step_ids(lease, batch.request_ids))
    admit, step = engine._declare_and_admit(
        batch, rids=batch.step_context.padded_request_ids, inputs=inputs,
        submodule=engine._submodules["LLM"].submodule, nvtx=False, ctx=batch.step_context, step=batch.step,
    )
    assert admit.ok
    engine._runner.plan(step)
    return lease


# --- the spans a step is planned with --------------------------------------


def test_planned_spans_are_the_chunk_the_decode_token_and_the_popped_prompt():
    engine, *_ = _engine()
    batch = _batch(
        {1: "decode", 2: "decode", 3: "prefill_text", 4: "prefill_text"},
        chunks={4: ChunkRange(1024, 1500, 1500, 476)},
        tensors={3: _prompt(514)},
    )
    assert engine.planned_spans(batch, continuing={1, 2, 4}) == (1, 1, 514, 476)


@pytest.mark.parametrize("case", ["keyed", "rebuild", "unknown"])
def test_planned_spans_leave_a_step_to_plan_inline(case):
    engine, *_ = _engine()
    batch = _batch({1: "decode", 3: "prefill_text"}, tensors={3: _prompt(514)})
    if case == "keyed":
        # the prefix cache decides this row's span when prepare looks it up
        engine._keyed_walks = {"LLM": ("prefill_text",)}
    elif case == "rebuild":
        batch.step_context.rebuild_rids = frozenset({3})
    else:
        batch.per_request_input_tensors = {}
    assert engine.planned_spans(batch, continuing={1}) is None


def test_only_a_walk_with_no_batched_capture_is_planned_by_spans():
    engine, *_ = _engine()
    assert engine.plans_by_spans("LLM", MIXED)
    assert not engine.plans_by_spans("LLM", "decode")
    assert not engine.plans_by_spans("LLM", "vision")


def test_the_graph_runner_tells_batched_captures_from_packed_ones():
    cg = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    cg._capture_configs = [
        BatchedAcceleratorGraphConfig(
            capture_graph_walk="decode", single_request_inputs=_Row(1), capture_batch_sizes=[8],
        ),
        PackedAcceleratorGraphConfig(
            capture_graph_walk=MIXED, capture_token_lengths=[2048], make_node_input=_Row, capture_batch_sizes=[8],
        ),
    ]
    assert cg.batched_capture_for("decode") and not cg.batched_capture_for(MIXED)


def test_declare_rows_keep_one_row_per_one_token_span_and_make_prompts_per_step():
    made = []
    config = PackedAcceleratorGraphConfig(
        capture_graph_walk=MIXED, capture_token_lengths=[2048], capture_batch_sizes=[8],
        make_node_input=lambda n: made.append(n) or _Row(n),
    )
    cg = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    lease = SlotLease(slot=0, bucket=BucketKey(MIXED, 8, 2048))
    cg._buckets = {lease.bucket: SimpleNamespace(config=config)}
    cg._span_rows = {}
    first = cg.declare_rows_for(lease, (1, 1, 514))
    assert [r.input_seq_len for r in first] == [1, 1, 514, 0, 0, 0, 0, 0]
    second = cg.declare_rows_for(lease, (1, 300))
    assert [r.input_seq_len for r in second] == [1, 300, 0, 0, 0, 0, 0, 0]
    # a decode row and a padding row are made once; each prompt span per step
    assert sorted(made) == [0, 1, 300, 514]
    assert first[0] is second[0] and first[3] is second[2]


# --- stage, promote, drop ---------------------------------------------------


def test_a_mixed_step_is_staged_over_its_spans_and_promoted():
    engine, res, submodule, cg = _engine()
    batch = _batch({1: "decode", 2: "decode", 3: "prefill_text"}, tensors={3: _prompt(514)})
    batch.planned_spans = engine.planned_spans(batch, continuing={1, 2})
    assert _stage(engine, batch)
    assert cg.leased == [(MIXED, 3, 516)]
    assert batch.lease_spans == (1, 1, 514) and batch.preplanned_rids == (1, 2, 3)
    assert res.calls == [("pre_admit", (1, 1, 514, 0, 0, 0, 0, 0), {}), ("pre_plan", (1, 1, 514, 0, 0, 0, 0, 0))]
    # the declaration over the scheduled spans is not kept: exec declares the real one
    assert batch.step is None and engine._runner.staged

    _prepare(batch, [_Row(1, "decode"), _Row(1, "decode"), _Row(514, "prefill_text")])
    res.calls.clear()
    lease = _exec_admit_and_plan(engine, batch)
    # the same lease, admit covered by the pre-admit, the stage promoted
    assert lease.bucket.num_tokens == 2048 and len(cg.leased) == 1
    assert res.calls == [("plan", (1, 1, 514, 0, 0, 0, 0, 0))]
    assert res.cleared == 0 and not engine._runner.staged


def test_a_row_that_finished_drops_the_stage_and_the_lease():
    engine, res, _, cg = _engine()
    batch = _batch({1: "decode", 2: "decode", 3: "prefill_text"}, tensors={3: _prompt(514)})
    batch.planned_spans = engine.planned_spans(batch, continuing={1, 2})
    assert _stage(engine, batch)
    # row 2 had no loop-back output from the step in flight
    batch.drop_rids({2})
    _prepare(batch, [_Row(1, "decode"), _Row(514, "prefill_text")])
    res.calls.clear()
    _exec_admit_and_plan(engine, batch)
    assert res.cleared == 1
    # leased again by the prepared rows, and admitted and planned inline
    assert cg.leased[-1] == (MIXED, 2, 515)
    assert [c[0] for c in res.calls] == ["admit", "plan"]


def test_a_prompt_that_came_out_another_length_drops_the_stage_and_the_lease():
    engine, res, _, cg = _engine()
    batch = _batch({1: "decode", 3: "prefill_text"}, tensors={3: _prompt(2000)})
    batch.planned_spans = engine.planned_spans(batch, continuing={1})
    assert _stage(engine, batch)
    assert batch.step_context.slot_lease.bucket.num_tokens == 2048
    # the rows are the same, the token count is not (and no longer fits the bucket)
    _prepare(batch, [_Row(1, "decode"), _Row(2100, "prefill_text")])
    assert engine.preplan_is_stale(batch)
    res.calls.clear()
    lease = _exec_admit_and_plan(engine, batch)
    assert res.cleared == 1 and lease.bucket.num_tokens == 4096
    assert cg.released and [c[0] for c in res.calls] == ["admit", "plan"]


def test_a_lease_by_spans_with_no_stage_is_dropped_when_the_spans_move():
    # the pre-plan leased the slot and then declared nothing to stage
    engine, res, submodule, cg = _engine()
    batch = _batch({1: "decode", 3: "prefill_text"}, tensors={3: _prompt(514)})
    batch.planned_spans = engine.planned_spans(batch, continuing={1})
    assert engine.reserve_replay_slot(batch) is not None and batch.lease_spans == (1, 514)
    _prepare(batch, [_Row(1, "decode"), _Row(600, "prefill_text")])
    assert not engine.preplan_is_stale(batch)  # no stage to be stale
    assert engine._spans_moved(batch)
    engine._drop_span_lease(batch)
    assert batch.step_context.slot_lease is None and batch.lease_spans is None
    # leased by the prepared inputs, the lease holds no spans to check
    assert engine.reserve_replay_slot(batch).bucket.num_tokens == 2048
    assert batch.lease_spans is None and not engine._spans_moved(batch)


def test_a_refused_pre_admit_drops_the_stage_and_exec_admits_inline():
    engine, res, _, cg = _engine(_Res(refuse_pre_admit=True))
    batch = _batch({1: "decode", 3: "prefill_text"}, tensors={3: _prompt(514)})
    batch.planned_spans = engine.planned_spans(batch, continuing={1})
    assert not _stage(engine, batch)
    assert batch.preplanned_rids is None and batch.step_context.slot_lease is None
    assert not engine._runner.staged and res.cleared == 1
    _prepare(batch, [_Row(1, "decode"), _Row(514, "prefill_text")])
    res.calls.clear()
    _exec_admit_and_plan(engine, batch)
    assert [c[0] for c in res.calls] == ["admit", "plan"]


def test_a_chunked_prompt_is_staged_across_two_mixed_steps():
    engine, res, *_ = _engine()
    # first chunk of a 1500-token prompt: the stage reserves the rest as exec would
    first = _batch(
        {1: "decode", 3: "prefill_text"},
        chunks={3: ChunkRange(0, 1024, 1500, 1024)}, tensors={3: _prompt(1500)},
    )
    first.planned_spans = engine.planned_spans(first, continuing={1})
    assert first.planned_spans == (1, 1024)
    assert _stage(engine, first)
    assert res.calls[0] == ("pre_admit", (1, 1024, 0, 0, 0, 0, 0, 0), {3: 476})
    _prepare(first, [_Row(1, "decode"), _Row(1024, "prefill_text", 0, 1500)])
    res.calls.clear()
    _exec_admit_and_plan(engine, first)
    assert res.calls == [("plan", (1, 1024, 0, 0, 0, 0, 0, 0))] and res.cleared == 0
    engine._runner.commit(first.step if first.step is not None else _declared(engine, first))

    # its last chunk, continuing mid-prefill beside the decode row: nothing left to reserve
    second = _batch(
        {1: "decode", 3: "prefill_text"}, chunks={3: ChunkRange(1024, 1500, 1500, 476)},
    )
    second.planned_spans = engine.planned_spans(second, continuing={1, 3}, prev=first)
    assert second.planned_spans == (1, 476)
    res.calls.clear()
    assert _stage(engine, second)
    assert res.calls[0] == ("pre_admit", (1, 476, 0, 0, 0, 0, 0, 0), {})
    _prepare(second, [_Row(1, "decode"), _Row(476, "prefill_text", 1024, 1500)])
    res.calls.clear()
    _exec_admit_and_plan(engine, second)
    assert res.calls == [("plan", (1, 476, 0, 0, 0, 0, 0, 0))] and res.cleared == 0


def _declared(engine, batch):
    cg = engine._submodules["LLM"].accelerator_graph_runner
    lease = batch.step_context.slot_lease
    step = engine._submodules["LLM"].submodule.declare_step(
        MIXED, batch.step_context.padded_request_ids, cg.pad_inputs(lease, batch.inputs),
    )
    step.set_ctx(batch.step_context)
    return step


# --- the worker -------------------------------------------------------------


def test_the_main_thread_fixes_spans_only_for_a_step_leased_by_tokens():
    engine, *_ = _engine()
    worker = make_worker(worker_id=0, device=torch.device("cpu"))
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    mixed = _batch({1: "decode", 3: "prefill_text"}, tensors={3: _prompt(514)})
    scheduled = ScheduledBatch(node_name="LLM", graph_walk=MIXED, request_walks={1: "decode", 3: "prefill_text"})
    Worker._plan_spans(worker, None, SimpleNamespace(scheduled_batch=scheduled, node_batch=mixed, continuing_rids={1}))
    assert mixed.planned_spans == (1, 514)
    decode = _batch({1: "decode", 2: "decode"})
    # a decode step asks the engine nothing
    worker.engine_manager = None
    Worker._plan_spans(worker, None, SimpleNamespace(
        scheduled_batch=ScheduledBatch(node_name="LLM", graph_walk="decode"), node_batch=decode, continuing_rids={1, 2},
    ))
    assert decode.planned_spans is None
