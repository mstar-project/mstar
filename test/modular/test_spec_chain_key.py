"""The fairness peek and the yield-away key on the walk a chain speculates into.

After a mixed step the step's walk is the combined one, which no row has, so
keying on it would count the chain's own parked rows as other work.
"""

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import Residency
from mstar.engine.resources.step import FULL_ADMIT_OK
from mstar.graph.runtime.base import (
    ColumnarEdgeSpecs,
    PopRidsOutput,
    ReadyNodeSpec,
    SpeculationOutput,
)
from mstar.model.submodule_base import StepOrder
from mstar.testing_utils.worker import bound, make_scheduler, make_worker
from mstar.utils.containers import ParallelList
from mstar.worker import step_pipeline as pipeline_mod
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.scheduling_policy import RoundRobin
from mstar.worker.speculation import SpeculationTarget
from mstar.worker.stream_buffers import StreamBufferManager


def _speculator(target, other_ready=True, speculation=None, mid_prefill_walks=frozenset(), budgeted=True, peek=True):
    peeked, excluded = [], []
    speculator = make_worker(
        chunk_loops=ChunkLoops(mid_prefill_walks),
        peek=peek,
        max_consecutive_spec=8,
        scheduler=SimpleNamespace(
            has_ready_excluding=lambda key, among=None: peeked.append(key) or other_ready,
            next_batch=lambda exclude_target: excluded.append(exclude_target),  # noqa: PLW0108
            chunks=lambda node_name, graph_walk: budgeted,
            policy=RoundRobin(),
        ),
    ).speculator
    scheduler = speculator.scheduler
    scheduler.chain_yields = MicroScheduler.chain_yields.__get__(scheduler)
    speculator.can_speculate = lambda batch: True
    speculator._choose_spec_target = lambda batch: target
    speculator._try_speculate_next = lambda pending, target: speculation
    return speculator, peeked, excluded


def _build(speculator, consecutive):
    """``build``, for a chain that has run ``consecutive`` steps."""
    speculator._consecutive = consecutive
    return speculator.build(_pending())


def _chain_step(node_name="LLM"):
    return SimpleNamespace(node_batch=SimpleNamespace(node_name=node_name), tp_seq=-1, is_yield_away=False)


def _pending():
    return SimpleNamespace(
        batch=object(), node_name="LLM", graph_walk="mixed",
        node_batch=SimpleNamespace(admit_error=None), tp_seq=-1,
    )


def test_after_a_mixed_step_the_chain_keys_on_its_target_walk():
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    speculator, peeked, excluded = _speculator(target)

    _, yielded_from = _build(speculator, 1)

    assert peeked == [("LLM", "decode")]
    assert yielded_from == ("LLM", "decode") and excluded == [("LLM", "decode")]


def test_with_no_target_the_chain_keys_on_the_step():
    speculator, peeked, _ = _speculator(None)

    _build(speculator, 1)

    assert peeked == [("LLM", "mixed")]


_PROMPT = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="prefill"), [0])


def test_a_chains_first_step_that_cannot_continue_yields_to_ready_work():
    """A prompt-only chunk step is never speculated; the step after it goes to
    the decode rows waiting, not to another prompt's chunk."""
    speculator, peeked, excluded = _speculator(_PROMPT)

    _, yielded_from = _build(speculator, 0)

    assert peeked == [("LLM", "prefill")]
    assert yielded_from == ("LLM", "prefill") and excluded == [("LLM", "prefill")]


def test_a_chains_first_step_that_cannot_continue_keeps_the_worker_busy_alone():
    speculator, _, excluded = _speculator(_PROMPT, other_ready=False)

    _, yielded_from = _build(speculator, 0)

    assert yielded_from is None and excluded == [None]


def test_a_chains_first_step_that_continues_does_not_peek():
    speculator, peeked, excluded = _speculator(
        _PROMPT, speculation=_chain_step(),
    )

    speculation, _ = _build(speculator, 0)

    assert speculation is not None and not peeked and not excluded


def test_a_prompt_chain_that_speculates_its_next_chunk_yields_from_its_first_step():
    """Speculating a row's next chunk must not hold decode rows behind a second
    prompt step, so the peek runs before the chain's first speculation."""
    chunk = _chain_step()
    speculator, peeked, excluded = _speculator(
        _PROMPT, speculation=chunk, mid_prefill_walks=frozenset({("LLM", "prefill")}),
    )

    speculation, yielded_from = _build(speculator, 0)

    assert speculation is not chunk and peeked == [("LLM", "prefill")]
    assert yielded_from == ("LLM", "prefill") and excluded == [("LLM", "prefill")]


def test_a_prompt_chain_alone_speculates_its_next_chunk():
    chunk = _chain_step()
    speculator, _, excluded = _speculator(
        _PROMPT, other_ready=False, speculation=chunk,
        mid_prefill_walks=frozenset({("LLM", "prefill")}),
    )

    speculation, _ = _build(speculator, 0)

    assert speculation is chunk and not excluded


def test_without_the_peek_a_step_that_cannot_continue_does_not_yield():
    speculator, peeked, excluded = _speculator(_PROMPT, peek=False)

    _build(speculator, 0)

    assert not peeked and excluded == [None]


def test_a_step_with_no_token_budget_that_cannot_continue_keeps_mains_order():
    """Its prompts ran whole (``max_batch_tokens: null``, or a model that never
    chunks), so there is no chunk step to yield after."""
    speculator, peeked, excluded = _speculator(_PROMPT, budgeted=False)

    _build(speculator, 0)

    assert not peeked and excluded == [None]


def test_a_prompt_chain_with_no_token_budget_does_not_yield_from_its_first_step():
    """Inside its chunk Loop but never cut: it speculates on as on main."""
    chunk = _chain_step()
    speculator, peeked, excluded = _speculator(
        _PROMPT, speculation=chunk, mid_prefill_walks=frozenset({("LLM", "prefill")}), budgeted=False,
    )

    speculation, _ = _build(speculator, 0)

    assert speculation is chunk and not peeked and not excluded


def test_the_cap_counts_a_yield_to_another_capture_group_of_the_same_walk():
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    speculator, _, excluded = _speculator(target, peek=False)
    speculator._max_consecutive = 2
    batch = SimpleNamespace(node_name="LLM", graph_walk="decode", tp_seq=-1)
    yield_away = SimpleNamespace(
        is_yield_away=True, gave_way=False, scheduled_batch=batch, node_batch=SimpleNamespace(),
    )
    speculator.scheduler.next_batch = lambda exclude_target: excluded.append(exclude_target) or batch
    speculator._yield_away_speculation = lambda picked: yield_away
    speculator.lockstep.broadcast = lambda node_batch: -1

    for _ in range(2):
        launched, _ = speculator.build(_pending())
        assert launched is yield_away
        speculator.note_launched(launched)

    _, yielded_from = speculator.build(_pending())
    assert yielded_from == ("LLM", "decode")
    assert excluded == [None, None, ("LLM", "decode")]


def test_the_cap_restarts_after_another_walk_runs():
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    speculator, _, excluded = _speculator(target, peek=False)
    speculator._max_consecutive = 2
    speculator.build(_pending())

    same = SimpleNamespace(
        is_yield_away=True, gave_way=False,
        scheduled_batch=SimpleNamespace(node_name="LLM", graph_walk="decode"),
    )
    other = SimpleNamespace(
        is_yield_away=True, gave_way=False,
        scheduled_batch=SimpleNamespace(node_name="audio", graph_walk="emit"),
    )
    speculator.note_launched(same)
    speculator.note_launched(other)
    _, yielded_from = speculator.build(_pending())
    assert yielded_from is None and excluded[-1] is None
    speculator.chain_ended()
    _, yielded_from = speculator.build(_pending())
    assert yielded_from is None and excluded[-1] is None


def test_another_capture_group_keeps_the_default_peek_cadence():
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0, 1])
    speculator, peeked, _ = _speculator(target, peek=True, budgeted=False)
    speculator.build(_pending())
    speculator.note_launched(SimpleNamespace(
        is_yield_away=True, gave_way=False,
        scheduled_batch=SimpleNamespace(node_name="LLM", graph_walk="decode"),
    ))

    speculator.build(_pending())
    assert peeked == []


def test_a_direct_step_on_another_walk_restarts_the_cap():
    target = [SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0])]
    speculator, _, _ = _speculator(target[0], peek=False)
    speculator._max_consecutive = 2
    speculator._choose_spec_target = lambda batch: target[0]
    speculator._try_speculate_next = lambda pending, picked: _chain_step(picked.info.node_name)

    launched, _ = speculator.build(_pending())
    assert launched.node_batch.node_name == "LLM"
    speculator.note_launched(launched)

    target[0] = SpeculationTarget(SimpleNamespace(node_name="audio", graph_walk="emit"), [0])
    launched, _ = speculator.build(_pending())
    assert launched.node_batch.node_name == "audio"
    speculator.note_launched(launched)

    # The first audio step did not continue the LLM chain.
    launched, yielded_from = speculator.build(_pending())
    assert launched.node_batch.node_name == "audio" and yielded_from is None


# ── End-to-end multi-step CPU tests (real MicroScheduler + Speculator) ──


class _MiniGraphRuntime:
    """Minimal pure-Python GraphRuntime stand-in for multi-step fairness tests."""

    def __init__(self, async_walks=frozenset({("T3", "decode")}), looping_walks=frozenset({("T3", "decode")})):
        self.async_walks = set(async_walks)
        self.looping_walks = set(looping_walks)
        # rid -> set of (node_name, graph_walk)
        self.ready: dict[int, set[tuple[str, str]]] = {}
        self.in_flight: set[tuple[int, str]] = set()
        self.combined_walk_of: dict[tuple[str, str], str] = {}

    def add_ready(self, rid: int, node_name: str, graph_walk: str) -> None:
        if (rid, node_name) not in self.in_flight:
            self.ready.setdefault(rid, set()).add((node_name, graph_walk))

    def _matches(self, target, node_name, walk):
        return target[0] == node_name and target[1] in (
            walk, self.combined_walk_of.get((node_name, walk)),
        )

    def _scan(self, exclude_rids, target, exclude_target):
        for rid in sorted(self.ready):
            if rid in exclude_rids:
                continue
            for node_name, walk in sorted(self.ready[rid]):
                if target is not None and not self._matches(target, node_name, walk):
                    continue
                if exclude_target is not None and self._matches(exclude_target, node_name, walk):
                    continue
                yield node_name, walk, rid

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        grouped: dict[tuple[str, str], list[int]] = {}
        for node_name, walk, rid in self._scan(exclude_rids, target, exclude_target):
            grouped.setdefault((node_name, walk), []).append(rid)
        return [ReadyNodeSpec(n, w, rids) for (n, w), rids in grouped.items()]

    def has_ready_excluding(self, exclude_rids, exclude_target=None):
        return any(True for _ in self._scan(exclude_rids, None, exclude_target))

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        del node_name, graph_walk
        return 0

    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        del check_ready
        rids = []
        for rid in request_ids:
            if (node_name, graph_walk) in self.ready.get(rid, ()):
                self.ready[rid].remove((node_name, graph_walk))
                rids.append(rid)
        return PopRidsOutput(
            wg_ids=ParallelList(rids, [0] * len(rids)),
            input_edges=ColumnarEdgeSpecs.empty(),
            output_signals=("out",),
        )

    def push_back_node(self, node_name, rids, wg_ids):
        del wg_ids
        # Re-readies on the single walk known for `node_name` when pushed back.
        for rid in rids:
            walk = next((w for (n, w) in self.async_walks if n == node_name), "decode")
            self.add_ready(rid, node_name, walk)

    def set_in_flight(self, node_name, rids, wg_ids, in_flight):
        del wg_ids
        for rid in rids:
            if in_flight:
                self.in_flight.add((rid, node_name))
            else:
                self.in_flight.discard((rid, node_name))

    def is_async_schedulable(self, node_name, graph_walk):
        return (node_name, graph_walk) in self.async_walks

    def speculate_node(self, curr_node, walks, rows, counts, loop_strides):
        del curr_node, rows, counts, loop_strides
        out = []
        for walk in walks:
            # Find matching looping target for curr_node/walk
            for n, w in sorted(self.looping_walks):
                if w == walk:
                    out.append(SpeculationOutput(
                        node_name=n, graph_walk=w, is_new_loop_iter=True,
                        loop_name="loop", output_signals=("out",),
                    ))
        return out

    def prep_spec_rows(self, spec_node, curr_node, graph_walk, candidates, max_continuing, loop_strides):
        del spec_node, curr_node, graph_walk, loop_strides
        kept = list(candidates) if max_continuing is None else list(candidates[:max_continuing])
        return SimpleNamespace(ready_rids=kept, spec_id=1, wg_id=0, input_edges=None)

    def get_consumed_edges(self, source_node, dest_node, graph_walk):
        del source_node, graph_walk
        return {("out", dest_node)}

    def get_output_signals(self, node_name, graph_walk):
        del node_name, graph_walk
        return ["out"]

    def commit_speculation(self, spec_id, success, dropped_rids, *, node=None, wg_ids=(), scheduled_rids=()):
        del spec_id, success, dropped_rids
        if node is not None and scheduled_rids:
            self.set_in_flight(node, scheduled_rids, wg_ids, True)

    def get_rid_string(self, rid):
        return str(rid)


class _MiniEngine:
    def __init__(self, groups: dict[int, bool], order: StepOrder = StepOrder.ARRIVAL, max_bs: int = 32):
        self.groups = dict(groups)
        self.order = order
        self.max_bs = max_bs

    def get_step_order(self, node_name, graph_walk):
        del node_name, graph_walk
        return self.order

    def get_max_batch_size(self, node_name, graph_walk):
        del node_name, graph_walk
        return self.max_bs

    def batch_cap_varies(self, node_name, graph_walk):
        del node_name, graph_walk
        return False

    def get_max_batch_tokens(self, node_name, graph_walk):
        del node_name, graph_walk

    def get_max_rows_in_combined(self, node_name, graph_walk):
        del node_name, graph_walk

    def supports_chunked_prefill(self, node_name, graph_walk):
        del node_name, graph_walk
        return False

    def declares_input_sequence_len(self, node_name):
        del node_name
        return False

    def splits_by_capture_key(self, node_name, graph_walk):
        return (node_name, graph_walk) == ("T3", "decode")

    def capture_group(self, node_name, graph_walk, rid, fwd_info, input_metadata=None):
        del node_name, graph_walk, fwd_info, input_metadata
        return self.groups.get(rid)

    def residency(self, node_name, rid):
        del node_name, rid
        return Residency.RESIDENT

    def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
        del node_name, rid, fwd_info, allow_reload
        return FULL_ADMIT_OK

    def prefix_cache_open(self, node_name):
        del node_name
        return False

    def release_cached_prefix(self, node_batch):
        del node_batch


class _MiniRequestState:
    def __init__(self, groups: dict[int, bool]):
        self.per_request_info = {}
        self.groups = dict(groups)

    def get_partition_for_node(self, node_name):
        del node_name
        return "default"

    def get_fwd_info(self, rid, partition):
        del partition
        return SimpleNamespace(dynamic_loop_iter_counts={}, cfg=self.groups.get(rid))

    def fwd_infos(self, request_ids, partition):
        return [self.get_fwd_info(rid, partition) for rid in request_ids]

    def ingested_stream_chunks(self, rid, node_name):
        del rid, node_name
        return {}


def _build_multi_step_worker(
    monkeypatch,
    *,
    groups: dict[int, bool],
    order: StepOrder = StepOrder.ARRIVAL,
    peek: bool = True,
    cap: int = 8,
    async_walks=frozenset({("T3", "decode")}),
    looping_walks=frozenset({("T3", "decode")}),
    on_finish=None,
):
    """Build a real Worker (MicroScheduler + FIFOBatchBuilder + Speculator + StepPipeline)
    over pure-Python graph/engine stand-ins with GPU execution stubbed."""
    runtime = _MiniGraphRuntime(async_walks=async_walks, looping_walks=looping_walks)
    engine = _MiniEngine(groups, order=order)
    engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    request_state = _MiniRequestState(groups)
    tensor_manager = SimpleNamespace(
        get_tensor=lambda uuid: torch.zeros(1),
        get_tensors=lambda uuids: [torch.zeros(1)] * len(uuids),
    )
    streams = StreamBufferManager(request_state, tensor_manager, [], None, graph_runtime=runtime)
    scheduler = make_scheduler(
        engine_manager=engine_manager,
        request_state=request_state,
        graph_runtime=runtime,
        tensor_manager=tensor_manager,
    )
    worker = make_worker(
        graph_runtime=runtime,
        engine_manager=engine_manager,
        request_state=request_state,
        tensor_manager=tensor_manager,
        streams=streams,
        scheduler=scheduler,
        communicator=SimpleNamespace(wait_for_work=lambda ms: None),
        peek=peek,
        max_consecutive_spec=cap,
    )
    bound(worker.scheduler, request_state, runtime=runtime)
    for rid in groups:
        request_state.per_request_info[rid] = object()
        worker.lifecycle.add(rid, str(rid))

    executed: list[tuple[str, str, tuple[int, ...]]] = []

    def submit(batch, node_batch, **kwargs):
        del kwargs
        runtime.set_in_flight(batch.node_name, list(batch.request_to_worker_graph), [], True)
        node_batch.launch_started_event = threading.Event()
        node_batch.launch_started_event.set()
        fut = Future()
        fut.set_result(SimpleNamespace(
            rows_only=True,
            row_request_ids=list(batch.request_to_worker_graph),
            loopback_rids=frozenset(),
        ))
        executed.append((batch.node_name, batch.graph_walk, tuple(batch.request_to_worker_graph)))
        return pipeline_mod.PendingBatch(
            batch=batch, node_batch=node_batch,
            node_name=batch.node_name, partition="default", graph_walk=batch.graph_walk,
            future=fut,
        )

    def clear_in_flight(batch):
        runtime.set_in_flight(batch.node_name, list(batch.request_to_worker_graph), [], False)

    def postprocess(step, outputs, **kwargs):
        del outputs, kwargs
        if on_finish is not None:
            on_finish(runtime, step)

    for stub_name in ("arm_preplan", "reset_skip_plan_flags"):
        setattr(worker.executor, stub_name, lambda *a, **kw: None)
    worker.executor.clear_in_flight = clear_in_flight
    worker.executor._set_in_flight = lambda batch, value: runtime.set_in_flight(
        batch.node_name, list(batch.request_to_worker_graph), [], value,
    )
    worker.executor.submit = submit
    worker.executor.postprocess = postprocess
    for stub_name in ("process_messages", "check_ready_tensors", "poll_stream_buffers"):
        setattr(worker.ingest, stub_name, lambda *a, **kw: None)
    worker.lockstep.broadcast = lambda *a, **kw: -1
    worker.lockstep.maybe_broadcast_nospec = lambda *a, **kw: None
    worker.pipeline._handle_error = lambda exc, *a: (_ for _ in ()).throw(exc)
    return worker, runtime, executed


def _label_step(groups, step):
    node, walk, rids = step
    if (node, walk) == ("T3", "decode"):
        return f"T{1 if groups[rids[0]] else 0}"
    return "O"


@pytest.mark.parametrize(("order", "cap", "expected_first_other"), [
    (StepOrder.ARRIVAL, 2, 3),
    (StepOrder.ARRIVAL, 8, 9),
    (StepOrder.TURNS, 2, 4),
    (StepOrder.TURNS, 8, 10),
])
def test_two_capture_groups_with_peek_off_yield_to_other_within_cap(
    order, cap, expected_first_other,
):
    """Issue #486 regression: with MSTAR_SPEC_PEEK_FOR_FAIRNESS=0, two capture
    groups in T3 decode must still yield to waiting s3gen within bounded steps."""
    groups = {0: False, 1: True, 2: False}

    def on_finish(runtime, step):
        for rid in step.batch.request_to_worker_graph:
            runtime.add_ready(rid, step.node_name, step.graph_walk)

    worker, runtime, executed = _build_multi_step_worker(
        None, groups=groups, order=order, peek=False, cap=cap, on_finish=on_finish,
    )
    runtime.add_ready(0, "T3", "decode")
    runtime.add_ready(1, "T3", "decode")
    runtime.add_ready(2, "s3gen", "s3gen_chunk")

    for _ in range(25):
        worker.pipeline._pass()

    labels = [_label_step(groups, s) for s in executed]
    assert "O" in labels, f"s3gen starved: {labels}"
    assert labels.index("O") == expected_first_other, labels
    assert labels.count("O") >= 2, f"s3gen ran only once: {labels}"


@pytest.mark.parametrize(("order", "cap", "expected_prefix"), [
    # Under ARRIVAL (Chatterbox T3's StepOrder), _left_to_its_chain skips
    # T3's backlog on peek yield, yielding immediately to s3gen (O).
    (StepOrder.ARRIVAL, 1024, ["T0", "T1", "T1", "O", "T0", "T1", "T1", "O"]),
    # Under TURNS, _left_to_its_chain drains the parked capture group before
    # yielding to s3gen, so peek alone keeps alternating T0/T1 until the cap fires.
    (StepOrder.TURNS, 4, ["T0", "T1", "T1", "T0", "T0", "T1", "O"]),
])
def test_two_capture_groups_with_default_peek_preserve_cadence(order, cap, expected_prefix):
    """Default peek=True preserves T0 T1 T1 O under ARRIVAL (Chatterbox T3)
    and drains parked TURNS groups until cap."""
    groups = {0: False, 1: True, 2: False}

    def on_finish(runtime, step):
        for rid in step.batch.request_to_worker_graph:
            runtime.add_ready(rid, step.node_name, step.graph_walk)

    worker, runtime, executed = _build_multi_step_worker(
        None, groups=groups, order=order, peek=True, cap=cap, on_finish=on_finish,
    )
    runtime.add_ready(0, "T3", "decode")
    runtime.add_ready(1, "T3", "decode")
    runtime.add_ready(2, "s3gen", "s3gen_chunk")

    for _ in range(12):
        worker.pipeline._pass()

    labels = [_label_step(groups, s) for s in executed]
    assert labels[:len(expected_prefix)] == expected_prefix, labels


def test_other_node_ready_mid_chain_runs_within_cap_when_peek_off():
    """An other node becoming ready mid-chain runs within `cap` steps."""
    groups = {0: False, 1: True, 2: False}

    def on_finish(runtime, step):
        for rid in step.batch.request_to_worker_graph:
            runtime.add_ready(rid, step.node_name, step.graph_walk)

    worker, runtime, executed = _build_multi_step_worker(
        None, groups=groups, order=StepOrder.ARRIVAL, peek=False, cap=4, on_finish=on_finish,
    )
    runtime.add_ready(0, "T3", "decode")
    runtime.add_ready(1, "T3", "decode")

    for _ in range(3):
        worker.pipeline._pass()
    assert all(node == "T3" for node, _, _ in executed)

    ready_at = len(executed)
    runtime.add_ready(2, "s3gen", "s3gen_chunk")
    for _ in range(8):
        worker.pipeline._pass()

    other_indices = [i for i, (node, _, _) in enumerate(executed) if node == "s3gen"]
    assert other_indices and other_indices[0] - ready_at <= 4, executed


def test_same_node_different_walk_prefill_runs_within_cap_when_peek_off():
    """With peek=False, two capture groups on (T3, decode) yield to (T3, prefill)
    (a different walk on the same node) within bounded steps."""
    groups = {0: False, 1: True, 2: False}

    def on_finish(runtime, step):
        for rid in step.batch.request_to_worker_graph:
            runtime.add_ready(rid, step.node_name, step.graph_walk)

    worker, runtime, executed = _build_multi_step_worker(
        None, groups=groups, order=StepOrder.ARRIVAL, peek=False, cap=2, on_finish=on_finish,
    )
    runtime.add_ready(0, "T3", "decode")
    runtime.add_ready(1, "T3", "decode")
    runtime.add_ready(2, "T3", "prefill")

    for _ in range(12):
        worker.pipeline._pass()

    prefill_steps = [i for i, (node, walk, _) in enumerate(executed) if (node, walk) == ("T3", "prefill")]
    assert len(prefill_steps) >= 2 and prefill_steps[0] <= 4, executed


def test_other_to_decode_transition_restarts_cap_cleanly():
    """When s3gen (target=None) yields back to T3/decode, T3/decode gets its
    full `cap` steps before yielding again."""
    groups = {0: False, 1: True, 2: False}

    def on_finish(runtime, step):
        for rid in step.batch.request_to_worker_graph:
            runtime.add_ready(rid, step.node_name, step.graph_walk)

    worker, runtime, executed = _build_multi_step_worker(
        None, groups=groups, order=StepOrder.ARRIVAL, peek=False, cap=3, on_finish=on_finish,
    )
    runtime.add_ready(0, "T3", "decode")
    runtime.add_ready(1, "T3", "decode")
    runtime.add_ready(2, "s3gen", "s3gen_chunk")

    for _ in range(16):
        worker.pipeline._pass()

    other_indices = [i for i, (node, _, _) in enumerate(executed) if node == "s3gen"]
    assert len(other_indices) >= 3, executed
    # After non-speculative s3gen runs, chain_ended() resets the cap: T3/decode
    # takes 1 non-speculative step + `cap=3` counted steps before s3gen runs again.
    gap = other_indices[1] - other_indices[0] - 1
    assert gap == 4, (executed, other_indices)


# ── Combined-walk _cap_key / note_launched tests ────────────────────────


def _combined_speculator(*, mixes: bool, cap: int = 2, peek: bool = False):
    excluded = []
    engine = SimpleNamespace(
        get_max_batch_tokens=lambda node, walk: 128 if mixes else None,
        get_step_order=lambda node, walk: StepOrder.ARRIVAL,
        splits_by_capture_key=lambda node, walk: False,
        get_max_batch_size=lambda node, walk: 8,
        batch_cap_varies=lambda node, walk: False,
        get_max_rows_in_combined=lambda node, walk: None,
    )
    scheduler = make_scheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        combined_walk_of={("LLM", "decode"): "mixed", ("LLM", "prefill"): "mixed"},
        speculation_mixes_walks={"LLM": mixes},
    )
    scheduler.has_ready_excluding = lambda key, among=None: False
    scheduler.next_batch = lambda exclude_target=None: excluded.append(exclude_target)
    speculator = make_worker(
        scheduler=scheduler, peek=peek, max_consecutive_spec=cap,
    ).speculator
    speculator.can_speculate = lambda batch: True
    speculator.lockstep.broadcast = lambda *a, **kw: -1
    return speculator, excluded


@pytest.mark.parametrize("batch_walk", ["mixed", "prefill"])
def test_combined_walk_gave_way_returning_mixed_or_prefill_counts_toward_cap(batch_walk):
    """A gave_way=True step stays on the chain's target even when take_chain_step
    returns a 'mixed' or 'prefill' batch; note_launched must keep counting toward cap."""
    speculator, excluded = _combined_speculator(mixes=True, cap=2)
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0])
    speculator._choose_spec_target = lambda batch: target
    gave_way_batch = SimpleNamespace(node_name="LLM", graph_walk=batch_walk, tp_seq=-1)
    gave_way_spec = SimpleNamespace(
        is_yield_away=True, gave_way=True,
        scheduled_batch=gave_way_batch, node_batch=SimpleNamespace(),
    )
    speculator._try_speculate_next = lambda pending, picked: gave_way_spec

    for expected_count in (1, 2):
        launched, yielded_from = speculator.build(_pending())
        assert launched is gave_way_spec and yielded_from is None
        speculator.note_launched(launched)
        assert speculator._cap_key == ("LLM", "mixed")
        assert speculator._since_other == expected_count

    _, yielded_from = speculator.build(_pending())
    assert yielded_from == ("LLM", "decode")
    assert excluded == [("LLM", "decode")]


def test_combined_walk_mixed_yield_away_counts_toward_cap():
    """On a mixing combined-walk chain, a yield-away batch labelled 'mixed'
    normalizes to the mixing scheduling key instead of resetting cap_steps."""
    speculator, excluded = _combined_speculator(mixes=True, cap=2)
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0])
    speculator._choose_spec_target = lambda batch: target
    speculator._try_speculate_next = lambda pending, picked: None
    mixed_batch = SimpleNamespace(node_name="LLM", graph_walk="mixed", tp_seq=-1)
    mixed_ya = SimpleNamespace(
        is_yield_away=True, gave_way=False,
        scheduled_batch=mixed_batch, node_batch=SimpleNamespace(),
    )
    speculator.scheduler.next_batch = lambda exclude_target=None: excluded.append(exclude_target) or mixed_batch
    speculator._yield_away_speculation = lambda picked: mixed_ya

    for expected_count in (1, 2):
        launched, yielded_from = speculator.build(_pending())
        assert launched is mixed_ya and yielded_from is None
        speculator.note_launched(launched)
        assert speculator._cap_key == ("LLM", "mixed")
        assert speculator._since_other == expected_count

    _, yielded_from = speculator.build(_pending())
    assert yielded_from == ("LLM", "decode")
    assert excluded == [None, None, ("LLM", "decode")]


def test_unmixed_combined_walk_resets_cap_between_constituent_walks():
    """With speculation_mixes=False, switching between 'decode' and 'prefill'
    on a combined walk still resets the cap."""
    speculator, excluded = _combined_speculator(mixes=False, cap=2)
    targets = [SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0])]
    speculator._choose_spec_target = lambda batch: targets[0]
    speculator._try_speculate_next = lambda pending, picked: _chain_step(picked.info.node_name)

    launched, yielded_from = speculator.build(_pending())
    assert yielded_from is None
    speculator.note_launched(launched)
    assert (speculator._cap_key, speculator._since_other) == (("LLM", "decode"), 1)

    targets[0] = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="prefill"), [0])
    launched, yielded_from = speculator.build(_pending())
    assert yielded_from is None
    speculator.note_launched(launched)
    assert (speculator._cap_key, speculator._since_other) == (("LLM", "prefill"), 1)

    launched, yielded_from = speculator.build(_pending())
    assert yielded_from is None and excluded == []


# ── Edge cases ──────────────────────────────────────────────────────────


def test_max_consecutive_spec_steps_zero_yields_on_first_step():
    """With cap=0 and peek=False, build() yields immediately on step 0."""
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0])
    speculator, _, excluded = _speculator(target, speculation=_chain_step(), peek=False)
    speculator._max_consecutive = 0

    launched, yielded_from = speculator.build(_pending())
    assert launched is None
    assert yielded_from == ("LLM", "decode")
    assert excluded == [("LLM", "decode")]


def test_cap_hit_alone_returns_none_and_chain_ended_resets_cap_key():
    """When T3/decode hits cap alone (no other node ready), next_batch(exclude_target)
    returns None, chain_ended() resets _cap_key and _since_other, and the
    non-speculative path keeps T3/decode running without stalling."""
    groups = {0: False, 1: True}

    def on_finish(runtime, step):
        for rid in step.batch.request_to_worker_graph:
            runtime.add_ready(rid, step.node_name, step.graph_walk)

    worker, runtime, executed = _build_multi_step_worker(
        None, groups=groups, order=StepOrder.ARRIVAL, peek=False, cap=2, on_finish=on_finish,
    )
    runtime.add_ready(0, "T3", "decode")
    runtime.add_ready(1, "T3", "decode")

    seen_resets = []
    orig_chain_ended = worker.speculator.chain_ended

    def spy_chain_ended():
        orig_chain_ended()
        seen_resets.append((worker.speculator._cap_key, worker.speculator._since_other))

    worker.speculator.chain_ended = spy_chain_ended

    for _ in range(10):
        worker.pipeline._pass()

    assert len(executed) == 10
    assert seen_resets and all(state == (None, 0) for state in seen_resets)


def test_admit_refused_skips_yield_away_and_chain_ended_resets_cap_key():
    """admit_refused skips yield-away and leaves _cap_key/_since_other untouched;
    chain_ended() resets both."""
    target = SpeculationTarget(SimpleNamespace(node_name="LLM", graph_walk="decode"), [0])
    speculator, _, excluded = _speculator(target, peek=False)
    speculator._max_consecutive = 2
    speculator.build(_pending())
    speculator.note_launched(SimpleNamespace(
        is_yield_away=False, gave_way=False,
        scheduled_batch=SimpleNamespace(node_name="LLM", graph_walk="decode"),
    ))
    assert (speculator._cap_key, speculator._since_other) == (("LLM", "decode"), 1)
    excluded.clear()

    refused_pending = SimpleNamespace(
        batch=object(), node_name="LLM", graph_walk="decode",
        node_batch=SimpleNamespace(admit_error="oom"), tp_seq=-1,
    )
    launched, yielded_from = speculator.build(refused_pending)
    assert (launched, yielded_from) == (None, None)
    assert excluded == []
    assert (speculator._cap_key, speculator._since_other) == (("LLM", "decode"), 1)

    speculator.chain_ended()
    assert (speculator._cap_key, speculator._since_other) == (None, 0)
    assert (speculator._consecutive, speculator._built_chain_key) == (0, None)
