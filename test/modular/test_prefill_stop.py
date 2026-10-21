"""A stop token sampled by the prefill ends the request there.

The prefill's sampled token is a request's first, and it comes before the
decode loop exists, so ``check_stop`` cannot stop that loop. It returns
``END_PARTITION`` instead: the worker reports the partition done on the
prefill's worker-graphs-done, and the conductor ends the partition rather
than start the decode walk the model chose. Without that, decode ran on past
the stop token.

Covers the worker (``Worker._postprocess_batch`` with stub routing), the
conductor (any report of the pass carries the flag), and Whisper's and
Qwen3-ASR's batched stop check on a prefill step, through the engine.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.conductor.conductor import Conductor, RequestData  # noqa: E402
from mstar.conductor.request_info import (  # noqa: E402
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    PartitionState,
)
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.model.base import ForwardPassArgs  # noqa: E402
from mstar.model.submodule_base import END_PARTITION, BatchedModelOutput, HostRows  # noqa: E402
from mstar.utils.ipc_format import WorkerGraphsDone  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.worker import PendingBatch, Worker  # noqa: E402

NODE, LOOP = "LLM", "decode_loop"


# --------------------------------------------------------------------------
# worker: END_PARTITION marks the partition done before the pass's report
# --------------------------------------------------------------------------


class _Sub:
    """Answers ``check_stop`` from a table: rid -> what it returns."""

    def __init__(self, stops):
        self.stops = stops

    def uniform_row_inputs(self, graph_walk, has_loopback=False):
        return None

    def prepare_inputs(self, **kw):
        return SimpleNamespace(input_seq_len=1)

    def can_batch(self, batch, model_inputs):
        return True

    def check_stop(self, rid, info, outputs):
        return set(self.stops.get(rid, ()))


class _Runtime:
    """Records what the worker hands the graph runtime, in order."""

    def __init__(self, events):
        self.events = events

    def pending_loop_stop_rids(self, graph_walk, loop_name):
        return set()

    def cleanup_consumed_inputs(self, node, rids, wgs):
        return [], []

    def clear_pending_loop_stops(self):
        pass

    def stop_loops_batched(self, partition, graph_walk, last_node_run, loop_names):
        for rid, names in loop_names:
            self.events.append(("stop_loops", rid, frozenset(names)))
        return [rid for rid, _ in loop_names]

    def update_dynamic_loop_iters(self, per_request_info, partition):
        pass

    def mark_stream_partition_done(self, rid, partition):
        self.events.append(("partition_done", rid, partition))


def _postprocess(stops, walk="prefill", rids=(1, 2, 3)):
    """Run ``Worker._postprocess_batch`` over one step whose stop check
    returns ``stops``; routing and sending are stubs. Returns the events."""
    events: list[tuple] = []
    eng = Engine.__new__(Engine)
    eng._submodules = {NODE: SimpleNamespace(submodule=_Sub(stops), resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    eng._runner = StepRunner({})
    batch = ExecutingBatch(
        node_name=NODE,
        per_request_info={
            r: SimpleNamespace(rid=r, update_publish_info=lambda p: None) for r in rids
        },
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False,
        ),
    )
    eng.prepare_inputs(batch)
    outputs = BatchedModelOutput(per_rid_outputs={
        r: {"new_token": [torch.tensor([7])]} for r in batch.request_ids
    })

    w = Worker.__new__(Worker)
    w._phase_period = 0
    w.enable_nvtx = False
    w.enable_prof = False
    w.device = torch.device("cpu")
    w._last_active = {}
    w._graph_runtime = _Runtime(events)
    w.tensor_manager = SimpleNamespace(
        cleanup_collectable=lambda *a: None, drop_host_copies=lambda uuids: None,
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda n: eng)
    w._innermost_loop_by_node = {}
    w._fail_requests = lambda failed: None
    w.request_state = SimpleNamespace(buffer_publish_info=lambda *a: None)
    w._store_and_route = lambda *a, **k: (
        SimpleNamespace(flat_uuids=[]),
        SimpleNamespace(freed_inputs=SimpleNamespace(uuids=[])),
        (),
    )
    w._register_outputs = lambda route_output: None
    # the pass's worker-graphs-done goes out from here
    w._stream_and_send = lambda *a, **k: events.append(("send",))

    pending = PendingBatch(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=walk,
            request_to_worker_graph=dict.fromkeys(rids, 0),
        ),
        node_batch=batch, node_name=NODE, partition="default", graph_walk=walk,
        future=Future(),
    )
    w._postprocess_batch(pending, outputs)
    return events


def test_a_prefill_stop_marks_its_partition_done_before_the_report():
    events = _postprocess({2: {END_PARTITION}})
    assert events == [("partition_done", 2, "default"), ("send",)]


def test_end_partition_never_reaches_the_runtime_as_a_loop_name():
    """A row that also stops a loop has that loop stopped as usual."""
    events = _postprocess({1: {LOOP}, 2: {END_PARTITION, LOOP}, 3: {END_PARTITION}}, walk="decode")
    stopped = {e[1]: e[2] for e in events if e[0] == "stop_loops"}
    assert stopped == {1: frozenset({LOOP}), 2: frozenset({LOOP})}
    assert sorted(e[1] for e in events if e[0] == "partition_done") == [2, 3]
    assert events[-1] == ("send",)


def test_a_step_without_partition_ends_marks_nothing():
    events = _postprocess({1: {LOOP}}, walk="decode")
    assert not [e for e in events if e[0] == "partition_done"]


# --------------------------------------------------------------------------
# conductor: the flag ends the partition instead of the next walk
# --------------------------------------------------------------------------


class _PrefillThenDecode:
    """Qwen3-ASR's state machine in miniature: prefill -> decode, with inputs."""

    def get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals,
                                        incoming_connections=None):
        partition_metadata.is_prefill = False
        partition_metadata.graph_walk = "decode"
        edge = GraphEdge(next_node=NODE, name="text_inputs")
        return ForwardPassArgs(full_metadata=partition_metadata, inputs=[edge], unpersist_tensors=[])


def _conductor():
    """One request whose prefill pass spans two worker graphs on two workers,
    the encoder's (0) and the LLM's (1), as a split config runs it."""
    c = object.__new__(Conductor)
    c.model = _PrefillThenDecode()
    c.enable_prof = False
    sent = []
    c._send_producer_done = lambda *a: None
    c._send_partition_inputs = lambda request_id, pname, fwd_args: sent.append(fwd_args.full_metadata.graph_walk)
    c._un_persist_tensors = lambda request_id, tensors: None
    c._set_partition_worker_graph_ids = lambda request_id, pname, walk: None
    pstate = PartitionState(
        partition_name="default",
        metadata=CurrentForwardConductorMetadata(graph_walk="prefill", is_prefill=True),
        current_worker_graph_ids={0, 1},
    )
    req = RequestData(
        persist_signals={}, persist_signal_ref_cnt={}, worker_graph_to_workers={0: ["enc"], 1: ["llm"]},
        all_worker_graph_ids={0, 1}, max_output_tokens=10_000, random_seed=0, resource_configs={},
        partition_states={"default": pstate},
        partition_definitions={"default": PartitionDefinition(name="default", graph_walks={"prefill", "decode"})},
    )
    c.requests = {"r": req}
    return c, pstate, sent


def _report(wg_id, partition_done=False):
    return WorkerGraphsDone(
        request_id="r", worker_graph_ids=[wg_id], is_first_tp_rank=True, partition_done=partition_done,
    )


@pytest.mark.parametrize("flag_on", ["first", "last"])
def test_a_flag_on_any_report_of_the_pass_ends_the_request(flag_on):
    """The run loop passes the completing report's own flag; the LLM's report
    can arrive first, and then the encoder's (flag-less) one completes the pass."""
    c, pstate, sent = _conductor()
    first, last = (_report(1, True), _report(0)) if flag_on == "first" else (_report(0), _report(1, True))
    assert c._process_worker_graphs_done(first) == []
    assert c._process_worker_graphs_done(last) == ["default"]
    all_done = c._process_done_forward("r", "default", partition_done_from_worker=last.partition_done)
    assert all_done is True and pstate.is_done
    assert sent == [], "the decode walk must not start"
    assert pstate.worker_reported_done is False  # per pass


def test_a_pass_without_the_flag_starts_decode():
    c, pstate, sent = _conductor()
    c._process_worker_graphs_done(_report(1))
    assert c._process_worker_graphs_done(_report(0)) == ["default"]
    assert c._process_done_forward("r", "default") is False
    assert sent == ["decode"] and not pstate.is_done


# --------------------------------------------------------------------------
# engine: a prefill step's batched stop check (``step_walk`` is the prefill)
# --------------------------------------------------------------------------


def _engine_check(sub, infos, tokens):
    """``Engine.check_stop_for_batch`` over a prefill step; the per-request
    path raises, so only the batched path can answer."""
    def _no_fallback(*a):
        raise AssertionError("fell back to the per-request stop check")

    sub.check_stop = _no_fallback
    engine = SimpleNamespace(_submodules={NODE: SimpleNamespace(submodule=sub)})
    rids = list(infos)
    failures = {}
    batch = SimpleNamespace(
        node_name=NODE, request_ids=rids, per_request_info_wrapped=infos,
        register_failure=failures.setdefault,
        step_context=SimpleNamespace(graph_walk=next(iter(infos.values())).graph_walk, request_walks=None),
    )
    per_rid = {rid: {"new_token": [tokens[i:i + 1]]} for i, rid in enumerate(rids)}
    stops = Engine.check_stop_for_batch(
        engine, batch, per_rid, host_rows=HostRows(tuple(rids), {"new_token": tokens}),
    )
    assert not failures
    return stops


def _info(walk, ignore_eos=False):
    return SimpleNamespace(
        graph_walk=walk, max_tokens=100, rid_handle=0, dynamic_loop_iter_counts={},
        resource_configs={"sampler": SimpleNamespace(ignore_eos=ignore_eos)},
    )


def test_whisper_batched_stop_check_runs_and_ends_a_first_token_eos():
    from mstar.model.whisper.config import DETECT_LANGUAGE_WALK, PREFILL_WALK
    from mstar.model.whisper.submodules import WhisperDecoderSubmodule

    sub = WhisperDecoderSubmodule.__new__(WhisperDecoderSubmodule)
    sub.config = SimpleNamespace(eos_token_id=50257, max_target_positions=448)
    infos = {"eos": _info(PREFILL_WALK), "word": _info(PREFILL_WALK)}
    stops = _engine_check(sub, infos, torch.tensor([50257, 1234]))
    assert stops == {"eos": {END_PARTITION}}
    # a step of a walk that never stops returns before reading its rows
    assert sub.check_stop_batched(["eos"], infos, HostRows(("eos",), {}), step_walk=DETECT_LANGUAGE_WALK) == {}


def test_qwen3_asr_batched_stop_check_runs_and_ends_a_first_token_stop():
    from mstar.model.qwen3_asr.config import PREFILL_WALK
    from mstar.model.qwen3_asr.submodules import Qwen3ASRLLMSubmodule

    sub = Qwen3ASRLLMSubmodule.__new__(Qwen3ASRLLMSubmodule)
    sub.config = SimpleNamespace(stop_token_ids={9, 10})
    infos = {"stop": _info(PREFILL_WALK), "word": _info(PREFILL_WALK), "ignored": _info(PREFILL_WALK, True)}
    stops = _engine_check(sub, infos, torch.tensor([10, 3, 9]))
    assert stops == {"stop": {END_PARTITION}}
