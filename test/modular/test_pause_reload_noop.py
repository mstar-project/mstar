"""Pause and weight reload cost nothing while they are not used.

Serving traffic must take main's path: the same messages, byte for byte, and no
call into the control code. Three checks:

- every per-request and per-step message encodes to the same bytes, and
  decodes back to itself, with the pause messages registered as without them;
- a request's whole life through the real ``Conductor.run`` sends the same
  messages with the pause state machine serving as with it absent, and never
  calls into the state machine;
- the real ``StepPipeline`` pass, fed per-request messages on its
  non-speculative path (the one with the control check), never runs a control
  step.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
import typing
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.request_types import (
    APIServerMessage,
    ConductorStatus,
    GenerationControlResult,
    RequestComplete,
    RequestFailed,
)
from mstar.communication import wire
from mstar.communication.wire import decode, encode
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.ipc_format import (
    AbortRequest,
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    EnableMetrics,
    FailRequests,
    InputSignals,
    MessageSource,
    MoveKind,
    NewRequest,
    NewRequestConductor,
    OffloadDelta,
    ReadsDone,
    RemoveRequest,
    ScheduleTPNode,
    SetupDone,
    StopLoops,
    TensorReceived,
    TPNoSpeculation,
    UnpersistTensors,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)

# ── wire bytes ──────────────────────────────────────────────────────────────


def _info(uuid: int) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[2, 8], dtype=torch.bfloat16, nbytes=32, address=0x1000, stride=(8, 1),
        uuid=uuid, source_session_id="host:1", source_entity="worker_0",
        source_tp_size=2, source_tp_rank=1, shm_segment="seg", shm_offset=64,
    )


def _fwd_info() -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="r0", fwd_index=3, random_seed=7, max_tokens=128, graph_walk="decode",
        partition_name="default", step_metadata={"is_prefill": False},
    )


def _edge() -> GraphEdge:
    return GraphEdge(next_node="llm", name="tok", tensor_info=[_info(11)], persist=True)


def _normal_messages() -> list[tuple[str, object]]:
    """Every message a request or a step sends, none of them a control one."""
    w, c = WorkerMessageType, ConductorMessageType
    delta = OffloadDelta.new()
    delta.add(MoveKind.EARMARK, "r1", "kv", 1, ["v"])
    return [
        ("worker.new_request", WorkerMessage(w.NEW_REQUEST, NewRequest(
            request_id="r0", partition_worker_graph_ids=[0],
            worker_graph_to_workers={0: ["worker_0", "worker_1"]},
            initial_inputs=[_edge()], request_info=_fwd_info(), loop_caps={}))),
        ("worker.input_signals", WorkerMessage(w.INPUT_SIGNALS, InputSignals(
            request_id="r0", inputs=[_edge()], request_info=_fwd_info(),
            producer_done={"talker"}, producer_done_items={"codes": 3}))),
        ("worker.remove", WorkerMessage(w.REMOVE_REQUEST, RemoveRequest(
            request_id="r0", source=MessageSource.TP_RANK_0, after_tp_seq=5))),
        ("worker.drain", WorkerMessage(w.DRAIN_REQUEST, DrainRequest(request_id="r0"))),
        ("worker.schedule_tp", WorkerMessage(w.SCHEDULE_TP, ScheduleTPNode(
            node_name="llm", graph_walk="decode", request_ids=["r0", "r1"], speculative=True,
            spec_seq=4, spec_from_seq=3, resident_delta=delta))),
        ("worker.tp_no_spec", WorkerMessage(w.TP_NO_SPEC, TPNoSpeculation(
            node_name="llm", graph_walk="decode", spec_from_seq=3))),
        ("worker.stop_loops", WorkerMessage(w.STOP_LOOPS, StopLoops(
            request_id="r0", loop_names={"decode_loop"}, partition_name="default",
            loop_stop_times={"decode_loop": NestedLoopIndices(
                loop_name_order=["decode_loop"], loop_indices={"decode_loop": 9}, wg_fwd_pass_idx=1)}))),
        ("worker.tensor_received", WorkerMessage(w.TENSOR_RECEIVED, TensorReceived(
            successful_tensors={11: 1}, failed_tensor_ids=[]))),
        ("worker.unpersist", WorkerMessage(w.UNPERSIST_TENSORS, UnpersistTensors(
            request_id="r0", uuid_to_ref_count={11: 2}))),
        ("worker.enable_metrics", WorkerMessage(w.ENABLE_METRICS, EnableMetrics())),
        ("conductor.new_request", ConductorMessage(c.NEW_REQUEST, NewRequestConductor(
            request_id="r0", initial_signals={"text_inputs": [_info(12)]},
            initial_input_modalities=["text"], initial_output_modalities=["audio"],
            input_metadata={"text": [{"len": 4}]}, model_kwargs={"temperature": 0.7},
            client_request_id="client-0"))),
        ("conductor.wgs_done", ConductorMessage(c.WORKER_GRAPHS_DONE, WorkerGraphsDone(
            request_id="r0", worker_graph_ids=[0], is_first_tp_rank=True,
            persist_signals={"new_token": [_info(13)]}, new_token_counts={"new_token": 1},
            output_signal_names=["new_token"], stream_items_sent={"codes": {"worker_1": 2}}))),
        ("conductor.fail", ConductorMessage(c.FAIL_REQUESTS, FailRequests(errors={"r0": "oom"}))),
        ("conductor.reads_done", ConductorMessage(c.READS_DONE, ReadsDone(
            request_id="r0", entity_id="worker_0"))),
        ("conductor.abort", ConductorMessage(c.ABORT_REQUEST, AbortRequest(request_id="r0"))),
        ("conductor.setup_done", ConductorMessage(c.SETUP_DONE, SetupDone(worker_id="worker_0"))),
        ("conductor.enable_metrics", ConductorMessage(c.ENABLE_METRICS, EnableMetrics())),
        ("api.request_complete", APIServerMessage("request_complete", RequestComplete(
            request_id="r0", output_chunk_counts={"new_token": 5},
            conductor_ingest_time=1.5, conductor_finish_time=2.5))),
        ("api.request_failed", APIServerMessage("request_failed", RequestFailed(
            request_id="r0", error_message="oom", status=503))),
        ("api.conductor_status", APIServerMessage("conductor_status", ConductorStatus(metrics=[]))),
        ("api.setup_done", APIServerMessage("setup_done")),
    ]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


# The tags pause and reload register. Only these, and the API server body union
# they widen, are this feature's; every other message's bytes belong to main.
PAUSE_TAGS = frozenset({
    "weight_control", "pause_generation", "continue_generation", "update_weights_from_disk",
    "get_generation_state", "weight_control_ack", "generation_control_result",
})


def _without_pause_types(monkeypatch) -> None:
    """The wire as it is without pause and reload: none of their tags, and the
    API server body union without their result. Plans are rebuilt from that."""
    assert PAUSE_TAGS <= wire._TAG_TO_TYPE.keys(), "a pause tag was renamed; update PAUSE_TAGS"
    branches = typing.get_args(APIServerMessage.__annotations__["body"])
    assert GenerationControlResult in branches
    monkeypatch.setattr(wire, "_TAG_TO_TYPE", {
        t: c for t, c in wire._TAG_TO_TYPE.items() if t not in PAUSE_TAGS})
    monkeypatch.setattr(wire, "_TYPE_TO_TAG", {
        c: t for c, t in wire._TYPE_TO_TAG.items() if t not in PAUSE_TAGS})
    monkeypatch.setitem(APIServerMessage.__annotations__, "body", typing.Union[
        tuple(b for b in branches if b is not GenerationControlResult)])
    monkeypatch.setattr(wire, "_PLANS", {})


def test_per_request_and_per_step_messages_encode_as_without_pause(monkeypatch):
    messages = _normal_messages()
    with_pause = {name: encode(msg) for name, msg in messages}
    read_with = {name: decode(data) for name, data in with_pause.items()}
    for name, msg in messages:
        assert type(read_with[name]) is type(msg)
        assert type(read_with[name].body) is type(msg.body), f"{name} decodes to another body"

    _without_pause_types(monkeypatch)
    assert {name: encode(msg) for name, msg in messages} == with_pause
    assert {name: decode(data) for name, data in with_pause.items()} == read_with


# ── a request's life through the real conductor ─────────────────────────────


PROMPT, TOKEN = 101, 202


def conductor_lifecycle(tmp_path, monkeypatch, generation: str) -> list[tuple[str, str]]:
    """Run two requests through the real ``Conductor.run``: one to completion and
    teardown, one aborted. ``generation`` is ``"serving"`` (the pause state
    machine present, untouched) or ``"absent"`` (``_generation = None``).
    Returns ``(peer, digest)`` per send, with the completion's wall-clock
    stamps zeroed."""
    from test_ingest_unpersist import (
        DECODE,
        LAST_PREFILL,
        _conductor,
        _decoder_walks,
        _msg,
        _new_request,
        _run,
        _ToyModel,
    )
    from test_ingest_unpersist import _info as _ptr

    from mstar.conductor.request_info import WalkStep

    plan = [WalkStep("prefill", step_metadata=LAST_PREFILL), DECODE]
    model = _ToyModel(_decoder_walks([("prefill", ())]), {"default": plan}, {"default": ["prefill", "decode"]})
    c = _conductor(model, [{"node_names": ["llm"], "ranks": [0]}], tmp_path, monkeypatch)
    if generation == "absent":
        c._generation = None
    touched: list[str] = []
    if generation == "serving":
        # recorded, not raised: ``Conductor.run`` logs and survives an exception
        for name in ("pause", "resume", "update", "report", "ack", "tick", "_start", "_answer"):
            setattr(c._generation, name, lambda *a, n=name, **k: touched.append(n))
    prefill = c.node_walk_to_wg[("llm", "prefill")].worker_graph_id
    decode = c.node_walk_to_wg[("llm", "decode")].worker_graph_id

    def done(rid, wg, persist=None):
        return _msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
            request_id=rid, worker_graph_ids=[wg], is_first_tp_rank=True,
            partition_name="default", persist_signals=persist or {},
            new_token_counts={"new_token": 1}, output_signal_names=["new_token"],
        ))

    def reads_done(rid, entity):
        return _msg(ConductorMessageType.READS_DONE, ReadsDone(request_id=rid, entity_id=entity))

    token = _ptr(TOKEN, entity="worker_0", node="llm", walk="prefill")
    _run(c, [
        [_new_request("r1", {"text_inputs": [_ptr(PROMPT)]})],
        [done("r1", prefill, {"new_token": [token]})],
        [done("r1", decode)],
        [reads_done("r1", "api_server_preprocess_worker")],
        [_new_request("r2", {"text_inputs": [_ptr(PROMPT + 1)]})],
        [_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest(request_id="r2"))],
        [reads_done("r2", e) for e in ("worker_0", "api_server_preprocess_worker")],
    ])
    assert touched == [], "serving traffic called into the pause state machine"
    out = []
    for peer, msg in c.comm.sent:
        body = getattr(msg, "body", None)
        if isinstance(body, RequestComplete):
            msg = dataclasses.replace(msg, body=dataclasses.replace(
                body, conductor_ingest_time=0.0, conductor_finish_time=0.0,
            ))
        out.append((peer, _digest(encode(msg))))
    return out


def test_a_request_while_serving_sends_what_main_sends(tmp_path, monkeypatch):
    """Same messages with the pause machine present as with it absent. A frozen
    digest list goes stale whenever main retouches a message or its fanout
    order, so this does not keep one."""
    (tmp_path / "serving").mkdir()
    (tmp_path / "absent").mkdir()
    serving = conductor_lifecycle(tmp_path / "serving", monkeypatch, "serving")
    absent = conductor_lifecycle(tmp_path / "absent", monkeypatch, "absent")
    peers = [peer for peer, _ in serving]
    assert "worker_0" in peers and "api_server" in peers, "the scenario sent nothing"
    assert serving == absent


# ── the worker loop ─────────────────────────────────────────────────────────


def serving_worker(inbox=()):
    """A worker of real components whose pipeline passes run the non-speculative
    path, with no step to schedule. Each pass reads the next entry of ``inbox``
    through the real ``InputIngest`` (per-request messages by default). Returns
    the worker; ``w.calls`` counts control-path and speculation calls."""
    from mstar.testing_utils.worker import make_worker
    from mstar.worker.weight_ops import WeightOps

    calls = {"idle_counts": 0, "dereference": 0, "speculate": 0}
    sent = []

    def idle_counts():
        calls["idle_counts"] += 1
        return (0, 0, 0)

    def dereference(*args):
        calls["dereference"] += 1

    per_request = [
        WorkerMessage(WorkerMessageType.TENSOR_RECEIVED, TensorReceived(
            successful_tensors={11: 1}, failed_tensor_ids=[])),
        WorkerMessage(WorkerMessageType.UNPERSIST_TENSORS, UnpersistTensors(
            request_id="r0", uuid_to_ref_count={11: 1})),
    ]
    passes = iter(inbox)
    w = make_worker(
        graph_runtime=SimpleNamespace(idle_counts=idle_counts),
        communicator=SimpleNamespace(
            send=lambda peer, msg: sent.append((peer, msg)),
            get_all_new_messages=lambda: next(passes, per_request),
            wait_for_work=lambda timeout: None,
        ),
        tensor_manager=SimpleNamespace(
            get_tensor=None, dereference_batch=dereference, get_ready_tensors=dict, take_failed_reads=list,
            increment_ref=lambda *a, **k: None, set_persist=lambda *a, **k: None,
        ),
        engine_manager=SimpleNamespace(engine=None),
        request_state=SimpleNamespace(per_request_info={}),
        streams=SimpleNamespace(poll=list, consumer_node={}),
        wakeup_event=SimpleNamespace(drain=lambda: None),
    )
    ops = WeightOps(w.pipeline._services, w.lifecycle, w.lockstep)
    w.weight_ops = w.pipeline.weight_ops = ops
    w.ingest._handlers[WorkerMessageType.WEIGHT_CONTROL] = ops.enqueue

    def build(pending):
        calls["speculate"] += 1
        return None, None

    w.speculator.build = build
    w.scheduler.next_batch = lambda *args, **kwargs: None
    w.calls, w.sent = calls, sent
    return w


def test_the_worker_loop_runs_no_control_step_while_serving():
    w = serving_worker()
    ran = []
    w.weight_ops.process = ran.append
    for _ in range(50):
        w.pipeline._pass()
    assert w.calls["dereference"] >= 50, "the loop did not take the scripted passes"
    assert w.calls["speculate"] == 50, "serving passes stopped speculating"
    assert ran == [] and w.calls["idle_counts"] == 0
    assert w.weight_ops.pending is None
    assert w.sent == [], "the loop sent a control answer"


def test_the_same_loop_runs_a_control_step_once_one_arrives():
    """The counter above can fire: a verify step in the third pass is run
    once, from the loop, and answered."""
    from mstar.utils.ipc_format import WeightControl

    control = WorkerMessage(WorkerMessageType.WEIGHT_CONTROL, WeightControl(op="verify", epoch=1))
    w = serving_worker(inbox=[[], [], [control]])
    for _ in range(10):
        w.pipeline._pass()
    acks = [msg.body for peer, msg in w.sent if msg.message_type == ConductorMessageType.WEIGHT_CONTROL_ACK]
    assert len(acks) == 1 and acks[0].epoch == 1 and acks[0].error is None
    assert w.calls["idle_counts"] == 1 and w.weight_ops.pending is None
