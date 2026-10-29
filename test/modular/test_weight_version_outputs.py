"""Every output carries the weight version its request was admitted under.

Workers stamp the version they hold on each result frame (both runtimes; the
Rust one in ``test/rust/test_weight_version_outputs_rust.py``), and the conductor
puts the version it admitted the request under on ``request_complete`` and
``request_failed``. Until a reload names a version nothing changes: no field on
the wire, empty frame metadata, the same ``/generate`` body.

The last test runs 1,000 requests through the real ``Conductor.run`` with a
reload every 100, with the ranks and the client played by a script, and checks
that no request ever sees two versions.
"""
from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.data_worker import InlineToken, PreprocessWorker
from mstar.api_server.request_types import (
    APIServerMessage,
    RequestComplete,
    RequestFailed,
    ResultTokens,
)
from mstar.communication.wire import decode, encode

# -- wire ----------------------------------------------------------------------


def _complete(**kw):
    return APIServerMessage("request_complete", RequestComplete(
        request_id="r0", output_chunk_counts={"new_token": 5},
        conductor_ingest_time=1.5, conductor_finish_time=2.5, **kw))


def _failed(**kw):
    return APIServerMessage("request_failed", RequestFailed(
        request_id="r0", error_message="oom", status=503, **kw))


@pytest.mark.parametrize("make", [_complete, _failed], ids=["complete", "failed"])
def test_no_version_adds_nothing_to_the_wire(make):
    """A deployment that never reloads sends the bytes it sent before."""
    assert b"weight_version" not in encode(make())
    assert decode(encode(make(weight_version="v3"))).body.weight_version == "v3"


def test_a_result_tokens_frame_keeps_its_metadata():
    msg = APIServerMessage("result_tokens", ResultTokens(
        request_ids=["r1"], values=[7], loop_indices=[None], signal="new_token",
        modality="text", metadata={"weight_version": "v2"}))
    assert decode(encode(msg)).body.metadata == {"weight_version": "v2"}


# -- the Python runtime ----------------------------------------------------------


def _python_frames(version, inline):
    from test_delivery_chunk_counts import SHAPES, drive, python_runtime

    runtime, put, recorder = python_runtime(SHAPES["plain"][1])
    if version is not None:
        runtime.set_weight_version(version)
    drive(runtime, put, *SHAPES["plain"], inline=inline)
    chunks = recorder.chunks()
    # the inline case must really send RESULT_TOKENS frames, or it checks nothing new
    assert any(isinstance(c, ResultTokens) for c in chunks) == inline
    return chunks


@pytest.mark.parametrize("inline", [False, True], ids=["tensor", "inline"])
def test_the_python_runtime_stamps_nothing_until_a_version_is_set(inline):
    chunks = _python_frames(None, inline)
    assert chunks and [c.metadata for c in chunks] == [{}] * len(chunks)


@pytest.mark.parametrize("inline", [False, True], ids=["tensor", "inline"])
def test_the_python_runtime_stamps_every_frame_once_set(inline):
    chunks = _python_frames("v2", inline)
    assert chunks and [c.metadata for c in chunks] == [{"weight_version": "v2"}] * len(chunks)
    # the api server keeps each frame's metadata per tensor: no shared dict
    assert len({id(c.metadata) for c in chunks}) == len(chunks)


# -- the API server ------------------------------------------------------------


def test_an_inline_token_carries_its_frames_metadata_to_the_chunk():
    from test_inline_tokens import _thread

    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.result_tensor_input_queue = queue.Queue()
    facade.per_request_reading_tensors = {"r1": 0}
    facade.received_chunk_counts = {"r1": {}}
    frame = {"weight_version": "v4"}
    facade.new_result_token("r1", 11, None, "new_token", "text", metadata=frame)
    facade.new_result_token("r1", 12, None, "new_token", "text")
    items = [facade.result_tensor_input_queue.get_nowait() for _ in range(2)]
    assert [i.metadata for i in items] == [frame, {}]

    worker, stop = _thread()
    for item in items:
        worker.result_tensor_queue.put(item)
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        chunks = [worker.out_queue.get(timeout=5) for _ in range(2)]
    finally:
        stop.set()
        thread.join(timeout=5)
    assert [c.metadata for c in chunks] == [frame, {}]
    chunks[0].metadata["sample_rate"] = 1
    assert frame == {"weight_version": "v4"}, "a chunk shares the frame's dict"


def test_an_error_chunk_carries_the_version_of_the_output_it_replaces():
    """A postprocess failure, and a read that never delivers, answer with an
    error chunk; it keeps the failed output's version."""
    from test_inline_tokens import _thread

    from mstar.communication.tensors import FailedRead
    from mstar.graph.base import GraphEdge

    class _Broken:
        def postprocess(self, output, modality, request_kwargs=None):
            raise RuntimeError("bad bytes")

    worker, stop = _thread()
    worker.model = _Broken()
    worker.result_tensor_queue.put(InlineToken("r1", 5, None, "text", {"weight_version": "v6"}))
    worker.result_tensor_queue.put(InlineToken("r2", 5, None, "text"))
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        chunks = {c.request_id: c for c in (worker.out_queue.get(timeout=5) for _ in range(2))}
    finally:
        stop.set()
        thread.join(timeout=5)
    assert chunks["r1"].modality == "error"
    assert chunks["r1"].metadata == {"status": 500, "weight_version": "v6"}
    assert chunks["r2"].metadata == {"status": 500}, "no version before a reload"

    worker, _ = _thread()
    info = SimpleNamespace(uuid=31)
    worker.tensor_uuid_to_metadata_per_request["r3"] = {31: {"weight_version": "v6"}}
    edge = GraphEdge(next_node="emit_to_client", name="out")
    edge.tensor_info = [info]
    worker._fail_unread_outputs(FailedRead(request_id="r3", graph_edges=[edge], error=OSError("gone")))
    assert worker.out_queue.get_nowait().metadata == {"status": 500, "weight_version": "v6"}

    # an image or video encode that fails on the pool
    import collections
    import concurrent.futures

    failed = concurrent.futures.Future()
    failed.set_exception(ValueError("bad frame"))
    state = object()
    worker.request_output_state["r4"] = state
    worker.tensor_manager = SimpleNamespace(dereference=lambda uuid: None)
    worker._encoded = collections.deque([(failed, "r4", state, None, "image", {"weight_version": "v6"}, 41)])
    worker._release_encoded()
    assert worker.out_queue.get_nowait().metadata == {"status": 400, "weight_version": "v6"}


def test_the_message_loop_keeps_the_admitted_version():
    """``request_complete`` and ``request_failed`` hand their version to the
    waiting request, through the API server's real message loop."""
    import collections

    from test_delivery_chunk_counts import _ScriptedApiComm

    from mstar.api_server.entrypoint import APIServer, PendingRequest
    from mstar.profile.format import RequestProfile

    server = object.__new__(APIServer)
    server.running = True
    server._liveness_interval_s = 3600.0
    server._conductor_exited = lambda: None
    server.request_lock = threading.Lock()
    server.recently_completed = collections.OrderedDict()
    server._recently_completed_ttl = 3600.0
    server.log_stats = False
    server.metrics = None
    server.enable_nvtx = False
    from mstar.api_server.request_types import ResultChunk

    worker = PreprocessWorker.__new__(PreprocessWorker)
    worker.received_chunk_counts = {"ok": {}, "bad": {}, "live": {}}
    worker.per_request_reading_tensors = {"ok": 0, "bad": 0, "live": 0}
    worker.output_queue = queue.Queue()
    # a live call streams a versioned output, then its input fails: no output
    # names the error, so it takes the version the request's chunks carried,
    # also once the stream has handed those chunks out
    worker.output_queue.put(ResultChunk("live", "text", b"a", {"weight_version": "v6"}))
    worker.profile_queue = queue.Queue()
    worker.reads_done_queue = queue.Queue()
    worker._signal = lambda: None
    server.preprocess_worker = worker
    reqs = {
        rid: PendingRequest(streaming=False, input_modalities=["text"],
                            output_modalities=["text"], profile=RequestProfile(rid=rid))
        for rid in ("ok", "bad", "live")
    }
    server.pending_requests = dict(reqs)
    batches = [[
        APIServerMessage("request_complete", RequestComplete(
            request_id="ok", output_chunk_counts={}, conductor_ingest_time=0.0,
            conductor_finish_time=0.0, weight_version="v5")),
        APIServerMessage("request_failed", RequestFailed(
            request_id="bad", error_message="generation was paused", status=503,
            weight_version="v5")),
    ], []]
    class _Comm(_ScriptedApiComm):
        def get_all_new_messages(self):
            if len(self.released_before) == 1:
                # the stream handler takes the first pass's chunk
                assert [c.data for c in reqs["live"].chunks] == [b"a"]
                reqs["live"].chunks = []
                worker.output_queue.put(ResultChunk("live", "error", b"stream input failed", {"status": 400}))
            return super().get_all_new_messages()

    server.communicator = _Comm(server, batches, reqs["ok"])
    server._process_messages()
    assert reqs["ok"].weight_version == "v5" and reqs["bad"].weight_version == "v5"
    assert reqs["bad"].error_status == 503
    assert [c.metadata for c in reqs["live"].chunks] == [{"status": 400, "weight_version": "v6"}]


def test_a_request_with_no_answer_takes_the_version_its_chunks_carry():
    """No completion or failure ever came (the conductor died): the
    synthesized terminal error falls back to the chunks' version."""
    from mstar.api_server.entrypoint import PendingRequest, _request_weight_version
    from mstar.profile.format import RequestProfile

    req = PendingRequest(streaming=True, input_modalities=["text"], output_modalities=["text"],
                         profile=RequestProfile(rid="r1"))
    assert _request_weight_version(req) is None
    req.chunk_weight_version = "v6"  # as the message loop records it
    assert _request_weight_version(req) == "v6"
    req.weight_version = "v7"  # the conductor's answer wins
    assert _request_weight_version(req) == "v7"


def test_generate_adds_the_version_only_once_there_is_one():
    from mstar.api_server.entrypoint import _with_weight_version

    assert _with_weight_version({"request_id": "r", "outputs": {}}, None) == {
        "request_id": "r", "outputs": {}}
    assert _with_weight_version({"request_id": "r"}, "v1") == {"request_id": "r", "weight_version": "v1"}


# -- 1,000 requests through the real conductor, a reload every 100 ----------------


ROUNDS, PER_ROUND, IN_FLIGHT = 10, 100, 8
# passes a rank takes per step: longer than a pause and a reload take, so a
# pause that did not wait for the requests in flight would copy under them
STEP_PASSES = 12
PREPROCESS = "api_server_preprocess_worker"


class _ScriptDone(BaseException):
    """Ends ``Conductor.run``, whose loop logs and swallows an ``Exception``."""

    def __init__(self, reason="done"):
        super().__init__(reason)
        self.reason = reason


class _World:
    """The conductor's communicator, playing every peer.

    Each rank holds a weight version, changed only by an apply step, and stamps
    it on every output it reports, as a worker stamps its result frames. It
    refuses a control step while it holds a request, as ``Worker`` does. The
    client keeps ``IN_FLIGHT`` requests going, and once a round's last request
    is in, pauses (in flight ones still running), reloads under a new version
    and continues. Every third pause aborts instead of waiting.
    """

    def __init__(self, workers):
        from mstar.utils.ipc_format import SetupDone

        self.workers = list(workers)
        self.sent: list[tuple[str, object]] = []
        self.seen = 0
        self.outbox = [self._msg("SETUP_DONE", SetupDone(w)) for w in self.workers]
        self.later: list[tuple[int, str, str, dict | None]] = []  # (due pass, rid, walk, persist)
        self.rank_version = dict.fromkeys(self.workers)
        self.rank_epoch = dict.fromkeys(self.workers, 0)
        self.held = {w: set() for w in self.workers}
        self.stage: dict[str, str] = {}  # rid -> next report: "prefill" | "decode"
        self.stamps: dict[str, list] = {}  # rid -> version on each output
        self.admitted_under: dict[str, str | None] = {}  # rid -> committed when submitted
        self.finished: dict[str, tuple[str, object]] = {}  # rid -> (kind, body)
        self.committed: str | None = None
        self.late: list[str] = []  # submitted while not serving
        self.round = 0
        self.submitted = 0
        self.serving = True
        self.passes = 0
        self.uuid = 1000
        self.wg = {}

    # the communicator interface
    def send(self, entity, msg):
        self.sent.append((entity, msg))

    def get_all_new_messages(self):
        self.passes += 1
        if self.passes > 200_000:
            raise _ScriptDone("stalled")
        for entity, msg in self.sent[self.seen:]:
            self._observe(entity, msg)
        self.seen = len(self.sent)
        due = [item for item in self.later if item[0] <= self.passes]
        self.later = [item for item in self.later if item[0] > self.passes]
        for _due, rid, walk, persist in due:
            if self.stage.get(rid) == walk:  # not drained meanwhile
                self._report(rid, walk, persist)
        if self.round == ROUNDS and not self._in_flight() and self.serving:
            raise _ScriptDone
        self._client()
        out, self.outbox = self.outbox, []
        return out

    # helpers
    @staticmethod
    def _msg(kind, body):
        from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType

        return ConductorMessage(message_type=getattr(ConductorMessageType, kind), body=body)

    def _ptr(self, entity):
        from test_ingest_unpersist import _info

        self.uuid += 1
        return _info(self.uuid, entity=entity)

    def _in_flight(self):
        return [r for r in self.admitted_under if r not in self.finished]

    def _report(self, rid, walk, persist=None):
        from mstar.utils.ipc_format import WorkerGraphsDone

        worker = self.workers[0]
        # the version the rank holds when it runs the step, as a frame is stamped
        self.stamps.setdefault(rid, []).append(self.rank_version[worker])
        self.stage[rid] = "decode" if walk == "prefill" else "done"
        self.outbox.append(self._msg("WORKER_GRAPHS_DONE", WorkerGraphsDone(
            request_id=rid, worker_graph_ids=[self.wg[walk]], is_first_tp_rank=True,
            partition_name="default", persist_signals=persist or {},
            new_token_counts={"new_token": 1}, output_signal_names=["new_token"],
        )))

    def _reads_done(self, rid, entity):
        from mstar.utils.ipc_format import ReadsDone

        self.outbox.append(self._msg("READS_DONE", ReadsDone(request_id=rid, entity_id=entity)))

    def _ack(self, worker, op):
        from mstar.utils.ipc_format import WeightControlAck, WeightOp

        error = None
        if op.op != WeightOp.VERIFY.value and self.held[worker]:
            error = "not idle: holds " + ", ".join(sorted(self.held[worker]))
        elif op.op == WeightOp.APPLY.value:
            self.rank_version[worker] = op.weight_version
            self.rank_epoch[worker] = op.epoch
        self.outbox.append(self._msg("WEIGHT_CONTROL_ACK", WeightControlAck(
            epoch=op.epoch, worker_id=worker, op=op.op, error=error,
            weight_version=self.rank_version[worker], weight_epoch=self.rank_epoch[worker],
        )))

    # what the conductor sent
    def _observe(self, entity, msg):
        from mstar.utils.ipc_format import WorkerMessageType

        kind = msg.message_type
        body = msg.body
        if entity in self.workers:
            if kind == WorkerMessageType.NEW_REQUEST:
                self.held[entity].add(body.request_id)
                self.stage[body.request_id] = "prefill"
                self.later.append((self.passes + STEP_PASSES, body.request_id, "prefill",
                                   {"new_token": [self._ptr(entity)]}))
            elif kind == WorkerMessageType.INPUT_SIGNALS and self.stage.get(body.request_id) == "decode":
                self.later.append((self.passes + STEP_PASSES, body.request_id, "decode", None))
            elif kind == WorkerMessageType.DRAIN_REQUEST:
                self.stage.pop(body.request_id, None)
                self._reads_done(body.request_id, entity)
            elif kind == WorkerMessageType.REMOVE_REQUEST:
                self.held[entity].discard(body.request_id)
            elif kind == WorkerMessageType.WEIGHT_CONTROL:
                self._ack(entity, body)
        elif entity == PREPROCESS and kind == WorkerMessageType.DRAIN_REQUEST:
            self._reads_done(body.request_id, PREPROCESS)
        elif entity == "api_server":
            if kind in ("request_complete", "request_failed"):
                if body.request_id in self.finished:
                    raise _ScriptDone(f"{body.request_id} answered twice")
                self.finished[body.request_id] = (kind, body)
                if kind == "request_complete":
                    self._reads_done(body.request_id, PREPROCESS)
            elif kind == "generation_control_result":
                self._control_answer(body)

    def _control_answer(self, body):
        from mstar.utils.ipc_format import ContinueGeneration, UpdateWeightsFromDisk

        if body.status != 200:
            raise _ScriptDone(f"control step refused: {body}")
        if body.op_id.startswith("pause"):
            self.outbox.append(self._msg("UPDATE_WEIGHTS_FROM_DISK", UpdateWeightsFromDisk(
                op_id=f"update{self.round}", model_path="/ckpt", weight_version=f"v{self.round}")))
        elif body.op_id.startswith("update"):
            self.committed = body.weight_version
            self.outbox.append(self._msg("CONTINUE_GENERATION", ContinueGeneration(op_id=f"continue{self.round}")))
        elif body.op_id.startswith("continue"):
            self.serving = True

    def _submit(self, rid, first=False):
        from mstar.utils.ipc_format import NewRequestConductor

        # ``first``: ahead of this pass's acks, so it lands before the one
        # that resumes, while still paused
        self.outbox.insert(0 if first else len(self.outbox), self._msg("NEW_REQUEST", NewRequestConductor(
            request_id=rid, initial_signals={"text_inputs": [self._ptr(PREPROCESS)]},
            initial_input_modalities=["text"], initial_output_modalities=["text"],
            input_metadata={}, model_kwargs={},
        )))

    # the client
    def _client(self):
        from mstar.utils.ipc_format import PauseGeneration

        if not self.serving:
            # one arrives on every pass of the pause and the reload, so in
            # every protocol state: each is refused, never admitted
            rid = f"late{self.round}-{self.passes}"
            self.late.append(rid)
            self._submit(rid, first=True)
            return
        if self.round == ROUNDS:
            return
        while self.submitted < PER_ROUND and len(self._in_flight()) < IN_FLIGHT:
            rid = f"r{self.round}-{self.submitted}"
            self.submitted += 1
            self.admitted_under[rid] = self.committed
            self._submit(rid)
        if self.submitted == PER_ROUND:
            self.round += 1
            self.submitted = 0
            self.serving = False
            mode = "abort" if self.round % 3 == 0 else "wait"
            self.outbox.append(self._msg("PAUSE_GENERATION", PauseGeneration(op_id=f"pause{self.round}", mode=mode)))


def test_a_thousand_requests_with_a_reload_every_hundred_never_mix_versions(tmp_path, monkeypatch):
    from test_ingest_unpersist import _edge, _fwd, _ToyModel

    from mstar.conductor import conductor as conductor_mod
    from mstar.conductor.conductor import Conductor
    from mstar.graph.base import GraphEdge, GraphNode, Loop
    from mstar.graph.special_destinations import EMIT_TO_CLIENT

    walks = {
        "prefill": GraphNode(name="llm", input_names=["text_inputs"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True),
        ]),
        "decode": Loop(name="decode_loop", max_iters=8, outputs=[], section=GraphNode(
            name="llm", input_names=["text_inputs"], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
                GraphEdge(next_node="llm", name="text_inputs"),
            ])),
    }

    def initial(partition, signals):
        edge = _edge("llm", "text_inputs", signals["text_inputs"])
        return _fwd("prefill", [edge], unpersist=edge.tensor_info)

    def next_args(metadata, persist_signals):
        if metadata.graph_walk == "prefill":
            edge = _edge("llm", "text_inputs", persist_signals.get("new_token", []))
            return _fwd("decode", [edge], unpersist=edge.tensor_info)
        return _fwd("decode", done=True)

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"max_seq_len": 1024, "node_groups": [
        {"node_names": ["llm"], "ranks": [0]}]}))
    world: _World | None = None

    def make_communicator(**kw):
        nonlocal world
        world = _World([w for w in kw["push_ids"] if w.startswith("worker_")])
        return world

    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", make_communicator)
    monkeypatch.setattr(conductor_mod.time, "sleep", lambda s: None)
    c = Conductor(_ToyModel(walks, initial, next_args), str(path), socket_path_prefix=str(tmp_path / "s"))
    world.wg = {walk: c.node_walk_to_wg[("llm", walk)].worker_graph_id for walk in walks}
    c._poll_worker_liveness = lambda: None
    with pytest.raises(_ScriptDone) as done:
        c.run()
    assert done.value.reason == "done", (
        f"stalled in round {world.round}: serving={world.serving}, in flight {world._in_flight()[:4]}, "
        f"held {world.held}, stages {dict(list(world.stage.items())[:4])}"
    )

    served = world.admitted_under
    assert len(served) == ROUNDS * PER_ROUND
    completed = failed = 0
    for rid, version in served.items():
        kind, body = world.finished[rid]
        assert body.weight_version == version, (rid, body.weight_version, version)
        assert set(world.stamps.get(rid, [])) <= {version}, f"{rid} saw {world.stamps[rid]}"
        if kind == "request_complete":
            completed += 1
            assert world.stamps[rid] == [version, version]
        else:
            failed += 1
            assert (body.status, body.error_message) == (503, "generation was paused, aborting requests in flight")
    assert len(world.late) > 6 * ROUNDS, "too few arrivals during the pauses"
    for rid in world.late:
        kind, body = world.finished[rid]
        assert (kind, body.status, body.weight_version) == ("request_failed", 503, None), rid
        assert rid not in world.stamps, f"{rid} reached a rank"
    # the abort pauses had requests in flight, the wait pauses let theirs finish
    assert failed > 0 and completed > ROUNDS * PER_ROUND - failed - 1
    assert {v for v in served.values()} == {None, *(f"v{n}" for n in range(1, ROUNDS))}
    assert world.committed == f"v{ROUNDS}"
