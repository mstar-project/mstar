"""Runtime inference errors must reach the client as a structured failure
instead of a blanket request timeout (issue #123).

What is left here after the resource-pool refactor is the part of that chain
that does not live in the engine: the worker turning a crash into a
FAIL_REQUESTS report (and never re-awaiting the future that raised), the api
server releasing the waiting client with an error, and the data worker turning
a postprocess blow-up into an error chunk.

The neighbouring pieces are covered elsewhere:

* engine-side attribution (admit refusals, per-rid stage failures) ->
  ``test_admit_failure_handling.py``
* excising a reported failure from the finished batch ->
  ``test_failed_request_reporting.py``
* the scheduler refusing to schedule a failed rid ->
  ``test_micro_scheduler.py``
* the conductor's teardown drain barrier -> ``test_teardown_barrier.py``
"""

import asyncio
import collections
import itertools
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mstar.api_server.entrypoint import APIServer, PendingRequest
from mstar.api_server.request_types import (
    APIServerMessage,
    RequestComplete,
    RequestFailed,
    ResultChunk,
    ResultTensors,
    ResultTokens,
)
from mstar.communication.tensors import FailedRead
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.profile.format import ReadQuarantine, RequestProfile, RequestTiming
from mstar.utils.ipc_format import ConductorMessageType, FailRequests
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import PendingBatch, Worker

# ── worker ─────────────────────────────────────────────────────────────────


class _Runtime:
    """Interning plus the speculative flag, which the runtime owns now."""

    def __init__(self, known_rids):
        self._known = set(known_rids)
        self.cleared: list[tuple[str, list]] = []

    def get_rid_handle(self, r):
        return r if r in self._known else None

    def get_rid_string(self, h):
        return h

    def set_in_flight(self, node, wg_id, rids, value):
        del wg_id
        if not value:
            self.cleared.append((node, list(rids)))


def _worker(known_rids=("r1", "r2")):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.sent = []
    w.communicator = SimpleNamespace(
        send=lambda entity_id, msg: w.sent.append((entity_id, msg))
    )
    w.request_state = SimpleNamespace(
        per_request_info={rid: object() for rid in known_rids}
    )
    # Identity interning: the rid string doubles as its own handle here.
    w._graph_runtime = _Runtime(known_rids)
    w.scheduler = MicroScheduler.__new__(MicroScheduler)
    w.scheduler.failed_rids = set()
    w.scheduler.held_until = {}
    w.scheduler.admit_errors = {}
    w.scheduler.backlog = {}
    w._reported_quarantine_reads = 0
    return w


def test_fail_requests_reports_per_rid_errors_once():
    w = _worker()
    w._fail_requests({"r1": "boom", "r2": "bang"})
    assert len(w.sent) == 1
    entity, msg = w.sent[0]
    assert entity == "conductor"
    assert msg.message_type == ConductorMessageType.FAIL_REQUESTS
    assert msg.body == FailRequests(errors={"r1": "boom", "r2": "bang"})
    assert w.scheduler.failed_rids == {"r1", "r2"}


def test_fail_requests_ignores_rids_the_worker_already_dropped():
    """The conductor answers a failure by tearing the request down. It won't
    start one for a request it no longer tracks, so reporting an unknown rid
    would pin it in failed_rids forever."""
    w = _worker(known_rids=("r1",))
    w._fail_requests({"gone": "boom"})
    assert w.sent == []
    assert w.scheduler.failed_rids == set()


def _pending_batch(rids, future=None):
    return PendingBatch(
        batch=ScheduledBatch(
            node_name="node", graph_walk="walk",
            request_to_worker_graph={rid: "wg" for rid in rids},
        ),
        # _handle_main_loop_error works off the ScheduledBatch alone; the
        # ExecutingBatch is never touched on this path.
        node_batch=None, node_name="node",
        partition="default", graph_walk="walk", future=future,
    )


def test_crashed_forward_fails_the_whole_batch():
    """A raise the worker can't attribute to one rid fails every request the
    iteration touched, rather than dropping them into the request timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        w = _worker()
        w._in_flight_rids = {"r1"}
        pending = _pending_batch(["r1"], future=executor.submit(lambda: 1 / 0))
        try:
            pending.future.result()
        except ZeroDivisionError as exc:
            w._handle_main_loop_error(exc, (pending, None), None)

        assert len(w.sent) == 1
        errors = w.sent[0][1].body.errors
        assert set(errors) == {"r1"}
        assert "ZeroDivisionError" in errors["r1"]
        # The node must not stay flagged as speculatively scheduled, or it can
        # never be re-queued. The runtime owns that flag now.
        assert w._graph_runtime.cleared == [("node", ["r1"])]
    finally:
        executor.shutdown(wait=True)


def test_error_handler_drains_the_speculative_batch_still_on_the_gpu():
    """The spec batch's future was submitted before the crash; abandoning it
    would leave the GPU thread writing into a batch nobody collects."""
    executor = ThreadPoolExecutor(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def _slow():
        started.set()
        release.wait(10)
        return "done"

    try:
        w = _worker()
        w._in_flight_rids = set()
        spec = _pending_batch(["r2"], future=executor.submit(_slow))
        started.wait(10)
        release.set()
        w._handle_main_loop_error(RuntimeError("boom"), (None, spec), None)

        assert spec.future.done()
        assert set(w.sent[0][1].body.errors) == {"r2"}
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_error_handler_also_fails_the_batch_built_this_iteration():
    w = _worker()
    w._in_flight_rids = {"r1"}
    scheduled = ScheduledBatch(
        node_name="node", graph_walk="walk",
        request_to_worker_graph={"r2": "wg"},
    )
    w._handle_main_loop_error(RuntimeError("boom"), (None, None), scheduled)
    assert set(w.sent[0][1].body.errors) == {"r1", "r2"}


def test_failed_input_read_fails_only_its_request_once():
    w = _worker()
    w.enable_nvtx = False
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    ingested = []
    w._graph_runtime.ingest_inputs_batch = lambda block, can_buffer: ingested.append(block)
    w._one_rids_edges = lambda rid, edges: [(rid, edges)]
    healthy = SimpleNamespace(is_streaming=False)
    failed = [FailedRead("r1", [], RuntimeError("Mooncake read failed. Status: -1"))]
    w.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {"r2": [healthy]},
        take_failed_reads=lambda: list(failed),
        quarantined_reads=0,
    )

    w._check_ready_tensors()
    w._check_ready_tensors()

    (sent,) = w.sent
    assert sent[1].message_type == ConductorMessageType.FAIL_REQUESTS
    assert list(sent[1].body.errors) == ["r1"]
    assert "Mooncake read failed" in sent[1].body.errors["r1"]
    assert ingested == [[("r2", [healthy])]] * 2


def test_worker_reports_its_read_quarantine_to_the_api_server_when_it_grows():
    """Kept buffers live in the worker, /health in the API server: the worker
    sends its totals each time they grow, and only then."""
    w = _worker()
    w.enable_nvtx = False
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    tm = SimpleNamespace(
        get_ready_tensors=lambda: {},
        take_failed_reads=lambda: [],
        quarantined_reads=0,
    )
    tm.read_quarantine = lambda: ReadQuarantine("w0", tm.quarantined_reads, 64 * tm.quarantined_reads, 100)
    w.tensor_manager = tm

    w._check_ready_tensors()
    assert w.sent == []
    # a teardown quarantined two reads between polls
    tm.quarantined_reads = 2
    w._check_ready_tensors()
    w._check_ready_tensors()

    (sent,) = w.sent
    assert sent[0] == "api_server"
    assert sent[1].message_type == "read_quarantine"
    assert sent[1].body == ReadQuarantine("w0", 2, 128, 100)
    assert sent[1].body.over_budget


# ── api server ─────────────────────────────────────────────────────────────


def _api_server(messages):
    s = APIServer.__new__(APIServer)
    s.pending_requests = {}
    s.recently_completed = {}
    s._recently_completed_ttl = 15.0
    s.request_lock = threading.Lock()
    s.running = True
    s.conductor_proc = None
    s._liveness_interval_s = 0.5
    s.log_stats = False
    s.cleaned = []
    s.communicator = SimpleNamespace(
        get_all_new_messages=lambda: messages.pop(0) if messages else []
    )
    s.preprocess_worker = SimpleNamespace(
        get_profile_updates=lambda: [],
        get_result_chunks=lambda: [],
        has_pending_tensors=lambda rid: False,
        received_final_chunks=lambda rid, outs: False,
        # The teardown drain: the api server reports it is done reading and the
        # conductor drives the hard cleanup.
        finished_reading=lambda rid, drained=True: s.cleaned.append(rid),
        new_result_tensors=lambda body: None,
        discard_result_tensors=lambda body: None,
    )
    return s


def _pending_request(streaming=False):
    return PendingRequest(
        streaming=streaming,
        input_modalities=["text"],
        output_modalities=["text"],
        profile=RequestProfile(rid="r1", timing=RequestTiming(recv_time=0.0)),
    )


def _drain(server):
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    try:
        yield_deadline = time.time() + 10
        while time.time() < yield_deadline:
            if server.pending_requests["r1"].event.is_set():
                return
            time.sleep(0.005)
        raise AssertionError("request was never released")
    finally:
        server.running = False
        thread.join(timeout=10)


def test_engine_failure_releases_the_client_with_an_error():
    server = _api_server([[APIServerMessage(
        message_type="request_failed",
        body=RequestFailed(request_id="r1", error_message="boom", status=500),
    )]])
    server.pending_requests["r1"] = _pending_request()
    _drain(server)

    req = server.pending_requests["r1"]
    assert req.error == "boom"
    assert req.error_status == 500
    # Parked for cleanup so the data worker's per-request state is reclaimed.
    assert "r1" in server.recently_completed


def test_engine_failure_does_not_clobber_an_earlier_error():
    server = _api_server([[APIServerMessage(
        message_type="request_failed",
        body=RequestFailed(request_id="r1", error_message="downstream", status=500),
    )]])
    req = _pending_request()
    req.error = "bad input"
    req.error_status = 400
    server.pending_requests["r1"] = req
    _drain(server)
    assert req.error == "bad input" and req.error_status == 400


_READ_ERROR = b"output tensor read failed: RuntimeError: Mooncake read failed. Status: -1"


def _finished_stream(chunks, error=None):
    """A server holding r1 as the message loop leaves it once r1 is done."""
    server = _api_server([])
    server.timeout_seconds = 5.0
    server.enable_nvtx = False
    req = _pending_request(streaming=True)
    req.chunks = list(chunks)
    req.error = error
    req.event.set()
    server.pending_requests["r1"] = req
    return server


def _collect(agen):
    async def run():
        return [item async for item in agen]
    return asyncio.run(run())


def _chunk(modality, data):
    return ResultChunk(request_id="r1", modality=modality, data=data, metadata={"status": 500})


def test_a_stream_ends_at_its_first_error():
    """The first error chunk is the response's last: a later output or second
    error the data worker emitted before it stopped is not delivered, and the
    iterator does not add its own copy of the error after it."""
    server = _finished_stream(
        [_chunk("text", b"a"), _chunk("error", _READ_ERROR), _chunk("text", b"late"),
         _chunk("error", _READ_ERROR)],
        error=_READ_ERROR.decode(),
    )
    got = _collect(server.iter_result_chunks("r1"))
    assert [(c.modality, c.data) for c in got] == [("text", b"a"), ("error", _READ_ERROR)]
    assert "r1" not in server.pending_requests

    server = _finished_stream([_chunk("error", _READ_ERROR)], error=_READ_ERROR.decode())
    assert [c.modality for c in _collect(server.iter_result_chunks("r1"))] == ["error"]


def test_a_batch_is_cut_at_its_first_error():
    """The chat stream reads whole batches: the first error is its batch's
    last chunk and the response's last, and the final flush adds no copy."""
    server = _finished_stream(
        [_chunk("text", b"a"), _chunk("error", _READ_ERROR), _chunk("text", b"late"),
         _chunk("error", _READ_ERROR)],
        error=_READ_ERROR.decode(),
    )
    batches = _collect(server.iter_result_chunk_batches("r1"))
    assert [[(c.modality, c.data) for c in b] for b in batches] == [
        [("text", b"a"), ("error", _READ_ERROR)],
    ]


def test_an_engine_failure_still_ends_the_stream_with_one_error():
    """``request_failed`` sets the error without a chunk; the iterator still
    reports it in-band, once."""
    server = _finished_stream([_chunk("text", b"a")], error="boom")
    got = _collect(server.iter_result_chunks("r1"))
    assert [(c.modality, c.data) for c in got] == [("text", b"a"), ("error", b"boom")]


def test_chat_stream_carries_one_error_event_and_nothing_after_it():
    """The order seen with injected read failures was content, error, error,
    done. The stream now ends content, error, done."""
    from mstar.api_server.openai.serving_chat import _stream

    server = _finished_stream(
        [_chunk("text", b"a"), _chunk("error", _READ_ERROR), _chunk("text", b"late"),
         _chunk("error", _READ_ERROR)],
        error=_READ_ERROR.decode(),
    )
    kinds = []
    for event in _collect(_stream(server, "m", "r1", 24000)):
        payload = event.removeprefix("data: ").strip()
        if payload == "[DONE]":
            kinds.append("done")
            continue
        body = json.loads(payload)
        if "error" in body:
            kinds.append("error")
        elif body["choices"][0]["delta"].get("content"):
            kinds.append("content")
        else:
            kinds.append("role")
    assert kinds == ["role", "content", "error", "done"]


# ── data worker ────────────────────────────────────────────────────────────


def _preprocess_thread(model):
    from mstar.api_server.data_worker import PreprocessWorkerThread, RequestOutputState

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue = queue.Queue()
    wt.model = model
    wt.request_model_kwargs = {}
    wt.tensor_uuid_to_metadata_per_request = {"r1": {"u1": {}}}
    wt.enable_prof = False
    wt.enable_nvtx = False
    wt.request_output_state = {
        "r1": RequestOutputState(order={"u1": (0, None)}, next_sequence=1),
    }
    # not in in_flight_requests, so a failure here sends no abort; the abort
    # tests below opt in with _tracked
    wt.in_flight_requests = set()
    wt._draining_rids = set()
    wt._reads_done_sent = set()
    wt._failed_rids = set()
    wt._pre_admit_cancelled = set()
    wt._pending_ids = set()
    wt._admit_lock = threading.Lock()
    wt.output_loop_idxs = {}
    wt.per_request_reading_tensors = {}
    return wt


def _tracked(wt, *rids, inflight_reads=False):
    """Make ``rids`` requests the thread is serving, and record what it sends."""
    wt.in_flight_requests.update(rids)
    wt.sent = []
    wt.communicator = SimpleNamespace(send=lambda entity, msg: wt.sent.append((entity, msg)))
    wt.tensor_manager.has_inflight_reads = lambda rid: inflight_reads
    wt.tensor_manager.force_cleanup_request = lambda rid: None
    return wt


def _sent(wt, message_type):
    return [m.body.request_id for e, m in wt.sent if e == "conductor" and m.message_type == message_type]


def _read_failure(rid, *uuids):
    edge = SimpleNamespace(name="text_output", tensor_info=[SimpleNamespace(uuid=u) for u in uuids])
    return FailedRead(rid, [edge], RuntimeError("Mooncake read failed. Status: -1"))


def test_output_postprocess_failure_becomes_an_error_chunk():
    """A model that blows up postprocessing a result tensor must fail that
    request, not silently drop the chunk and leave the client hanging."""

    class _BadModel:
        def postprocess(self, tensor, modality, request_kwargs=None):
            raise RuntimeError("decode failed")

    wt = _preprocess_thread(_BadModel())
    dereferenced = []
    edge = SimpleNamespace(
        name="text_output", tensor_info=[SimpleNamespace(uuid="u1")],
    )
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {"r1": [edge]},
        take_failed_reads=lambda: [],
        get_tensor=lambda uuid: object(),
        dereference=dereferenced.append,
    )

    assert wt._process_read_tensors() is True
    chunk: ResultChunk = wt.out_queue.get_nowait()
    assert chunk.request_id == "r1"
    assert chunk.modality == "error"
    assert b"decode failed" in chunk.data
    assert chunk.metadata["status"] == 500
    # The tensor is still released even though postprocessing died.
    assert dereferenced == ["u1"]


def test_a_failed_output_drops_the_outputs_held_behind_it():
    """Sequence 1 finished first and waits in the reorder buffer for 0. When 0
    fails, the error goes out and 1 is dropped, not released after the error,
    and nothing is left held for the request."""

    class _BadModel:
        def postprocess(self, tensor, modality, request_kwargs=None):
            raise RuntimeError("decode failed")

    wt = _preprocess_thread(_BadModel())
    held = ResultChunk(request_id="r1", modality="text", data=b"later", metadata={})
    wt.request_output_state["r1"].pending[1] = held
    wt.request_output_state["r1"].next_sequence = 2
    edge = SimpleNamespace(
        name="text_output", tensor_info=[SimpleNamespace(uuid="u1")],
    )
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {"r1": [edge]},
        take_failed_reads=lambda: [],
        get_tensor=lambda uuid: object(),
        dereference=lambda uuid: None,
    )

    assert wt._process_read_tensors() is True
    assert wt.out_queue.get_nowait().modality == "error"
    assert wt.out_queue.empty()
    assert wt.request_output_state["r1"].pending == {}


def test_result_transfer_failure_answers_for_every_queued_tensor():
    """The API server decrements its outstanding-read count once per chunk, so
    a read that never starts owes one error chunk per tensor it dropped."""
    wt = _preprocess_thread(model=None)
    edge = SimpleNamespace(
        name="text_output",
        tensor_info=[SimpleNamespace(uuid="u1"), SimpleNamespace(uuid="u2")],
    )
    result = ResultTensors(
        request_id="r1", modality="text", graph_edge=edge, loop_indices=None,
    )
    try:
        raise RuntimeError("arena full")
    except RuntimeError as exc:
        wt._fail_request(
            "r1", exc, "text output transfer",
            count=len(result.graph_edge.tensor_info),
        )
    assert wt.out_queue.qsize() == 2
    assert b"arena full" in wt.out_queue.get_nowait().data


def test_failed_output_read_becomes_an_error_chunk_and_drops_held_outputs():
    """The read of sequence 0 failed while 1 waits in the reorder buffer. The
    error goes out and 1 is dropped with the rest of the request."""
    wt = _preprocess_thread(model=None)
    held = ResultChunk(request_id="r1", modality="text", data=b"later", metadata={})
    wt.request_output_state["r1"].pending[1] = held
    wt.request_output_state["r1"].next_sequence = 2
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {},
        take_failed_reads=lambda: [_read_failure("r1", "u1")],
    )

    assert wt._process_read_tensors() is True
    error = wt.out_queue.get_nowait()
    assert error.modality == "error" and error.metadata["status"] == 500
    assert b"output tensor read failed: RuntimeError: Mooncake read failed" in error.data
    assert wt.out_queue.empty()
    assert wt.request_output_state["r1"].pending == {}
    assert wt.request_output_state["r1"].order == {}
    assert wt.tensor_uuid_to_metadata_per_request["r1"] == {}


def test_a_failed_read_behind_an_unfinished_output_is_reported_at_once():
    """Output 0 is still being read when the read of output 1 fails. The client
    hears about the failure now, not after output 0 arrives; output 0 arriving
    later is released, not sent."""
    wt = _preprocess_thread(model=None)
    state = wt.request_output_state["r1"]
    state.order = {"u0": (0, None), "u1": (1, None)}
    state.next_sequence = 2
    ready, failed, dereferenced = {}, [_read_failure("r1", "u1")], []
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: ready,
        take_failed_reads=lambda: failed,
        dereference=dereferenced.append,
    )

    assert wt._process_read_tensors() is True
    error = wt.out_queue.get_nowait()
    assert error.modality == "error" and wt.out_queue.empty()

    failed.clear()
    ready["r1"] = [SimpleNamespace(name="text_output", tensor_info=[SimpleNamespace(uuid="u0")])]
    assert wt._process_read_tensors() is True
    assert wt.out_queue.empty()
    assert dereferenced == ["u0"]
    assert state.pending == {} and state.order == {}


def test_no_output_of_a_failed_request_goes_out_after_its_error():
    """One poll returns a failed read for r1 and finished reads for r1 and r2.
    r1 gets only its error; its finished output is released without being
    postprocessed or sent. r2 is unaffected."""

    class _Model:
        def __init__(self):
            self.postprocessed = []

        def postprocess(self, tensor, modality, request_kwargs=None):
            self.postprocessed.append(tensor)
            return tensor

    from mstar.api_server.data_worker import RequestOutputState

    model = _Model()
    wt = _preprocess_thread(model)
    wt.tensor_uuid_to_metadata_per_request = {"r1": {"u1": {}, "u2": {}}, "r2": {"v1": {}}}
    wt.request_output_state["r1"].order["u2"] = (1, None)
    wt.request_output_state["r1"].next_sequence = 2
    wt.request_output_state["r2"] = RequestOutputState(order={"v1": (0, None)}, next_sequence=1)
    dereferenced = []
    edge = lambda uuid: SimpleNamespace(name="text_output", tensor_info=[SimpleNamespace(uuid=uuid)])  # noqa: E731
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {"r1": [edge("u2")], "r2": [edge("v1")]},
        take_failed_reads=lambda: [_read_failure("r1", "u1")],
        get_tensor=lambda uuid: f"tensor-{uuid}".encode(),
        dereference=dereferenced.append,
    )

    assert wt._process_read_tensors() is True
    chunks = []
    while not wt.out_queue.empty():
        chunks.append(wt.out_queue.get_nowait())
    assert [(c.request_id, c.modality) for c in chunks] == [("r1", "error"), ("r2", "text")]
    assert model.postprocessed == [b"tensor-v1"]
    assert sorted(dereferenced) == ["u2", "v1"]


def test_a_failed_request_reports_one_error_with_one_traceback(caplog):
    """A read of two tensors fails, then a later read of the same request is
    dropped by the manager with the same error, then a later output fails to
    postprocess. The client gets one error chunk per tensor of the first
    failure (the API server counts outstanding tensors per chunk) and the log
    one traceback; everything after is debug only."""
    wt = _preprocess_thread(model=None)
    failed = [_read_failure("r1", "u1", "u2")]
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {},
        take_failed_reads=lambda: failed,
    )

    with caplog.at_level("DEBUG", logger="mstar.api_server.data_worker"):
        wt._process_read_tensors()
        failed[:] = [_read_failure("r1", "u3")]
        wt._process_read_tensors()
        wt._fail_request("r1", RuntimeError("decode failed"), "text output postprocessing")

    chunks = [wt.out_queue.get_nowait() for _ in range(wt.out_queue.qsize())]
    assert [c.modality for c in chunks] == ["error", "error"]
    assert all(b"Mooncake read failed" in c.data for c in chunks)
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and errors[0].exc_info is not None
    assert len([r for r in caplog.records if "which already failed" in r.getMessage()]) == 2


def test_the_first_failed_output_read_aborts_the_request_once():
    """A failed output read releases the client with an error; the request
    itself must stop too, or its walk keeps producing outputs that are read
    and dropped. The thread aborts it once and starts its own drain, so the
    outputs still on their way are acked unread instead of read."""
    wt = _preprocess_thread(model=None)
    failed = [_read_failure("r1", "u1")]
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {},
        take_failed_reads=lambda: failed,
    )
    _tracked(wt, "r1")

    wt._process_read_tensors()
    failed[:] = [_read_failure("r1", "u2")]
    wt._process_read_tensors()
    wt._fail_request("r1", RuntimeError("decode failed"), "text output postprocessing")

    from mstar.utils.ipc_format import ConductorMessageType as T
    assert _sent(wt, T.ABORT_REQUEST) == ["r1"]
    # no read in flight, so the drain acks at once
    assert _sent(wt, T.READS_DONE) == ["r1"]
    assert "r1" in wt._draining_rids


def test_a_request_already_being_torn_down_is_not_aborted_again():
    from mstar.utils.ipc_format import ConductorMessageType as T

    for teardown in ("_draining_rids", "_reads_done_sent"):
        wt = _preprocess_thread(model=None)
        wt.tensor_manager = SimpleNamespace(
            get_ready_tensors=lambda: {}, take_failed_reads=lambda: [_read_failure("r1", "u1")],
        )
        _tracked(wt, "r1")
        getattr(wt, teardown).add("r1")
        wt._process_read_tensors()
        assert wt.out_queue.qsize() == 1
        assert _sent(wt, T.ABORT_REQUEST) == []

    # torn down already, or never reached the conductor
    wt = _preprocess_thread(model=None)
    wt.tensor_manager = SimpleNamespace()
    _tracked(wt)
    wt._fail_request("r1", RuntimeError("gone"), "text output transfer")
    wt._fail_request("r2", ValueError("bad knob"), "preprocessing", abort=False)
    assert _sent(wt, T.ABORT_REQUEST) == []


def _api_notice_orders():
    """Every order of the conductor's REMOVE_REQUEST and any of the API
    server's later notices for the rid: a finish (drained or not, from the
    completion prune) and a client abort, alone or both."""
    orders = []
    for notices in ([], ["finish"], ["drained finish"], ["abort"],
                    ["finish", "abort"], ["drained finish", "abort"]):
        for order in itertools.permutations(["remove", *notices]):
            if order not in orders:
                orders.append(order)
    return orders


@pytest.mark.parametrize("order", _api_notice_orders(), ids=" > ".join)
def test_a_self_aborted_request_ignores_the_api_servers_later_notices(order):
    """The API server parks a failed rid; its completion prune can then report
    it finished and its client still abort it, before or after the
    conductor's REMOVE_REQUEST. The thread's own abort already covers each:
    whatever the order, one ABORT_REQUEST and one READS_DONE go out and no
    drain entry outlives the teardown."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    wt = _preprocess_thread(model=None)
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {}, take_failed_reads=lambda: [_read_failure("r1", "u1")],
    )
    _tracked(wt, "r1", inflight_reads=True)
    wt._process_read_tensors()
    assert _sent(wt, T.ABORT_REQUEST) == ["r1"] and _sent(wt, T.READS_DONE) == []

    for notice in order:
        if notice == "remove":
            # the conductor removes only once every reader has drained
            wt.tensor_manager.has_inflight_reads = lambda rid: False
            wt._complete_drain_if_ready("r1")
            wt._hard_cleanup("r1")
        elif notice == "abort":
            wt._handle_client_abort("r1")
        else:
            wt._finish_reading("r1", drained=notice == "drained finish")
        for rid in list(wt._draining_rids):  # what each pass of run() does
            wt._complete_drain_if_ready(rid)

    assert _sent(wt, T.ABORT_REQUEST) == ["r1"]
    assert _sent(wt, T.READS_DONE) == ["r1"]
    assert wt._draining_rids == set()
    assert wt._reads_done_sent == set()
    assert wt._failed_rids == set() and wt.in_flight_requests == set()


def test_an_output_queued_before_teardown_is_acked_unread():
    """A failed, self-aborted request is removed while one of its output
    descriptors still waits in the result queue. The real loop handles the
    REMOVE_REQUEST first; the descriptor must then be acked back unread, not
    start a read of a producer buffer that may be reclaimed, recreate the
    request's state, or send a chunk after the request's error."""
    from mstar.api_server.data_worker import PreprocessWorkerThread
    from mstar.utils.ipc_format import RemoveRequest, WorkerMessage, WorkerMessageType

    inbox = [[WorkerMessage(message_type=WorkerMessageType.REMOVE_REQUEST, body=RemoveRequest("r1"))]]
    tm = SimpleNamespace(read_started=[], acked=[], removed=[])
    tm.start_read_tensors = lambda request_id, graph_edges: tm.read_started.append(request_id)
    tm.ack_unread_tensors = lambda request_id, graph_edges: tm.acked.append(request_id)
    tm.force_cleanup_request = tm.removed.append
    tm.has_inflight_reads = lambda rid: False
    tm.get_ready_tensors = lambda: {}
    tm.take_failed_reads = lambda: []
    stop = threading.Event()
    wt = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop,
        communicator=SimpleNamespace(
            get_all_new_messages=lambda: inbox.pop(0) if inbox else [], send=lambda *a: None,
        ),
        tensor_manager=tm,
    )
    # r1 failed and aborted itself; its drain is done and the REMOVE_REQUEST is in
    wt.in_flight_requests.add("r1")
    wt._failed_rids.add("r1")
    wt._draining_rids.add("r1")
    wt._reads_done_sent.add("r1")
    edge = SimpleNamespace(name="text_output", tensor_info=[SimpleNamespace(uuid="u9")])
    wt.result_tensor_queue.put(ResultTensors(
        request_id="r1", modality="text", graph_edge=edge, loop_indices=None,
    ))

    thread = threading.Thread(target=wt.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not (tm.removed and wt.result_tensor_queue.empty() and tm.acked + tm.read_started):
            assert time.monotonic() < deadline, "the loop never handled the descriptor"
            time.sleep(0.005)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert tm.removed == ["r1"]
    assert tm.acked == ["r1"] and tm.read_started == []
    assert wt.out_queue.empty()
    assert "r1" not in wt.request_output_state
    assert "r1" not in wt.tensor_uuid_to_metadata_per_request


class _ShmPreprocessHarness:
    """A data worker with a real SHM tensor manager and no model: requests go
    through real preprocessing, and the test drives the steps of ``run()``."""

    def __init__(self, tmp_path):
        from mstar.api_server.data_worker import PreprocessWorkerThread
        from mstar.communication.tensors import SharedMemoryCommunicationManager

        self.inbox, self.sent = [], []
        comm = SimpleNamespace(
            send=lambda entity, msg: self.sent.append((entity, msg)),
            get_all_new_messages=self._take_inbox,
        )
        self.tm = SharedMemoryCommunicationManager(
            "api_server_preprocess_worker", "host", "cpu", comm, shm_dir=str(tmp_path),
        )
        q = queue.Queue
        self.wt = PreprocessWorkerThread(
            in_queue=q(), result_tensor_queue=q(), out_queue=q(), profile_queue=q(),
            cleanup_request_queue=q(), abort_request_queue=q(), reads_done_queue=q(),
            discard_tensor_queue=q(), stop_event=threading.Event(),
            communicator=comm, tensor_manager=self.tm,
        )

    def _take_inbox(self):
        batch, self.inbox[:] = list(self.inbox), []
        return batch

    def submit(self, rid):
        from mstar.api_server.request_types import PreprocessInput

        pending = getattr(self.wt, "_pending_ids", None)
        if pending is None:
            self.wt._pending_ids = set()
            pending = self.wt._pending_ids
        pending.add(rid)
        self.wt.in_queue.put(PreprocessInput(
            request_id=rid, text="hi", file_paths=None,
            input_modalities=["text"], output_modalities=["text"], model_kwargs={},
        ))

    def fail_output(self, rid):
        edge = SimpleNamespace(name="text_output", tensor_info=[SimpleNamespace(uuid=10**9)])
        self.tm._failed_reads.append(FailedRead(rid, [edge], RuntimeError("Mooncake read failed")))
        self.wt._process_read_tensors()

    def remove(self, rid):
        from mstar.utils.ipc_format import RemoveRequest, WorkerMessage, WorkerMessageType

        self.inbox.append(WorkerMessage(message_type=WorkerMessageType.REMOVE_REQUEST, body=RemoveRequest(rid)))
        self.wt._process_messages()

    def sent_to_conductor(self, message_type):
        return [m.body.request_id for e, m in self.sent if e == "conductor" and m.message_type == message_type]


def _api_on_shm(tmp_path):
    """Real submit/abort on the SHM harness; the test drives admission."""
    from mstar.api_server.data_worker import PreprocessWorker

    h = _ShmPreprocessHarness(tmp_path)
    pending, cancelled = set(), set()
    idxs, counts, held = {}, {}, {}
    admit_lock = threading.Lock()
    h.wt._pending_ids = pending
    h.wt._pre_admit_cancelled = cancelled
    h.wt._admit_lock = admit_lock
    h.wt.output_loop_idxs = idxs
    h.wt.per_request_reading_tensors = counts
    h.wt._held_errors = held
    pw = PreprocessWorker.__new__(PreprocessWorker)
    pw.request_input_queue = h.wt.in_queue
    pw.abort_request_queue = h.wt.abort_request_queue
    pw.reads_done_queue = h.wt.reads_done_queue
    pw.result_tensor_input_queue = h.wt.result_tensor_queue
    pw.discard_tensor_queue = h.wt.discard_tensor_queue
    pw.output_loop_idxs = idxs
    pw.per_request_reading_tensors = counts
    pw._held_errors = held
    pw._pending_ids = pending
    pw._pre_admit_cancelled = cancelled
    pw._admit_lock = admit_lock

    server = APIServer.__new__(APIServer)
    server.request_lock = threading.Lock()
    server.fatal_error = None
    server.pending_requests = {}
    server.recently_completed = collections.OrderedDict()
    server._recently_completed_ttl = 15.0
    server._next_internal_id = 0
    server._id_ns = "t"
    server._client_to_internal = {}
    server._internal_to_client = {}
    pw.output_queue = h.wt.out_queue
    pw.profile_queue = h.wt.profile_queue
    server.preprocess_worker = pw
    server.timeout_seconds = 600.0
    server.enable_nvtx = False
    server.log_stats = False
    server.running = True
    server.conductor_proc = None
    server.on_fatal = None
    server._liveness_interval_s = 0.5
    server.worker_read_quarantine = {}
    server._inbox = []
    server.communicator = SimpleNamespace(
        get_all_new_messages=lambda: server._inbox.pop(0) if server._inbox else [],
    )
    return server, h


def _submit_text(server, rid):
    return server.submit_request(
        text="hi",
        input_modalities=["text"],
        output_modalities=["text"],
        request_id=rid,
    )


def _internal(server, client_id):
    return server._client_to_internal[client_id]


def _collect_chunks(server, client_id):
    async def _run():
        return [c async for c in server.iter_result_chunks(client_id)]
    return asyncio.run(_run())


def _deliver_text(server, internal_id, data=b"hello"):
    req = server.pending_requests[internal_id]
    req.chunks.append(ResultChunk(
        request_id=internal_id, modality="text", data=data,
    ))
    req.event.set()


_FINAL_LOOPS = NestedLoopIndices(
    loop_name_order=[], loop_indices={}, wg_fwd_pass_idx=0,
)


def _request_complete(rid, final_name="text_output"):
    return APIServerMessage(
        message_type="request_complete",
        body=RequestComplete(
            request_id=rid,
            final_outputs={final_name: _FINAL_LOOPS},
            conductor_ingest_time=1.0,
            conductor_finish_time=2.0,
        ),
    )


def _queue_completes(server, *rids):
    """One RequestComplete batch, matching loop idxs so prune can drain."""
    for rid in rids:
        if rid in server.pending_requests:
            server.preprocess_worker.output_loop_idxs.setdefault(rid, {})[
                "text_output"
            ] = _FINAL_LOOPS
    server._inbox.append([_request_complete(rid) for rid in rids])


def _start_msg_loop(server):
    server.running = True
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    return thread


def _stop_msg_loop(server, thread):
    server.running = False
    thread.join(timeout=5)
    assert not thread.is_alive(), "message loop did not stop"


def _wait_until(pred, timeout=5.0, msg="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.001)
    raise AssertionError(f"{msg} not met")


def _all_id_maps_empty(server, h):
    assert server.pending_requests == {}
    assert server._client_to_internal == {}
    assert server._internal_to_client == {}
    assert dict(server.recently_completed) == {}
    assert h.wt._pending_ids == set()
    assert h.wt._pre_admit_cancelled == set()
    assert server.preprocess_worker.output_loop_idxs == {}
    assert server.preprocess_worker.per_request_reading_tensors == {}
    assert getattr(server.preprocess_worker, "_held_errors", {}) == {}
    assert h.wt._reads_done_sent == set()
    assert h.wt._draining_rids == set()
    assert h.wt._failed_rids == set()
    assert h.wt.in_flight_requests == set()
    assert h.wt.tensor_uuid_to_metadata_per_request == {}
    assert h.wt.request_model_kwargs == {}
    assert h.wt.request_output_state == {}


def _remove_after_drain(h, *rids):
    h.wt._drain_api_notices()
    for rid in rids:
        h.remove(rid)


def _submit_admit(server, h, rid):
    _submit_text(server, rid)
    internal = _internal(server, rid)
    h.wt._admit_next_input()
    return internal


def _fail_remove(h, rid):
    h.fail_output(rid)
    h.remove(rid)


def test_cancel_after_resubmit_before_admission_is_not_swallowed(tmp_path):
    """A reused client id mints a new internal id, so the earlier
    submission's teardown cannot swallow the new submission's pre-admit cancel."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    _fail_remove(h, old)
    _submit_text(server, "r1")
    new = _internal(server, "r1")
    assert new != old
    server.abort_request("r1")
    h.wt._admit_next_input()
    assert h.sent_to_conductor(T.NEW_REQUEST) == [old]
    assert new not in h.wt.in_flight_requests


def test_stacked_pre_admission_cancels_skip_then_a_live_submit_admits(tmp_path):
    """Each cancelled submit of the same client id skips NEW_REQUEST; a later
    uncancelled one is still admitted."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    for _ in range(2):
        _submit_text(server, "r1")
        server.abort_request("r1")
        h.wt._admit_next_input()
    _submit_text(server, "r1")
    live = _internal(server, "r1")
    h.wt._admit_next_input()
    assert h.sent_to_conductor(T.NEW_REQUEST) == [live]
    assert live in h.wt.in_flight_requests
    assert h.wt._pre_admit_cancelled == set()


def test_id_bookkeeping_does_not_grow_without_bound(tmp_path):
    """Consumed unique submits leave no pending, cancel, or mapping state."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    n = 10_000
    for i in range(n):
        _submit_text(server, f"u{i}")
        server.abort_request(f"u{i}")
        h.wt._admit_next_input()
    assert h.sent_to_conductor(T.NEW_REQUEST) == []
    _all_id_maps_empty(server, h)


def test_old_abort_does_not_abort_a_new_admitted_request(tmp_path):
    """An abort of the previous internal id must not abort the new one."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    _fail_remove(h, old)
    new = _submit_admit(server, h, "r1")
    server.preprocess_worker.abort_request(old)
    h.wt._drain_api_notices()
    assert new in h.wt.in_flight_requests
    assert new not in h.wt._draining_rids
    assert h.sent_to_conductor(T.NEW_REQUEST) == [old, new]
    assert h.sent_to_conductor(T.ABORT_REQUEST) == [old]


def test_old_notices_do_not_discard_a_new_submission_output(tmp_path):
    """Old abort/finish must not pop the new request's output bookkeeping."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    _fail_remove(h, old)
    server.pending_requests.pop(old, None)
    new = _submit_admit(server, h, "r1")
    server.preprocess_worker.abort_request(old)
    server.preprocess_worker.finished_reading(old, drained=False)
    h.wt._drain_api_notices()
    assert new in server.preprocess_worker.output_loop_idxs

    body = ResultTensors(
        request_id=new, modality="text",
        graph_edge=SimpleNamespace(name="text_output", tensor_info=[]),
        loop_indices=SimpleNamespace(max=lambda other: other),
    )
    server.preprocess_worker.new_result_tensors(body)
    assert server.preprocess_worker.discard_tensor_queue.empty()
    assert server.preprocess_worker.result_tensor_input_queue.get_nowait().request_id == new
    _deliver_text(server, new, b"gen2-output")
    chunks = _collect_chunks(server, "r1")
    assert [(c.request_id, c.data) for c in chunks] == [("r1", b"gen2-output")]
    assert h.sent_to_conductor(T.NEW_REQUEST) == [old, new]


def test_parked_completion_ttl_cannot_fail_a_new_submission(tmp_path):
    """Prune of an older parked completion must not fail the new submit."""
    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    _fail_remove(h, old)
    server._park_completed(old)
    server.pending_requests.pop(old, None)
    new = _submit_admit(server, h, "r1")
    server.recently_completed[old] = time.time() - 20.0
    server._prune_recently_completed()
    req = server.pending_requests[new]
    assert req.error is None and not req.event.is_set()
    _deliver_text(server, new, b"still-alive")
    chunks = _collect_chunks(server, "r1")
    assert [(c.request_id, c.data) for c in chunks] == [("r1", b"still-alive")]


def test_client_sees_its_own_id_in_response_and_stream(tmp_path):
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    assert _submit_text(server, "client-rid") == "client-rid"
    internal = _internal(server, "client-rid")
    assert internal.startswith("client-rid#")
    assert "client-rid" not in server.pending_requests
    h.wt._admit_next_input()
    assert h.sent_to_conductor(T.NEW_REQUEST) == [internal]
    _deliver_text(server, internal, b"payload")
    chunks = _collect_chunks(server, "client-rid")
    assert [c.request_id for c in chunks] == ["client-rid"]
    assert chunks[0].data == b"payload"


def test_a_client_cannot_read_or_abort_another_requests_internal_id(tmp_path):
    """A client-supplied string equal to A's internal id must not resolve to A.

    ``/generate``, the websocket, and the Rust bridge all pass the client
    string into ``iter_result_chunks`` / ``abort_request``, so this is the
    path those surfaces share.
    """
    server, h = _api_on_shm(tmp_path)
    _submit_text(server, "foo")
    a_internal = _internal(server, "foo")
    _deliver_text(server, a_internal, b"secret")
    assert _collect_chunks(server, a_internal) == []
    assert a_internal in server.pending_requests
    assert server.pending_requests[a_internal].consumed_chunks == 0
    server.abort_request(a_internal)
    assert a_internal in server.pending_requests
    _submit_text(server, a_internal)
    b_internal = _internal(server, a_internal)
    assert b_internal != a_internal
    assert server.pending_requests[a_internal].client_id == "foo"


def test_two_submits_with_the_same_client_id_get_the_same_seed(tmp_path):
    from mstar.api_server.entrypoint import _seed_from_client_id
    from mstar.conductor.conductor import _req_id_to_seed

    server, h = _api_on_shm(tmp_path)
    expected = _req_id_to_seed("r1")
    server.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r1", model_kwargs={"seed": None},
    )
    server.submit_request(
        text="hi", input_modalities=["text"], output_modalities=["text"],
        request_id="r1", model_kwargs={"seed": 42},
    )
    _submit_text(server, "r1")
    _submit_text(server, "other")
    seeds = []
    while not h.wt.in_queue.empty():
        seeds.append(h.wt.in_queue.get().model_kwargs.get("seed"))
    assert seeds[0] == expected == _seed_from_client_id("r1")
    assert seeds[1] == 42
    assert seeds[2] == expected
    assert seeds[3] == _seed_from_client_id("other") != expected


def test_a_failed_request_profile_includes_the_read_quarantine(capsys):
    """collect_results must print --log-stats, including quarantine, on error."""
    from mstar.profile.format import ReadQuarantine

    server = APIServer.__new__(APIServer)
    server.request_lock = threading.Lock()
    server.pending_requests = {}
    server.recently_completed = {}
    server._client_to_internal = {"r1": "r1#t-1"}
    server._internal_to_client = {"r1#t-1": "r1"}
    server.log_stats = True
    server.log_stats_file = None
    server.read_quarantines = lambda: [
        ReadQuarantine("worker_0", 3, 4096, 1 << 30),
    ]
    req = PendingRequest(
        streaming=False,
        input_modalities=["text"],
        output_modalities=["text"],
        profile=RequestProfile(rid="r1", timing=RequestTiming(recv_time=0.0)),
        client_id="r1",
        internal_id="r1#t-1",
    )
    req.error = "output tensor read failed"
    req.error_status = 500
    req.event.set()
    server.pending_requests["r1#t-1"] = req
    with pytest.raises(Exception) as exc:
        asyncio.run(server.collect_results("r1"))
    assert exc.value.status_code == 500
    assert "Read quarantine" in capsys.readouterr().out


def _assert_pre_admit_cancel_clean(server, h):
    from mstar.utils.ipc_format import ConductorMessageType as T

    assert h.sent_to_conductor(T.NEW_REQUEST) == []
    assert h.sent_to_conductor(T.ABORT_REQUEST) == []
    assert h.wt._pending_ids == set()
    assert h.wt._pre_admit_cancelled == set()
    assert h.wt.in_flight_requests == set()
    assert server.pending_requests == {}
    assert server._client_to_internal == {}
    assert server.preprocess_worker.output_loop_idxs == {}
    assert server.preprocess_worker.per_request_reading_tensors == {}
    assert getattr(server.preprocess_worker, "_held_errors", {}) == {}


def _run_until_admitted(h, admitted):
    orig = h.wt._admit_next_input

    def _admit():
        orig()
        admitted.set()

    h.wt._admit_next_input = _admit
    loop = threading.Thread(target=h.wt.run, daemon=True)
    loop.start()
    return loop, orig


def test_cancel_queued_then_new_request_then_drain_cleans_maps(tmp_path):
    """Codex ordering: abort is queued, new_request creates the producer
    maps, then run() drains the cancel. Skip-admission must pop the maps."""
    server, h = _api_on_shm(tmp_path)
    release_drain = threading.Event()
    orig_drain = h.wt._drain_api_notices

    def drain():
        release_drain.wait(5)
        return orig_drain()

    h.wt._drain_api_notices = drain
    admitted = threading.Event()
    loop, _ = _run_until_admitted(h, admitted)
    try:
        def hook(internal_id):
            assert internal_id in h.wt._pending_ids
            server.abort_request("r1")

        server._after_publish = hook
        _submit_text(server, "r1")
        assert server.preprocess_worker.output_loop_idxs
        assert server.preprocess_worker.per_request_reading_tensors
        release_drain.set()
        assert admitted.wait(5)
        _assert_pre_admit_cancel_clean(server, h)
    finally:
        release_drain.set()
        h.wt.stop_event.set()
        loop.join(timeout=5)


def test_cancel_recorded_before_enqueue_cleans_maps(tmp_path):
    """Existing ordering: run() records the cancel before new_request."""
    server, h = _api_on_shm(tmp_path)
    admitted = threading.Event()
    loop, _ = _run_until_admitted(h, admitted)
    try:
        recorded = threading.Event()

        def hook(internal_id):
            assert internal_id in h.wt._pending_ids
            server.abort_request("r1")
            deadline = time.time() + 5
            while time.time() < deadline:
                if internal_id in h.wt._pre_admit_cancelled:
                    recorded.set()
                    return
                time.sleep(0.001)
            raise AssertionError("run() dropped the abort as unknown")

        server._after_publish = hook
        _submit_text(server, "r1")
        assert recorded.is_set()
        assert admitted.wait(5)
        _assert_pre_admit_cancel_clean(server, h)
    finally:
        h.wt.stop_event.set()
        loop.join(timeout=5)


def test_new_request_then_abort_before_admit_cleans_maps(tmp_path):
    """Reverse ordering: maps and enqueue first, then abort, then admit."""
    server, h = _api_on_shm(tmp_path)
    release_admit = threading.Event()
    admitted = threading.Event()
    orig_admit = h.wt._admit_next_input

    def _admit():
        release_admit.wait(5)
        orig_admit()
        admitted.set()

    h.wt._admit_next_input = _admit
    loop = threading.Thread(target=h.wt.run, daemon=True)
    loop.start()
    try:
        _submit_text(server, "r1")
        assert server.preprocess_worker.output_loop_idxs
        server.abort_request("r1")
        release_admit.set()
        assert admitted.wait(5)
        _assert_pre_admit_cancel_clean(server, h)
    finally:
        release_admit.set()
        h.wt.stop_event.set()
        loop.join(timeout=5)


def test_delayed_drain_does_not_leak_producer_maps(tmp_path):
    """10k Codex-order cancels: queue abort, create maps, then drain."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    for i in range(10_000):
        rid = f"u{i}"

        def hook(_internal_id, client=rid):
            server.abort_request(client)

        server._after_publish = hook
        _submit_text(server, rid)
        h.wt._drain_api_notices()
        h.wt._admit_next_input()
    assert h.sent_to_conductor(T.NEW_REQUEST) == []
    _all_id_maps_empty(server, h)


def test_client_error_text_does_not_contain_the_internal_id(tmp_path):
    server, _h = _api_on_shm(tmp_path)
    _submit_text(server, "client-rid")
    internal = _internal(server, "client-rid")
    leaked = f"Mooncake memory registration failed for request id {internal}, uuid 1."
    req = server.pending_requests[internal]
    req.chunks.append(ResultChunk(
        request_id=internal, modality="error",
        data=leaked.encode(), metadata={"status": 500},
    ))
    req.event.set()
    chunks = _collect_chunks(server, "client-rid")
    text = chunks[0].data.decode()
    assert "client-rid" in text
    assert internal not in text
    assert chunks[0].request_id == "client-rid"


def test_empty_client_id_is_rewritten_in_error_text(tmp_path):
    server, _h = _api_on_shm(tmp_path)
    _submit_text(server, "")
    internal = _internal(server, "")
    leaked = f"Mooncake memory registration failed for request id {internal}, uuid 1."
    req = server.pending_requests[internal]
    req.chunks.append(ResultChunk(
        request_id=internal, modality="error",
        data=leaked.encode(), metadata={"status": 500},
    ))
    req.event.set()
    chunks = _collect_chunks(server, "")
    text = chunks[0].data.decode()
    assert internal not in text
    assert chunks[0].request_id == ""


def test_superseded_completion_does_not_drop_the_next_in_the_batch(tmp_path):
    """An old superseded completion and a healthy one in the same
    ``_process_messages`` batch: the healthy request still gets its metadata."""
    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    new = _submit_admit(server, h, "r1")
    assert old not in server.pending_requests
    loop = _start_msg_loop(server)
    try:
        _queue_completes(server, old, new)
        _wait_until(
            lambda: (
                new in server.pending_requests
                and bool(server.pending_requests[new].final_outputs)
                and server.pending_requests[new].event.is_set()
            ),
            msg="healthy completion metadata",
        )
        req = server.pending_requests[new]
        assert req.final_outputs
        assert req.profile.timing.conductor_ingest_time == 1.0
        assert req.profile.timing.conductor_finish_time == 2.0
        assert req.event.is_set()
        h.wt._drain_api_notices()
    finally:
        _stop_msg_loop(server, loop)


def test_cancel_after_teardown_sends_reads_done_once(tmp_path):
    """Attach, supersede, complete+prune+REMOVE, then cancel the collector
    before it consumes: READS_DONE stays 1 and drain is not re-armed."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    loop = _start_msg_loop(server)

    async def _run():
        attached = asyncio.Event()
        orig = server._attach_consumer

        def hook(iid):
            orig(iid)
            attached.set()

        server._attach_consumer = hook
        task = asyncio.create_task(server.collect_results("r1"))
        await attached.wait()
        new = _submit_admit(server, h, "r1")
        assert new != old
        assert old in server.pending_requests
        _queue_completes(server, old)
        _wait_until(
            lambda: (
                server.pending_requests.get(old) is not None
                and server.pending_requests[old].event.is_set()
                and old not in server.recently_completed
            ),
            msg="old request pruned",
        )
        h.wt._drain_api_notices()
        assert h.sent_to_conductor(T.READS_DONE) == [old]
        h.remove(old)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        h.wt._drain_api_notices()
        assert h.sent_to_conductor(T.READS_DONE) == [old]
        assert old not in h.wt._draining_rids
        assert old not in h.wt._reads_done_sent

    try:
        asyncio.run(_run())
    finally:
        _stop_msg_loop(server, loop)


def test_reclaimed_internal_id_is_not_a_public_lookup(tmp_path):
    """After the latest submit is aborted, maps can be empty while an old
    internal id is still parked. That id is not a client-facing handle."""
    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    new = _submit_admit(server, h, "r1")
    h.wt._drain_api_notices()
    loop = _start_msg_loop(server)
    try:
        _queue_completes(server, old)
        server.abort_request("r1")
        h.wt._drain_api_notices()
        assert server._client_to_internal == {}
        assert server._internal_to_client == {}
        with server.request_lock:
            server.recently_completed[old] = time.time()
        assert server._lookup_client(old) is None
        server.abort_request(old)
        assert old in server.recently_completed
        assert server._lookup_client(old) is None
    finally:
        _stop_msg_loop(server, loop)


def test_unattached_superseded_submit_is_aborted(tmp_path):
    """A reuse with no attached consumer aborts the previous submit once."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    new = _submit_admit(server, h, "r1")
    assert old not in server.pending_requests
    assert new in server.pending_requests
    h.wt._drain_api_notices()
    assert h.sent_to_conductor(T.ABORT_REQUEST) == [old]
    loop = _start_msg_loop(server)
    try:
        _queue_completes(server, new)

        async def _run():
            chunks = await server.collect_results("r1")
            assert chunks == []

        asyncio.run(_run())
        _wait_until(
            lambda: new not in server.recently_completed,
            msg="latest prune",
        )
        h.wt._drain_api_notices()
        assert h.sent_to_conductor(T.ABORT_REQUEST) == [old]
        _remove_after_drain(h, old, new)
        _all_id_maps_empty(server, h)
    finally:
        _stop_msg_loop(server, loop)


def test_attached_superseded_submit_still_gets_its_output(tmp_path):
    """A consumer attached before reuse stays bound to that submit and
    receives its full output."""
    server, h = _api_on_shm(tmp_path)
    old = _submit_admit(server, h, "r1")
    loop = _start_msg_loop(server)

    async def _run():
        attached = asyncio.Event()
        orig = server._attach_consumer

        def hook(iid):
            orig(iid)
            attached.set()

        server._attach_consumer = hook
        task = asyncio.create_task(
            _agen_list(server.iter_result_chunks("r1"))
        )
        await attached.wait()
        new = _submit_admit(server, h, "r1")
        assert new != old
        assert old in server.pending_requests
        assert server.pending_requests[old].consumer_attached
        h.wt._drain_api_notices()
        from mstar.utils.ipc_format import ConductorMessageType as T
        assert h.sent_to_conductor(T.ABORT_REQUEST) == []
        server.pending_requests[old].chunks.append(ResultChunk(
            request_id=old, modality="text", data=b"first-out",
        ))
        _queue_completes(server, old)
        first = await task
        assert [c.data for c in first] == [b"first-out"]
        server.pending_requests[new].chunks.append(ResultChunk(
            request_id=new, modality="text", data=b"second-out",
        ))
        _queue_completes(server, new)
        second = await server.collect_results("r1")
        assert [c.data for c in second] == [b"second-out"]
        _wait_until(
            lambda: not server.recently_completed,
            msg="both pruned",
        )
        h.wt._drain_api_notices()
        _remove_after_drain(h, old, new)
        _all_id_maps_empty(server, h)

    try:
        asyncio.run(_run())
    finally:
        _stop_msg_loop(server, loop)


def test_overlapping_reuse_loop_leaves_no_maps(tmp_path):
    """10k overlapping pairs through real RequestComplete messages."""
    server, h = _api_on_shm(tmp_path)
    loop = _start_msg_loop(server)

    async def _run():
        for _ in range(10_000):
            old = _submit_admit(server, h, "r1")
            new = _submit_admit(server, h, "r1")
            _remove_after_drain(h, old)
            _queue_completes(server, new)
            await server.collect_results("r1")
            _wait_until(
                lambda done=new: done not in server.recently_completed,
                msg="pair prune",
            )
            _remove_after_drain(h, new)
        _all_id_maps_empty(server, h)

    try:
        asyncio.run(_run())
    finally:
        _stop_msg_loop(server, loop)


def test_healthy_completion_loop_leaves_no_maps(tmp_path):
    """10k healthy completes through real RequestComplete messages."""
    server, h = _api_on_shm(tmp_path)
    loop = _start_msg_loop(server)

    async def _run():
        for i in range(10_000):
            rid = _submit_admit(server, h, f"u{i}")
            _queue_completes(server, rid)
            await server.collect_results(f"u{i}")
            _wait_until(
                lambda done=rid: done not in server.recently_completed,
                msg="healthy prune",
            )
            _remove_after_drain(h, rid)
        _all_id_maps_empty(server, h)

    try:
        asyncio.run(_run())
    finally:
        _stop_msg_loop(server, loop)


def test_admitted_abort_loop_leaves_no_maps(tmp_path):
    server, h = _api_on_shm(tmp_path)
    for i in range(10_000):
        rid = _submit_admit(server, h, f"u{i}")
        server.abort_request(f"u{i}")
        _remove_after_drain(h, rid)
    _all_id_maps_empty(server, h)


class _BoomPrompt:
    def process_prompt(self, *args, **kwargs):
        raise ValueError("bad prompt")


def _error_chunks(chunks):
    return [c for c in chunks if c.modality == "error"]


def test_output_error_survives_cleanup_before_get_result_chunks(tmp_path):
    """Output read fails, then REMOVE, then get_result_chunks: one error."""
    server, h = _api_on_shm(tmp_path)
    rid = _submit_admit(server, h, "r1")
    _fail_remove(h, rid)
    assert rid not in h.wt.in_flight_requests
    assert rid not in server.preprocess_worker.per_request_reading_tensors
    chunks = server.preprocess_worker.get_result_chunks()
    errors = _error_chunks(chunks)
    assert len(errors) == 1
    assert b"Mooncake read failed" in errors[0].data
    assert rid not in server.preprocess_worker._held_errors


def test_preprocess_error_survives_cleanup_before_get_result_chunks(tmp_path):
    """Preprocess raises, immediate cleanup, then get_result_chunks: one error."""
    server, h = _api_on_shm(tmp_path)
    h.wt.model = _BoomPrompt()
    rid = _submit_admit(server, h, "r1")
    assert rid not in h.wt.in_flight_requests
    assert rid not in server.preprocess_worker.per_request_reading_tensors
    chunks = server.preprocess_worker.get_result_chunks()
    errors = _error_chunks(chunks)
    assert len(errors) == 1
    assert b"bad prompt" in errors[0].data
    assert errors[0].metadata["status"] == 400
    assert rid not in server.preprocess_worker._held_errors


def test_output_error_reaches_collector_after_teardown(tmp_path):
    """REMOVE before the API loop polls; collect_results gets one labelled error."""
    from fastapi import HTTPException

    server, h = _api_on_shm(tmp_path)
    server.timeout_seconds = 2.0
    loop = _start_msg_loop(server)
    try:
        rid = _submit_admit(server, h, "r1")
        _fail_remove(h, rid)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(server.collect_results("r1"))
        assert exc.value.status_code == 500
        assert "Mooncake read failed" in exc.value.detail
        assert "timed out" not in exc.value.detail
        assert rid not in exc.value.detail
        assert rid not in server.pending_requests
        _wait_until(
            lambda: rid not in server.recently_completed, msg="output-fail prune",
        )
        _all_id_maps_empty(server, h)
    finally:
        _stop_msg_loop(server, loop)


def test_preprocess_error_reaches_collector_after_cleanup(tmp_path):
    """Preprocess cleanup before the API loop polls; no 500 timeout."""
    from fastapi import HTTPException

    server, h = _api_on_shm(tmp_path)
    server.timeout_seconds = 2.0
    h.wt.model = _BoomPrompt()
    loop = _start_msg_loop(server)
    try:
        rid = _submit_admit(server, h, "r1")
        with pytest.raises(HTTPException) as exc:
            asyncio.run(server.collect_results("r1"))
        assert exc.value.status_code == 400
        assert "bad prompt" in exc.value.detail
        assert "timed out" not in exc.value.detail
        assert rid not in exc.value.detail
        assert rid not in server.pending_requests
        _wait_until(
            lambda: rid not in server.recently_completed, msg="preprocess-fail prune",
        )
        _all_id_maps_empty(server, h)
    finally:
        _stop_msg_loop(server, loop)


def _complete_in_api(server, *rids):
    """Apply RequestComplete for ``rids`` (loop indices already final)."""
    _queue_completes(server, *rids)
    for message in server._inbox.pop(0):
        server._route_message(message)


def _teardown_after_pending_check(server, h, rid):
    """Run the data worker's REMOVE teardown of ``rid`` right after prune's
    pending check and before it reads the loop indices, as the worker thread
    can between the two."""
    pw = server.preprocess_worker
    check = pw.has_pending_tensors
    fired = []

    def has_pending_tensors(request_id):
        pending = check(request_id)
        if request_id == rid and not fired:
            fired.append(request_id)
            h.remove(request_id)
        return pending

    pw.has_pending_tensors = has_pending_tensors
    return fired


def test_teardown_between_the_pending_check_and_the_read_completes_normally(tmp_path, caplog):
    server, h = _api_on_shm(tmp_path)
    rid = _submit_admit(server, h, "r1")
    _complete_in_api(server, rid)
    server.pending_requests[rid].chunks.append(
        ResultChunk(request_id="r1", modality="text", data=b"hello"),
    )
    fired = _teardown_after_pending_check(server, h, rid)
    with caplog.at_level(logging.WARNING):
        with server.request_lock:
            server._prune_recently_completed()
    assert fired == [rid]
    assert rid not in server.preprocess_worker.output_loop_idxs
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    req = server.pending_requests[rid]
    assert req.event.is_set() and req.error is None
    assert rid not in server.recently_completed
    assert [c.data for c in asyncio.run(server.collect_results("r1"))] == [b"hello"]
    h.wt._drain_api_notices()
    _all_id_maps_empty(server, h)


def test_teardown_before_the_api_takes_the_error_still_delivers_it(tmp_path):
    """Same interleaving after an output failure whose error is still queued:
    a torn-down request is not reported drained until its error is taken."""
    from fastapi import HTTPException

    server, h = _api_on_shm(tmp_path)
    rid = _submit_admit(server, h, "r1")
    _complete_in_api(server, rid)
    h.fail_output(rid)
    fired = _teardown_after_pending_check(server, h, rid)
    with server.request_lock:
        server._prune_recently_completed()
    assert fired == [rid]
    assert not server.pending_requests[rid].event.is_set()
    assert rid in server.recently_completed
    for chunk in server.preprocess_worker.get_result_chunks():
        server._route_result_chunk(chunk)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.collect_results("r1"))
    assert exc.value.status_code == 500
    assert "Mooncake read failed" in exc.value.detail
    with server.request_lock:
        server._prune_recently_completed()
    h.wt._drain_api_notices()
    assert h.sent_to_conductor(ConductorMessageType.ABORT_REQUEST) == [rid]
    _all_id_maps_empty(server, h)


def test_a_raising_drain_check_fails_only_its_request(tmp_path, caplog):
    server, h = _api_on_shm(tmp_path)
    bad = _submit_admit(server, h, "bad")
    good = _submit_admit(server, h, "good")
    _complete_in_api(server, bad, good)
    pw = server.preprocess_worker
    check = pw.received_final_chunks

    def received_final_chunks(request_id, final_outputs):
        if request_id == bad:
            raise RuntimeError("drain check bug")
        return check(request_id, final_outputs)

    pw.received_final_chunks = received_final_chunks
    with caplog.at_level(logging.ERROR):
        with server.request_lock:
            server._prune_recently_completed()
    assert dict(server.recently_completed) == {}
    assert server.pending_requests[good].event.is_set()
    assert server.pending_requests[good].error is None
    failed = server.pending_requests[bad]
    assert failed.event.is_set() and failed.error_status == 500
    assert failed.error == "result delivery failed; response is incomplete"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and bad in errors[0].getMessage()
    assert "drain check bug" in str(errors[0].exc_info[1])


def test_a_bad_message_or_chunk_does_not_drop_the_rest_of_the_batch(tmp_path, caplog):
    server, h = _api_on_shm(tmp_path)
    server.timeout_seconds = 2.0
    bad = _submit_admit(server, h, "bad")
    good = _submit_admit(server, h, "good")
    pw = server.preprocess_worker
    pw.output_loop_idxs[good]["text_output"] = _FINAL_LOOPS
    pw.per_request_reading_tensors[good] = 1
    route_tensors = pw.new_result_tensors

    def new_result_tensors(body):
        if body.request_id == bad:
            raise RuntimeError("routing bug")
        route_tensors(body)

    pw.new_result_tensors = new_result_tensors
    h.wt.out_queue.put(ResultChunk(request_id=bad, modality="error", data=None))
    h.wt.out_queue.put(ResultChunk(request_id=good, modality="text", data=b"hello"))
    server._inbox.append([
        APIServerMessage(
            message_type="result_tensors",
            body=ResultTensors(request_id=bad, modality="text", graph_edge=None, loop_indices=None),
        ),
        _request_complete(good),
    ])
    with caplog.at_level(logging.ERROR):
        loop = _start_msg_loop(server)
        try:
            chunks = asyncio.run(server.collect_results("good"))
        finally:
            _stop_msg_loop(server, loop)
    assert [c.data for c in chunks] == [b"hello"]
    logged = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("result_tensors" in m and bad in m for m in logged)
    assert any("the error chunk" in m and bad in m for m in logged)
    assert not any("Error in message processing loop" in m for m in logged)


class _TokenModel:
    """Detokenizes ``v`` to ``<v>``; a negative value fails postprocessing."""

    def postprocess(self, output, modality, request_kwargs=None):
        value = int(output[0])
        if value < 0:
            raise ValueError("bad token")
        return f"<{value}>".encode()


def _token_frame(server, rid, value, step=0):
    """Route one RESULT_TOKENS frame through the API message path."""
    loops = NestedLoopIndices(
        loop_name_order=["decode_loop"], loop_indices={"decode_loop": step},
        wg_fwd_pass_idx=step + 1,
    )
    server._route_message(APIServerMessage(
        message_type="result_tokens",
        body=ResultTokens(
            request_ids=[rid], values=[value], loop_indices=[loops],
            signal="text_output", modality="text",
        ),
    ))


def _drain_inline_tokens(h):
    """The inline branch of the data worker's run() loop."""
    from mstar.api_server.data_worker import InlineToken

    while not h.wt.result_tensor_queue.empty():
        item = h.wt.result_tensor_queue.get()
        assert isinstance(item, InlineToken)
        h.wt._emit_inline_token(item)


def test_an_inline_token_after_its_requests_error_is_dropped(tmp_path):
    """An inline postprocess failure goes through the first-error path (one
    held error, one abort); later tokens, before or after teardown, emit
    nothing and recreate no state."""
    from fastapi import HTTPException

    server, h = _api_on_shm(tmp_path)
    rid = _submit_admit(server, h, "r1")
    h.wt.model = _TokenModel()
    for step, value in enumerate((5, -1, 7)):
        _token_frame(server, rid, value, step)
    _drain_inline_tokens(h)
    assert rid in h.wt._failed_rids
    assert rid in server.preprocess_worker._held_errors
    assert h.sent_to_conductor(ConductorMessageType.ABORT_REQUEST) == [rid]
    for chunk in server.preprocess_worker.get_result_chunks():
        server._route_result_chunk(chunk)
    _token_frame(server, rid, 8, 3)          # queued before teardown
    h.remove(rid)
    _drain_inline_tokens(h)
    _token_frame(server, rid, 9, 4)          # arrives after teardown
    assert h.wt.result_tensor_queue.empty()
    assert h.wt.out_queue.empty()
    assert rid not in h.wt.request_output_state
    assert rid not in server.preprocess_worker.output_loop_idxs
    req = server.pending_requests[rid]
    assert [(c.modality, c.data[:3]) for c in req.chunks] == [("text", b"<5>"), ("error", b"tex")]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.collect_results("r1"))
    assert exc.value.status_code == 400 and "bad token" in exc.value.detail
    with server.request_lock:
        server._prune_recently_completed()
    h.wt._drain_api_notices()
    assert h.sent_to_conductor(ConductorMessageType.ABORT_REQUEST) == [rid]
    _all_id_maps_empty(server, h)


def test_an_inline_token_for_a_draining_request_is_dropped(tmp_path):
    server, h = _api_on_shm(tmp_path)
    rid = _submit_admit(server, h, "r1")
    h.wt.model = _TokenModel()
    _token_frame(server, rid, 5)             # queued, then the client aborts
    server.abort_request("r1")
    h.wt._drain_api_notices()
    assert rid in h.wt._draining_rids
    _drain_inline_tokens(h)
    assert h.wt.out_queue.empty()
    assert h.sent_to_conductor(ConductorMessageType.READS_DONE) == [rid]
    h.remove(rid)
    _all_id_maps_empty(server, h)


def test_new_result_token_waits_for_the_admit_lock_and_respects_teardown(tmp_path):
    """Teardown holding ``_admit_lock`` blocks a token until it is done, and
    the token then sees the request gone; a token admitted just before
    teardown is dropped by the data worker without recreating state."""
    server, h = _api_on_shm(tmp_path)
    pw = server.preprocess_worker
    rid = _submit_admit(server, h, "r1")
    h.wt.model = _TokenModel()
    _token_frame(server, rid, 5)             # admitted before teardown
    with pw._admit_lock:
        racer = threading.Thread(
            target=pw.new_result_token,
            args=(rid, 6, None, "text_output", "text"),
        )
        racer.start()
        racer.join(timeout=0.2)
        assert racer.is_alive(), "new_result_token ran without the admit lock"
        h.wt._drop_producer_state(rid)       # teardown's pops, under the lock
    racer.join(timeout=5)
    assert not racer.is_alive()
    assert rid not in pw.output_loop_idxs and rid not in pw.per_request_reading_tensors
    assert h.wt.result_tensor_queue.qsize() == 1
    h.remove(rid)
    _drain_inline_tokens(h)
    assert h.wt.out_queue.empty()
    assert rid not in h.wt.request_output_state


def test_a_bad_entry_in_a_token_frame_does_not_drop_the_others(tmp_path, caplog):
    server, h = _api_on_shm(tmp_path)
    bad = _submit_admit(server, h, "bad")
    good = _submit_admit(server, h, "good")
    h.wt.model = _TokenModel()
    pw = server.preprocess_worker
    route = pw.new_result_token

    def new_result_token(rid, *args, **kwargs):
        if rid == bad:
            raise RuntimeError("token bug")
        route(rid, *args, **kwargs)

    pw.new_result_token = new_result_token
    wakes = []
    pw.wake = lambda: wakes.append(1)
    with caplog.at_level(logging.ERROR):
        server._route_message(APIServerMessage(
            message_type="result_tokens",
            body=ResultTokens(
                request_ids=[bad, good], values=[1, 2], loop_indices=[None, None],
                signal="text_output", modality="text",
            ),
        ))
    assert wakes == [1]  # one wake for the frame, after the good value
    _drain_inline_tokens(h)
    assert [(c.request_id, c.data) for c in server.preprocess_worker.get_result_chunks()] == [
        (good, b"<2>"),
    ]
    logged = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(logged) == 1 and bad in logged[0]


def _trigger_failure(server, h, rid, path):
    """Fail or tear down ``rid`` the way ``path`` does, off the event loop."""
    if path == "output_read_error":
        h.fail_output(rid)
    elif path == "inline_token_error":
        _token_frame(server, rid, -1)
        _drain_inline_tokens(h)
    elif path == "engine_failure":
        server._inbox.append([APIServerMessage(
            message_type="request_failed",
            body=RequestFailed(request_id=rid, error_message="engine boom", status=500),
        )])
    elif path == "ttl":
        server._recently_completed_ttl = 0.0
        with server.request_lock:
            server._park_completed(rid)
    elif path == "aborted":
        server._abort_internal(rid)


@pytest.mark.parametrize("path, expected", [
    ("output_read_error", "Mooncake read failed"),
    ("inline_token_error", "bad token"),
    ("engine_failure", "engine boom"),
    ("ttl", "result delivery timed out"),
    ("aborted", None),
])
def test_every_error_and_teardown_path_wakes_the_stream(tmp_path, monkeypatch, path, expected):
    """With the poll backstop pushed to 30 s, the stream only ends in time
    if the failing path notifies its handler."""
    monkeypatch.setattr(APIServer, "_stream_wake_backstop_s", 30.0)
    server, h = _api_on_shm(tmp_path)
    rid = _submit_admit(server, h, "r1")
    h.wt.model = _TokenModel()
    loop = _start_msg_loop(server)

    async def _run():
        agen = server.iter_result_chunks("r1")
        first = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.1)             # the handler is asleep on its wake
        assert not first.done()
        _trigger_failure(server, h, rid, path)
        chunks = []
        try:
            chunks.append(await asyncio.wait_for(first, timeout=5))
            chunks.extend([c async for c in agen])
        except StopAsyncIteration:
            pass
        return chunks

    try:
        t0 = time.monotonic()
        chunks = asyncio.run(_run())
        assert time.monotonic() - t0 < 5
    finally:
        _stop_msg_loop(server, loop)
    if expected is None:
        assert chunks == []
    else:
        assert [c.modality for c in chunks] == ["error"]
        assert expected in chunks[0].data.decode()


def test_output_failure_loop_leaves_no_maps(tmp_path):
    """10k output-read failures: consume the error through the real API
    loop after transport teardown, then let normal prune reclaim state."""
    from fastapi import HTTPException

    server, h = _api_on_shm(tmp_path)
    server.timeout_seconds = 2.0
    loop = _start_msg_loop(server)

    async def _run():
        for i in range(10_000):
            client = f"u{i}"
            rid = _submit_admit(server, h, client)
            _fail_remove(h, rid)
            with pytest.raises(HTTPException) as exc:
                await server.collect_results(client)
            assert exc.value.status_code == 500
            assert "Mooncake read failed" in exc.value.detail
            assert "timed out" not in exc.value.detail
            _wait_until(
                lambda done=rid: done not in server.recently_completed,
                msg="fail prune",
            )
        _all_id_maps_empty(server, h)

    try:
        asyncio.run(_run())
    finally:
        _stop_msg_loop(server, loop)


def test_disconnect_loop_leaves_no_maps(tmp_path):
    """10k attached-then-cancelled collectors, the disconnect abort path."""
    server, h = _api_on_shm(tmp_path)
    for i in range(10_000):
        rid = _submit_admit(server, h, f"u{i}")
        with server.request_lock:
            server._attach_consumer(rid)
        server._abort_internal(rid)
        _remove_after_drain(h, rid)
    _all_id_maps_empty(server, h)


def test_ttl_expiry_loop_leaves_no_maps(tmp_path):
    """10k parked completions expire via TTL and leave no per-request state."""
    server, h = _api_on_shm(tmp_path)
    server._recently_completed_ttl = 0.0
    loop = _start_msg_loop(server)
    try:
        for i in range(10_000):
            rid = _submit_admit(server, h, f"u{i}")
            server._inbox.append([_request_complete(rid)])
            _wait_until(
                lambda done=rid: (
                    server.pending_requests.get(done) is not None
                    and server.pending_requests[done].event.is_set()
                    and done not in server.recently_completed
                ),
                msg="ttl prune",
            )
            server.pending_requests.pop(rid, None)
            server._forget_mapping(rid)
            _remove_after_drain(h, rid)
        _all_id_maps_empty(server, h)
    finally:
        _stop_msg_loop(server, loop)


async def _agen_list(agen):
    return [item async for item in agen]
