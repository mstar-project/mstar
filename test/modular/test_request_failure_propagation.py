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

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from mstar.api_server.entrypoint import APIServer, PendingRequest
from mstar.api_server.request_types import (
    APIServerMessage,
    RequestFailed,
    ResultChunk,
    ResultTensors,
)
from mstar.profile.format import RequestProfile, RequestTiming
from mstar.utils.ipc_format import ConductorMessageType, FailRequests
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import PendingBatch, Worker

# ── worker ─────────────────────────────────────────────────────────────────


def _worker(known_rids=("r1", "r2")):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.sent = []
    w.communicator = SimpleNamespace(
        send=lambda entity_id, msg: w.sent.append((entity_id, msg))
    )
    w.worker_graphs_manager = SimpleNamespace(
        per_request_info={rid: object() for rid in known_rids}
    )
    w.scheduler = MicroScheduler.__new__(MicroScheduler)
    w.scheduler.failed_rids = set()
    w.scheduler.held_until = {}
    w.scheduler.admit_errors = {}
    w.scheduler.backlog = {}
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
            node_objects={
                rid: SimpleNamespace(_speculatively_scheduled=True) for rid in rids
            },
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
        # never be re-queued.
        assert all(
            not n._speculatively_scheduled
            for n in pending.batch.node_objects.values()
        )
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
        node_objects={"r2": SimpleNamespace(_speculatively_scheduled=True)},
        request_to_worker_graph={"r2": "wg"},
    )
    w._handle_main_loop_error(RuntimeError("boom"), (None, None), scheduled)
    assert set(w.sent[0][1].body.errors) == {"r1", "r2"}


# ── api server ─────────────────────────────────────────────────────────────


def _api_server(messages):
    s = APIServer.__new__(APIServer)
    s.pending_requests = {}
    s.recently_completed = {}
    s._recently_completed_ttl = 15.0
    s.request_lock = threading.Lock()
    s.running = True
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


# ── data worker ────────────────────────────────────────────────────────────


def _preprocess_thread(model):
    from mstar.api_server.data_worker import PreprocessWorkerThread

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue = queue.Queue()
    wt.model = model
    wt.request_model_kwargs = {}
    wt.tensor_uuid_to_metadata_per_request = {"r1": {"u1": {}}}
    wt.enable_prof = False
    return wt


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
        get_tensor=lambda request_id, uuid: object(),
        dereference=lambda request_id, uuid: dereferenced.append(uuid),
    )

    assert wt._process_read_tensors() is True
    chunk: ResultChunk = wt.out_queue.get_nowait()
    assert chunk.request_id == "r1"
    assert chunk.modality == "error"
    assert b"decode failed" in chunk.data
    assert chunk.metadata["status"] == 500
    # The tensor is still released even though postprocessing died.
    assert dereferenced == ["u1"]


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
