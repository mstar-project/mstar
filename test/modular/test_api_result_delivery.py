"""API-server result delivery under load.

Covers the completion-vs-delivery race: a finished request's chunks flow
api-side through the data worker, whose single thread also runs media
preprocessing. The worker must drain queued result reads ahead of (multi-
second) preprocess items, and the API server's post-completion TTL must fail
a request whose chunks never arrived instead of closing it as an empty
success.
"""

import asyncio
import collections
import queue
import threading
import time

import torch

from mstar.api_server.data_worker import (
    DeliveryProgress,
    PreprocessWorker,
    PreprocessWorkerThread,
)
from mstar.api_server.entrypoint import APIServer, PendingRequest
from mstar.api_server.request_types import PreprocessInput, ResultChunk, ResultTensors
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices


class _RecordingTensorManager:
    def __init__(self):
        self.read_started = []

    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        self.read_started.append(request_id)
        return []

    def get_ready_tensors(self):
        return {}

    def cleanup_request(self, request_id):
        pass

    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass

    def set_persist(self, request_id, uuid, persist):
        pass

    def ack_unread_tensors(self, request_id, graph_edges):
        pass

    def force_cleanup_request(self, request_id):
        pass


class _RecordingCommunicator:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass


class _BlockingModel:
    """process_prompt blocks until released, like a long video preprocess."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()

    def process_prompt(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(timeout=10)
        return {}


def _result_tensors(rid, name="new_token"):
    return ResultTensors(
        request_id=rid,
        modality="text",
        graph_edge=GraphEdge(next_node="emit_to_client", name=name),
        loop_indices=NestedLoopIndices(
            loop_name_order=[], loop_indices={}, wg_fwd_pass_idx=0,
        ),
    )


def test_result_reads_drain_ahead_of_preprocess():
    """Queued result-tensor reads must all start before a preprocess item
    (which can block for seconds on media decode) is picked up."""
    model = _BlockingModel()
    tm = _RecordingTensorManager()
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(),
        result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(),
        profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(),
        stop_event=stop,
        communicator=_RecordingCommunicator(),
        tensor_manager=tm,
        model=model,
    )

    worker.in_queue.put(PreprocessInput(
        request_id="req-preprocess",
        text="hello",
        file_paths=None,
        input_modalities=["text"],
        output_modalities=["text"],
        model_kwargs={},
    ))
    n_results = 6
    for i in range(n_results):
        worker.result_tensor_queue.put(_result_tensors(f"req-out-{i}"))

    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert model.entered.wait(timeout=5), "preprocess never started"
        # The preprocess is still blocked; every queued read must already
        # have been started.
        assert len(tm.read_started) == n_results
    finally:
        model.release.set()
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


class _StubPreprocessWorker:
    def __init__(self, pending=False, final=True, active=False):
        self.pending = pending
        self.final = final
        self.active = active
        self.cleaned = []
        self.drained_flags = []

    def has_pending_tensors(self, rid):
        return self.pending

    def delivery_active(self, rid, since):
        return self.active

    def received_final_chunks(self, rid, final_outputs):
        return self.final

    def finished_reading(self, rid, drained=True):
        # New teardown: the API server signals READS_DONE to the conductor
        # instead of cleaning up locally; the conductor drives the hard cleanup.
        self.cleaned.append(rid)
        self.drained_flags.append(drained)


def _pending_request():
    return PendingRequest(
        streaming=True,
        input_modalities=["text"],
        output_modalities=["text"],
        profile=None,
    )


def _api_server_stub(preprocess_worker):
    server = object.__new__(APIServer)
    server.recently_completed = collections.OrderedDict()
    server._recently_completed_ttl = 15.0
    server.pending_requests = {}
    server.preprocess_worker = preprocess_worker
    server.log_stats = False
    server.request_lock = threading.Lock()
    server.timeout_seconds = 5.0
    return server


def test_ttl_expiry_with_undelivered_chunks_fails_request():
    pw = _StubPreprocessWorker(pending=True, final=False)
    server = _api_server_stub(pw)
    req = _pending_request()
    server.pending_requests["r1"] = req
    server.recently_completed["r1"] = time.time() - 20.0

    server._prune_recently_completed()

    assert req.event.is_set()
    assert req.error is not None
    assert req.error_status == 500
    assert "r1" not in server.recently_completed
    assert pw.cleaned == ["r1"]
    # Chunks were still pending, so reads may be in flight: the READS_DONE ACK
    # must be gated on them rather than sent outright.
    assert pw.drained_flags == [False]


def test_ttl_holds_while_the_worker_is_still_delivering():
    """A read or postprocess that outlives the TTL (a video encode on the
    data worker's single thread) is delivery in progress, not a lost chunk:
    the request stays open until the worker goes quiet, then the TTL
    applies as before."""
    pw = _StubPreprocessWorker(pending=True, final=False, active=True)
    server = _api_server_stub(pw)
    req = _pending_request()
    server.pending_requests["r5"] = req
    server.recently_completed["r5"] = time.time() - 20.0

    server._prune_recently_completed()

    assert not req.event.is_set()
    assert req.error is None
    assert "r5" in server.recently_completed
    assert pw.cleaned == []

    pw.active = False
    server._prune_recently_completed()

    assert req.event.is_set()
    assert req.error_status == 500
    assert pw.cleaned == ["r5"]
    assert pw.drained_flags == [False]


def test_delivery_active_reads_the_worker_state():
    pw = object.__new__(PreprocessWorker)
    pw.delivery = DeliveryProgress()
    pw.per_request_reading_tensors = {}
    now = time.time()

    assert not pw.delivery_active("r", now - 15.0)
    pw.delivery.touch("r")
    assert pw.delivery_active("r", now - 15.0)
    pw.delivery.last_touch["r"] = now - 30.0
    assert not pw.delivery_active("r", now - 15.0)
    pw.delivery.active = frozenset({"r"})
    assert pw.delivery_active("r", now - 15.0)
    assert not pw.delivery_active("other", now - 15.0)
    pw.delivery.active = frozenset()
    pw.delivery.forget("r")
    assert not pw.delivery_active("r", now - 15.0)

    # announced and the worker busy: held; announced and idle, or busy but
    # never announced: not held
    pw.per_request_reading_tensors["r"] = 1
    pw.delivery.busy = True
    assert pw.delivery_active("r", now - 15.0)
    assert not pw.delivery_active("other", now - 15.0)
    pw.delivery.busy = False
    assert not pw.delivery_active("r", now - 15.0)


class _ReadyTensorManager(_RecordingTensorManager):
    """Every started read is ready on the next poll, with a tiny tensor,
    except for the requests in ``never_ready`` (a producer that never
    delivers)."""

    def __init__(self, never_ready=()):
        super().__init__()
        self.ready = {}
        self.never_ready = set(never_ready)

    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        super().start_read_tensors(request_id, graph_edges, graph_walk)
        if request_id not in self.never_ready:
            self.ready.setdefault(request_id, []).extend(graph_edges)
        return []

    def get_ready_tensors(self, graph_walk=None):
        ready, self.ready = self.ready, {}
        return ready

    def get_tensor(self, request_id, uuid):
        return torch.zeros(1)

    def dereference(self, request_id, uuid, n=1):
        pass


class _SlowPostprocessModel:
    """postprocess blocks until released, like a cold video encode."""

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()

    def postprocess(self, tensor, modality, request_kwargs=None):
        self.entered.set()
        assert self.release.wait(timeout=10)
        return b"encoded"


def test_a_long_postprocess_claims_the_requests_of_its_pass_only():
    """The API server's backstop reads what the worker claimed: the request
    being postprocessed and the one queued behind it in the same pass are
    active for the whole postprocess, a request whose read never completes
    is held only while the worker is busy, a request nothing announced is
    not held at all, and nothing is claimed once the pass is over."""
    model = _SlowPostprocessModel()
    tm = _ReadyTensorManager(never_ready={"req-lost"})
    delivery = DeliveryProgress()
    pw = object.__new__(PreprocessWorker)
    pw.delivery = delivery
    # what the API server's own bookkeeping says is announced and unread
    pw.per_request_reading_tensors = {"req-lost": 1}
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(),
        result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(),
        profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(),
        stop_event=stop,
        communicator=_RecordingCommunicator(),
        tensor_manager=tm,
        model=model,
        delivery=delivery,
    )
    for rid in ("req-slow", "req-behind", "req-lost"):
        result = _result_tensors(rid)
        result.graph_edge.tensor_info.append(TensorPointerInfo(
            dims=[1], dtype="float32", nbytes=4, address=0, stride=[1],
            uuid=f"u-{rid}", source_session_id="s", source_entity="worker_0",
        ))
        worker.result_tensor_queue.put(result)

    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert model.entered.wait(timeout=5), "postprocess never started"
        # Every read started (and was touched) before the postprocess began.
        assert tm.read_started == ["req-slow", "req-behind", "req-lost"]
        started = delivery.last_touch["req-lost"]
        since = time.time()
        for _ in range(20):
            assert delivery.busy
            assert pw.delivery_active("req-slow", since)
            assert pw.delivery_active("req-behind", since)
            # announced, so the busy worker holds it
            assert pw.delivery_active("req-lost", since)
            # never announced: nothing holds it
            assert not pw.delivery_active("req-unannounced", since)
            time.sleep(0.005)
        assert pw.delivery_active("req-lost", started - 1.0)
        model.release.set()
        chunks = {}
        for _ in range(2):
            chunk = worker.out_queue.get(timeout=5)
            chunks[chunk.request_id] = chunk.data
        assert chunks == {"req-slow": b"encoded", "req-behind": b"encoded"}
        deadline = time.time() + 2.0
        while delivery.active and time.time() < deadline:
            time.sleep(0.005)
        assert delivery.active == frozenset()
        assert delivery.last_touch["req-slow"] >= since
        assert delivery.last_touch["req-behind"] >= since
        assert delivery.last_touch["req-lost"] == started
        deadline = time.time() + 2.0
        while delivery.busy and time.time() < deadline:
            time.sleep(0.005)
        # idle now: the announced read that never completes expires
        assert not delivery.busy
        assert not pw.delivery_active("req-lost", since)
    finally:
        model.release.set()
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_drained_completion_stays_successful():
    pw = _StubPreprocessWorker(pending=False, final=True)
    server = _api_server_stub(pw)
    req = _pending_request()
    server.pending_requests["r2"] = req
    server.recently_completed["r2"] = time.time()

    server._prune_recently_completed()

    assert req.event.is_set()
    assert req.error is None
    # Fully delivered: ACK immediately, no drain check on the happy path.
    assert pw.drained_flags == [True]


def test_client_gone_gates_the_reads_done_ack():
    """The client handler pops pending_requests on its own timeout, which can
    happen with chunks still in flight — that ACK must be gated too."""
    pw = _StubPreprocessWorker(pending=True, final=False)
    server = _api_server_stub(pw)
    server.recently_completed["r4"] = time.time()

    server._prune_recently_completed()

    assert pw.cleaned == ["r4"]
    assert pw.drained_flags == [False]


def test_stream_carries_error_as_final_chunk():
    pw = _StubPreprocessWorker()
    server = _api_server_stub(pw)
    req = _pending_request()
    req.chunks.append(ResultChunk(request_id="r3", modality="text", data=b"partial"))
    req.error = "result delivery timed out; response is incomplete"
    req.error_status = 500
    req.event.set()
    server.pending_requests["r3"] = req

    async def _collect():
        return [chunk async for chunk in server.iter_result_chunks("r3")]

    chunks = asyncio.run(_collect())
    assert [c.modality for c in chunks] == ["text", "error"]
    assert chunks[-1].metadata["status"] == 500
