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
from dataclasses import dataclass, field

import pytest

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.entrypoint import APIServer, PendingRequest
from mstar.api_server.request_types import PreprocessInput, ResultChunk, ResultTensors
from mstar.graph.base import GraphEdge
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
    def __init__(self, pending=False, final=True):
        self.pending = pending
        self.final = final
        self.cleaned = []

    def has_pending_tensors(self, rid):
        return self.pending

    def received_final_chunks(self, rid, final_outputs):
        return self.final

    def cleanup_request(self, rid):
        self.cleaned.append(rid)


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


def test_drained_completion_stays_successful():
    pw = _StubPreprocessWorker(pending=False, final=True)
    server = _api_server_stub(pw)
    req = _pending_request()
    server.pending_requests["r2"] = req
    server.recently_completed["r2"] = time.time()

    server._prune_recently_completed()

    assert req.event.is_set()
    assert req.error is None


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
