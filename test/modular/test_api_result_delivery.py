"""API-server result delivery under load.

Covers the completion-vs-delivery race: a finished request's chunks flow
api-side through the data worker, whose single thread also runs media
preprocessing. The worker must drain queued result reads ahead of (multi-
second) preprocess items, and the API server's post-completion TTL must fail
a request whose chunks never arrived instead of closing it as an empty
success.
"""

import asyncio
import queue
import threading
import time

import pytest
from api_server_helpers import make_test_api_server
from fastapi import HTTPException

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.entrypoint import PendingRequest
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

    def force_cleanup_request(self, request_id):
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
    def __init__(self, pending=False, final=True):
        self.pending = pending
        self.final = final
        self.cleaned = []
        self.drained_flags = []
        self.aborted = []

    def abort_request(self, rid):
        self.aborted.append(rid)

    def has_pending_tensors(self, rid):
        return self.pending

    def received_final_chunks(self, rid, output_chunk_counts):
        return self.final

    def finished_reading(self, rid, drained=True):
        # New teardown: the API server signals READS_DONE to the conductor
        # instead of cleaning up locally; the conductor drives the hard cleanup.
        self.cleaned.append(rid)
        self.drained_flags.append(drained)


def _pending_request(streaming=True):
    return PendingRequest(
        streaming=streaming,
        input_modalities=["text"],
        output_modalities=["text"],
        profile=None,
    )


def _api_server_stub(preprocess_worker):
    return make_test_api_server(preprocess_worker=preprocess_worker)


def _register(server, rid, req):
    """What ``submit_request`` registers, with the request's key equal to its id."""
    server.pending_requests[rid] = req
    server._live[rid] = rid
    server._client_ids[rid] = rid


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
    _register(server, "r3", req)

    async def _collect():
        return [chunk async for chunk in server.iter_result_chunks("r3")]

    chunks = asyncio.run(_collect())
    assert [c.modality for c in chunks] == ["text", "error"]
    assert chunks[-1].metadata["status"] == 500


def test_a_chunk_from_another_thread_wakes_the_stream_at_once():
    """The handler sleeps on an event the message thread sets, so a chunk is
    delivered without waiting out a poll interval (the backstop is 50 ms)."""
    pw = _StubPreprocessWorker()
    server = _api_server_stub(pw)
    req = _pending_request()
    _register(server, "r4", req)

    async def _collect():
        loop = asyncio.get_running_loop()
        times = []

        def producer():
            time.sleep(0.2)
            for i in range(3):
                time.sleep(0.15)
                with server.request_lock:
                    req.chunks.append(ResultChunk(request_id="r4", modality="text", data=f"c{i}".encode()))
                    req.notify()
                times.append(time.perf_counter())
            time.sleep(0.15)
            with server.request_lock:
                req.event.set()
                req.notify()

        threading.Thread(target=producer, daemon=True).start()
        got = []
        async for chunk in server.iter_result_chunks("r4"):
            got.append((chunk.data, time.perf_counter()))
        del loop
        return got, times

    got, times = asyncio.run(_collect())
    assert [d for d, _ in got] == [b"c0", b"c1", b"c2"]
    # each chunk reached the handler within the 50 ms backstop, not a poll later
    for (_, t_got), t_put in zip(got, times, strict=True):
        assert t_got - t_put < 0.05, f"chunk took {1000 * (t_got - t_put):.1f} ms"


class _Client:
    """The ``Request`` that ``collect_results`` asks whether the client left;
    counts the asks."""

    def __init__(self, gone=False):
        self.gone = gone
        self.checks = 0

    async def is_disconnected(self):
        self.checks += 1
        return self.gone


def test_a_finish_from_another_thread_wakes_collect_results_at_once():
    """A non-streaming handler sleeps on the same wake as a streaming one: the
    finish reaches it at once, and while it waits the client's connection is
    checked on the 100 ms backstop, not every few ms (polling each waiting
    request every 5 ms held the event loop at a few hundred in flight)."""
    pw = _StubPreprocessWorker()
    server = _api_server_stub(pw)
    req = _pending_request(streaming=False)
    _register(server, "r7", req)
    client = _Client()

    async def _collect():
        finished = []

        def producer():
            time.sleep(0.5)
            with server.request_lock:
                req.chunks.append(ResultChunk(request_id="r7", modality="text", data=b"done"))
                req.event.set()
                finished.append(time.perf_counter())
                req.notify()

        start = time.perf_counter()
        threading.Thread(target=producer, daemon=True).start()
        chunks = await server.collect_results("r7", client)
        end = time.perf_counter()
        return chunks, end - finished[0], end - start

    chunks, lag, waited = asyncio.run(_collect())
    assert [c.data for c in chunks] == [b"done"]
    assert lag < 0.05, f"the finish took {1000 * lag:.1f} ms to reach the handler"
    assert client.checks <= waited / 0.1 + 1, f"{client.checks} disconnect checks in {waited:.2f} s"
    assert "r7" not in server.pending_requests


def _feed_tokens(server, req, rid, every=0.02, until=3.0):
    """The message thread's per-token path: a chunk and a ``notify_many`` every
    ``every`` s, until the request leaves ``pending_requests`` (or ``until``)."""
    from mstar.api_server.entrypoint import notify_many

    def run():
        end = time.time() + until
        while time.time() < end and rid in server.pending_requests:
            with server.request_lock:
                req.chunks.append(ResultChunk(request_id=rid, modality="text", data=b"t"))
            notify_many([req])
            time.sleep(every)

    threading.Thread(target=run, daemon=True).start()


@pytest.mark.parametrize("generating", [False, True], ids=["idle", "generating"])
def test_collect_results_still_releases_a_request_whose_client_left(generating):
    """Also while the request keeps producing tokens: a wake per token must
    not keep the handler from asking whether the client left."""
    pw = _StubPreprocessWorker()
    server = _api_server_stub(pw)
    req = _pending_request(streaming=False)
    _register(server, "r8", req)
    if generating:
        _feed_tokens(server, req, "r8")

    start = time.perf_counter()
    assert asyncio.run(server.collect_results("r8", _Client(gone=True))) == []
    assert time.perf_counter() - start < 1.0
    assert pw.aborted == ["r8"]
    assert "r8" not in server.pending_requests


@pytest.mark.parametrize("generating", [False, True], ids=["idle", "generating"])
def test_collect_results_still_times_out(generating):
    pw = _StubPreprocessWorker()
    server = _api_server_stub(pw)
    server.timeout_seconds = 0.3
    req = _pending_request(streaming=False)
    _register(server, "r9", req)
    if generating:
        _feed_tokens(server, req, "r9")

    start = time.perf_counter()
    with pytest.raises(HTTPException) as e:
        asyncio.run(server.collect_results("r9", _Client()))
    assert time.perf_counter() - start < 1.5
    assert e.value.status_code == 500
    assert pw.aborted == ["r9"]
    assert "r9" not in server.pending_requests


def test_tokens_do_not_wake_a_handler_waiting_for_the_finish():
    """``collect_results`` reads no chunk before the finish, so a chunk's
    notification leaves its wake alone; the finish sets it."""
    from mstar.api_server.entrypoint import notify_many

    class _Loop:
        def call_soon_threadsafe(self, fn, *args):
            fn(*args)

    req = _pending_request(streaming=False)
    req.loop, req.wake, req.wake_on_finish = _Loop(), asyncio.Event(), True
    notify_many([req])
    req.notify()
    assert not req.wake.is_set()
    req.event.set()
    notify_many([req])
    assert req.wake.is_set()
def test_the_message_thread_polls_on_the_socket_and_the_data_worker():
    """Between passes the message thread waits at most 1 ms, ended early by
    the communicator (a message, or the data worker's chunk signal)."""
    from mstar.communication.event import EventWakeup

    class _Comm:
        def __init__(self):
            self.registered, self.polls = [], []

        def register_event_for_poll(self, event):
            self.registered.append(event)

        def poll_for_messages(self, timeout_ms):
            self.polls.append(timeout_ms)

    pw = _StubPreprocessWorker()
    pw.results_wakeup = EventWakeup()
    server = _api_server_stub(pw)
    server.communicator = _Comm()
    server._pass_wait()()
    assert server.communicator.registered == [pw.results_wakeup]
    assert server.communicator.polls == [1]
    # without a poll to block on, the 1 ms sleep stays
    server.communicator = object()
    t0 = time.perf_counter()
    server._pass_wait()()
    assert time.perf_counter() - t0 >= 0.001


def test_notify_many_wakes_every_handler_with_one_loop_call():
    """A pass that delivered chunks to many requests wakes their handlers
    through one call into the loop, and skips requests with no handler."""
    from mstar.api_server.entrypoint import notify_many

    class _Loop:
        def __init__(self):
            self.calls = []

        def call_soon_threadsafe(self, fn, *args):
            self.calls.append((fn, args))
            fn(*args)

    class _Closed:
        def call_soon_threadsafe(self, fn, *args):
            raise RuntimeError("Event loop is closed")

    loop = _Loop()
    reqs = [_pending_request() for _ in range(4)]
    for req in reqs[:3]:
        req.loop, req.wake = loop, asyncio.Event()
    reqs[3].loop, reqs[3].wake = _Closed(), asyncio.Event()
    orphan = _pending_request()  # no handler attached yet
    notify_many(reqs + [orphan])
    assert len(loop.calls) == 1
    assert all(req.wake.is_set() for req in reqs[:3])
    assert not reqs[3].wake.is_set()


def test_chunks_waiting_together_come_as_one_batch():
    """Everything the message thread appended since the last wake is drained
    as one list, so a streaming handler can send it as one body frame; a
    chunk that lands later comes on its own, and the finish ends the stream
    without an empty batch."""
    pw = _StubPreprocessWorker()
    server = _api_server_stub(pw)
    req = _pending_request()
    for i in range(3):
        req.chunks.append(ResultChunk(request_id="r5", modality="text", data=f"t{i}".encode()))
    _register(server, "r5", req)

    async def _collect():
        batches = server.iter_result_chunk_batches("r5")
        got = [await batches.__anext__()]
        with server.request_lock:
            req.chunks.append(ResultChunk(request_id="r5", modality="text", data=b"t3"))
            req.notify()
        got.append(await batches.__anext__())
        with server.request_lock:
            req.event.set()
            req.notify()
        async for batch in batches:
            got.append(batch)
        return got

    got = asyncio.run(_collect())
    assert [[c.data for c in b] for b in got] == [[b"t0", b"t1", b"t2"], [b"t3"]]
    assert "r5" not in server.pending_requests


def test_error_tail_is_its_own_last_batch():
    pw = _StubPreprocessWorker()
    server = _api_server_stub(pw)
    req = _pending_request()
    req.chunks.append(ResultChunk(request_id="r6", modality="text", data=b"partial"))
    req.error = "result delivery timed out; response is incomplete"
    req.error_status = 500
    req.event.set()
    _register(server, "r6", req)

    async def _collect():
        return [[c.modality for c in b] async for b in server.iter_result_chunk_batches("r6")]

    assert asyncio.run(_collect()) == [["text"], ["error"]]
