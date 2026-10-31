"""API-server result delivery under load.

Covers the completion-vs-delivery race: a finished request's chunks flow
api-side through the data worker, whose single thread also runs media
preprocessing. The worker must drain queued result reads ahead of (multi-
second) preprocess items, and the API server's post-completion TTL must fail
a request whose chunks never arrived instead of closing it as an empty
success.
"""

import asyncio
import gc
import queue
import threading
import time
import weakref
from types import SimpleNamespace

import pytest
from api_server_helpers import StandInSocket, make_test_api_server
from fastapi import HTTPException

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import PreprocessWorkerThread, ResultQueue
from mstar.api_server.entrypoint import PendingRequest
from mstar.api_server.request_types import PreprocessInput, ResultChunk, ResultTensors
from mstar.communication.event import EventWakeup
from mstar.graph.base import GraphEdge
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.profile.format import OutputInfo, RequestProfile


class _RecordingTensorManager:
    def __init__(self):
        self.read_started = []

    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        self.read_started.append(request_id)
        return []

    def take_failed_reads(self):

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

    def get_partition_topology(self):
        from mstar.streaming.topology import PartitionTopology
        return PartitionTopology(partitions=["default"])

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
        out_queue=ResultQueue(EventWakeup()),
        profile_queue=ResultQueue(EventWakeup()),
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
        self.results_wakeup = EventWakeup()
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


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("batched", [True, False])
def test_consumed_chunks_are_released_before_stream_completion(streaming, batched):
    """Even the Rust relay's non-streaming submissions drain through the iterator."""
    server = make_test_api_server(preprocess_worker=SimpleNamespace(abort_request=lambda rid: None))
    req = _pending_request()
    req.streaming = streaming
    _register(server, "retention", req)

    async def consume():
        stream = (
            server.iter_result_chunk_batches("retention") if batched
            else server.iter_result_chunks("retention")
        )
        refs = []
        try:
            for i in range(128):
                with server.request_lock:
                    req.chunks.append(ResultChunk(
                        request_id="retention", modality="video_frame", data=bytes([i]) * (64 * 1024),
                    ))
                    refs.append(weakref.ref(req.chunks[-1]))
                    req.notify()
                result = await anext(stream)
                if batched:
                    assert len(result) == 1
                    assert result[0].data == bytes([i]) * (64 * 1024)
                else:
                    assert result.data == bytes([i]) * (64 * 1024)
                del result

            assert "retention" in server.pending_requests
            assert not req.event.is_set()
            assert len(req.chunks) == 0
            gc.collect()
            # The suspended sender may still own its current chunk, but no history.
            assert all(ref() is None for ref in refs[:-1])
            with server.request_lock:
                req.event.set()
                req.notify()
            assert [result async for result in stream] == []
        finally:
            await stream.aclose()
        gc.collect()
        assert all(ref() is None for ref in refs)

    asyncio.run(consume())
    assert "retention" not in server.pending_requests


def test_final_flush_drains_chunks_appended_while_yielding_a_completed_batch():
    server = make_test_api_server(preprocess_worker=SimpleNamespace(abort_request=lambda rid: None))
    req = _pending_request()
    req.chunks.append(ResultChunk(request_id="tail", modality="text", data=b"first"))
    req.event.set()
    _register(server, "tail", req)

    async def consume():
        stream = server.iter_result_chunk_batches("tail")
        try:
            assert [c.data for c in await anext(stream)] == [b"first"]
            with server.request_lock:
                req.chunks.append(ResultChunk(request_id="tail", modality="text", data=b"last"))
            assert [[c.data for c in batch] async for batch in stream] == [[b"last"]]
        finally:
            await stream.aclose()

    asyncio.run(consume())
    assert req.chunks == []
    assert "tail" not in server.pending_requests


def _deliver_result_chunks(server, *chunks):
    """Drive one real message-loop pass with outputs from a fake data worker."""
    def stop_after_this_pass():
        server.running = False
        return []

    server.communicator = StandInSocket(stop_after_this_pass)
    server.preprocess_worker = SimpleNamespace(
        get_profile_updates=lambda: [], get_result_chunks=lambda: chunks,
        abort_request=lambda rid: None, results_wakeup=EventWakeup(),
    )
    server.running = True
    server._process_messages()


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("log_stats", [True, False])
def test_output_totals_survive_draining_and_native_collection(monkeypatch, streaming, log_stats):
    monkeypatch.setattr(entrypoint, "pretty_print_profile", lambda *args: None)
    server = make_test_api_server(log_stats=log_stats)
    req = PendingRequest(
        streaming=streaming, input_modalities=["text"], output_modalities=["text", "audio"],
        profile=RequestProfile(rid="profile"),
    )
    _register(server, "profile", req)

    async def consume():
        stream = server.iter_result_chunk_batches("profile") if streaming else None
        delivered = []
        for batch in (
            [("text", b"a"), ("audio", b"1234")],
            [("text", b"bc")],
        ):
            _deliver_result_chunks(server, *[
                ResultChunk(request_id="profile", modality=modality, data=data) for modality, data in batch
            ])
            if stream is not None:
                delivered.extend(c.data for c in await anext(stream))
                assert len(req.chunks) == 0
        req.event.set()
        if stream is not None:
            assert [batch async for batch in stream] == []
        else:
            delivered = [c.data for c in await server.collect_results("profile")]
        return delivered

    assert asyncio.run(consume()) == [b"a", b"1234", b"bc"]
    assert "profile" not in server.pending_requests
    assert req.profile.outputs == (
        [OutputInfo(modality="text", count=2, total_bytes=3), OutputInfo(modality="audio", count=1, total_bytes=4)]
        if log_stats else []
    )


def test_the_profile_timeline_keeps_the_stage_labels_its_readers_parse(tmp_path):
    """The worker's span sits beside the checkpoint chain, not in it: the wan22
    bench reads ``conductor ingest → first chunk`` as its compute span, and a
    streamed first chunk can land before the worker's last step."""
    from benchmark.request_timeline.report import parse
    from benchmark.wan22.profile_log import parse_profile_file
    from mstar.profile.display import pretty_print_profile
    from mstar.profile.format import GraphTiming, RequestTiming

    prof = RequestProfile(rid="r", timing=RequestTiming(
        recv_time=0.0, preprocess_finish_time=0.01, conductor_ingest_time=0.02,
        first_chunk_time=0.03, last_output_read_time=0.05, last_chunk_time=0.06,
        conductor_finish_time=0.07, finish_time=0.08, output_postprocess_s=0.002,
    ), graph_timings=[GraphTiming(
        "dit", "denoise", 2, 0.04, 0.03, 0.005, 0.005, first_start=0.021, last_end=0.055,
    )])
    path = tmp_path / "stats.txt"
    pretty_print_profile(prof, str(path))

    (wan,) = parse_profile_file(str(path))
    assert wan.timeline_seg("conductor ingest -> first chunk") == pytest.approx(10.0)
    (timeline,) = parse(path.read_text())
    assert list(timeline.offsets) == [
        "recv", "preprocess done", "conductor ingest", "first chunk", "last chunk", "conductor done", "finish",
    ]
    assert timeline.worker_span_ms == pytest.approx(34.0)


def test_profile_counts_accepted_errors_and_outputs_after_the_error(monkeypatch):
    """Profiling counts accepted chunks, even when terminal filtering hides a tail."""
    monkeypatch.setattr(entrypoint, "pretty_print_profile", lambda *args: None)
    server = make_test_api_server(log_stats=True)
    req = PendingRequest(
        streaming=True, input_modalities=["text"], output_modalities=["text"],
        profile=RequestProfile(rid="error"), request_id="public",
    )
    _register(server, "error", req)
    _deliver_result_chunks(
        server,
        ResultChunk(request_id="error", modality="text", data=b"ok"),
        ResultChunk(request_id="error", modality="error", data=b"error failed"),
        ResultChunk(request_id="error", modality="text", data=b"late"),
    )

    async def consume():
        return [(c.modality, c.data) async for c in server.iter_result_chunks("error")]

    assert asyncio.run(consume()) == [("text", b"ok"), ("error", b"public failed")]
    assert req.chunks == []
    assert req.profile.outputs == [
        OutputInfo(modality="text", count=2, total_bytes=6),
        OutputInfo(modality="error", count=1, total_bytes=len(b"public failed")),
    ]
