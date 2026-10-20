"""A request_id reused while the engine still tears the older request down (#485).

A client can reuse its ``request_id`` as soon as the API server frees it (its
answer arrived, or it aborted), while the conductor and the data worker are
still tearing the older request down. Every message below the API server names
a request by its id, so the API server runs each submission under a key of its
own and the older request's teardown can never name the newer one.

An id whose request is still running is refused with a 409 and the running
request is untouched; once it finished, failed or was aborted the id is free,
including for a client that drops a stream and resubmits at once.

The data worker is the real ``PreprocessWorkerThread`` loop on its own thread,
the conductor is the hand-built one of ``test_teardown_barrier``, and the API
server is real down to its preprocess-worker facade. Messages are delivered in
the order the conductor and the data worker send them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from api_server_helpers import make_test_api_server
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_teardown_barrier import PREPROCESS, _conductor, _remove_targets, _request_data

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.api_server.entrypoint import request_id_in_use
from mstar.api_server.request_types import APIServerMessage, RequestFailed, ResultChunk
from mstar.utils.ipc_format import (
    ConductorMessageType,
    NewRequestConductor,
    ReadsDone,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)

TEXT = dict(input_modalities=["text"], output_modalities=["text"])


class _Model:
    def process_prompt(self, *args, **kwargs):
        return None

    def postprocess(self, output, modality, request_kwargs=None):
        return f"<{int(output[0])}>".encode()


class _Socket:
    """The data worker's socket to the conductor; each loop pass polls it once."""

    def __init__(self):
        self.inbox = queue.Queue()
        self.sent = []
        self.polls = 0
        self.cond = threading.Condition()

    def get_all_new_messages(self):
        msgs = []
        while not self.inbox.empty():
            msgs.append(self.inbox.get())
        with self.cond:
            self.polls += 1
            self.cond.notify_all()
        return msgs

    def send(self, entity, msg):
        self.sent.append((entity, msg))

    # the data worker's outbox encodes with this: identity, so sends arrive as objects
    codec = SimpleNamespace(encode=lambda msg: msg)

    def try_send_bytes(self, entity, msg):
        self.send(entity, msg)
        return True

    def deliver(self, message_type, body):
        self.inbox.put(WorkerMessage(message_type=message_type, body=body))

    def sent_of(self, message_type):
        return [m.body for e, m in self.sent if e == "conductor" and m.message_type == message_type]


class _TensorManager:
    def __init__(self):
        self.forced = []

    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass

    def get_ready_tensors(self):
        return {}

    def has_inflight_reads(self, request_id):
        return False

    def force_cleanup_request(self, request_id):
        self.forced.append(request_id)

    def ack_unread_tensors(self, request_id, graph_edges):
        pass


def _api_server(preprocess_worker):
    return make_test_api_server(preprocess_worker=preprocess_worker)


class _Rig:
    def __init__(self):
        self.socket = _Socket()
        self.tm = _TensorManager()
        self.stop = threading.Event()
        self.worker = PreprocessWorkerThread(
            in_queue=queue.Queue(),
            result_tensor_queue=queue.Queue(),
            out_queue=queue.Queue(),
            profile_queue=queue.Queue(),
            cleanup_request_queue=queue.Queue(),
            abort_request_queue=queue.Queue(),
            reads_done_queue=queue.Queue(),
            discard_tensor_queue=queue.Queue(),
            stop_event=self.stop,
            communicator=self.socket,
            tensor_manager=self.tm,
            model=_Model(),
        )
        facade = PreprocessWorker.__new__(PreprocessWorker)
        facade.request_input_queue = self.worker.in_queue
        facade.result_tensor_input_queue = self.worker.result_tensor_queue
        facade.abort_request_queue = self.worker.abort_request_queue
        facade.reads_done_queue = self.worker.reads_done_queue
        facade.cleanup_request_queue = self.worker.cleanup_request_queue
        facade.discard_tensor_queue = self.worker.discard_tensor_queue
        facade.output_queue = self.worker.out_queue
        facade.received_chunk_counts = {}
        facade.per_request_reading_tensors = {}
        self.data_worker = facade
        self.server = _api_server(facade)
        self.thread = threading.Thread(target=self.worker.run, daemon=True)
        self.thread.start()

    def settle(self):
        """Return once a whole loop pass has run after everything queued so far."""
        with self.socket.cond:
            target = self.socket.polls + 2
            assert self.socket.cond.wait_for(lambda: self.socket.polls >= target, timeout=10)

    def submit(self, text):
        """Submit under the client id ``X``; returns the NEW_REQUEST the
        conductor gets for it."""
        self.server.submit_request(text=text, request_id="X", streaming=False, **TEXT)
        self.settle()
        return self.socket.sent_of(ConductorMessageType.NEW_REQUEST)[-1]

    def output(self, request_id, value=7):
        """A worker's output for ``request_id``, as the message thread hands it
        over; returns the chunk the data worker emits, or None."""
        self.data_worker.new_result_token(request_id, value, None, "text_inputs", "text")
        try:
            return self.worker.out_queue.get(timeout=2)
        except queue.Empty:
            return None

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


@pytest.fixture
def rig():
    r = _Rig()
    yield r
    r.close()


def test_the_older_request_s_drain_does_not_free_the_reuse_on_the_conductor(rig):
    """The client aborts X and resubmits X at once. The conductor admits the
    newer request while the older one still drains; the older one's last
    READS_DONE must free only the older one."""
    c = _conductor({})
    admitted = []
    c._admit_request = lambda body: admitted.append(body.request_id)

    first = rig.submit("first")
    c._ingest_request(first)
    c.requests[first.request_id] = _request_data(workers=("w0",))  # its dispatch

    rig.server.abort_request("X")
    rig.settle()
    (abort,) = rig.socket.sent_of(ConductorMessageType.ABORT_REQUEST)
    c._abort_request(abort.request_id)
    for reads_done in rig.socket.sent_of(ConductorMessageType.READS_DONE):
        c._handle_reads_done(reads_done)
    assert first.request_id in c.draining

    second = rig.submit("second")
    c._ingest_request(second)
    assert admitted == [first.request_id, second.request_id]
    newer = _request_data(workers=("w1",))
    c.requests[second.request_id] = newer

    c._handle_reads_done(ReadsDone(request_id=first.request_id, entity_id="w0"))

    assert c.requests.get(second.request_id) is newer, (
        "the older request's teardown freed the newer request's state"
    )
    assert _remove_targets(c, second.request_id) == set()
    assert _remove_targets(c, first.request_id) == {"w0", PREPROCESS}


def test_the_older_request_s_drain_does_not_drop_the_reuse_s_outputs(rig):
    """The data worker is still draining the aborted request when the reuse's
    first output arrives."""
    rig.submit("first")
    rig.server.abort_request("X")
    rig.settle()
    second = rig.submit("second")

    chunk = rig.output(second.request_id)

    assert chunk is not None, "the newer request's output was dropped as draining"
    assert chunk.data == b"<7>"
    assert chunk.request_id in rig.server.pending_requests


def test_the_older_request_s_remove_does_not_unlink_the_reuse_s_inputs(rig):
    """The older request finished and its client read the answer; the reuse is
    preprocessed before the older request's REMOVE_REQUEST lands."""
    first = rig.submit("first")
    s = rig.server
    with s.request_lock:
        s.recently_completed[first.request_id] = time.time()
        s.pending_requests[first.request_id].event.set()
    assert asyncio.run(s.collect_results("X")) == []
    with s.request_lock:
        s._prune_recently_completed()
    rig.settle()

    second = rig.submit("second")
    rig.socket.deliver(WorkerMessageType.REMOVE_REQUEST, RemoveRequest(first.request_id))
    rig.settle()

    assert second.request_id not in rig.tm.forced, (
        "the older request's REMOVE_REQUEST dropped the inputs the newer one is using"
    )
    assert rig.tm.forced == [first.request_id]
    chunk = rig.output(second.request_id)
    assert chunk is not None and chunk.data == b"<7>"


def _received(rig, key):
    """A worker's chunk for ``key`` as the message thread hands it over; returns
    the chunks the API server reads back from the data worker."""
    rig.data_worker.new_result_token(key, 7, None, "text_inputs", "text")
    rig.settle()
    return rig.data_worker.get_result_chunks()


def _completed(s, key, output_chunk_counts):
    """The conductor's request_complete for ``key``, then one prune pass;
    returns whether the request was released."""
    with s.request_lock:
        s.recently_completed[key] = time.time()
        s.pending_requests[key].output_chunk_counts = output_chunk_counts
        s._prune_recently_completed()
        return s.pending_requests[key].event.is_set()


def test_a_reuse_does_not_inherit_the_older_request_s_chunks(rig):
    """X finished with two chunks reported and one received, so it is held open
    and its id stays in use. The client aborts it and resubmits X; the older
    request's last chunk lands after the reuse started and must not count
    toward the reuse's hold."""
    s = rig.server
    first = rig.submit("first").request_id
    assert len(_received(rig, first)) == 1
    assert not _completed(s, first, {"text_inputs": 2})
    with pytest.raises(HTTPException) as refused:
        s.submit_request(text="early", request_id="X", streaming=False, **TEXT)
    assert refused.value.status_code == 409

    s.abort_request("X")
    rig.settle()
    second = rig.submit("second").request_id
    assert _received(rig, first) == []

    assert not _completed(s, second, {"text_inputs": 1}), (
        "the reuse was released on the older request's chunk"
    )
    assert [c.request_id for c in _received(rig, second)] == [second]
    with s.request_lock:
        s._prune_recently_completed()
    assert s.pending_requests[second].event.is_set()


def test_the_older_request_s_release_does_not_release_the_reuse(rig):
    """X failed, which frees the id while the data worker still holds its
    state until the client lets go. The reuse is held for its second chunk when
    the older request is released; the reuse stays held until that chunk."""
    s = rig.server
    first = rig.submit("first").request_id
    with s.request_lock:
        s.pending_requests[first].event.set()
        s.recently_completed[first] = time.time()

    second = rig.submit("second").request_id
    assert len(_received(rig, second)) == 1
    assert not _completed(s, second, {"text_inputs": 2})

    with s.request_lock:
        s.pending_requests.pop(first)  # its client read the error
        s._prune_recently_completed()
    assert first not in s.recently_completed
    assert not s.pending_requests[second].event.is_set(), (
        "the older request's release released the reuse"
    )
    assert rig.data_worker.received_chunk_counts[second] == {"text_inputs": 1}

    assert len(_received(rig, second)) == 1
    with s.request_lock:
        s._prune_recently_completed()
    assert s.pending_requests[second].event.is_set()


def test_the_seed_still_derives_from_the_client_request_id(rig):
    from mstar.conductor.conductor import _req_id_to_seed, _request_seed, _seed_id

    first = rig.submit("first")
    rig.server.abort_request("X")
    second = rig.submit("second")

    assert first.request_id != second.request_id
    assert first.client_request_id == second.client_request_id == "X"
    assert _request_seed({}, _seed_id(second)) == _req_id_to_seed("X")


def test_an_empty_pinned_request_id_keeps_its_seed():
    from mstar.conductor.conductor import _req_id_to_seed, _seed_id

    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="", **TEXT)
    s.abort_request("")
    s.submit_request(text="second", request_id="", **TEXT)

    seeds = {
        _req_id_to_seed(_seed_id(NewRequestConductor(
            request_id=inp.request_id,
            initial_signals={},
            initial_input_modalities=["text"],
            initial_output_modalities=["text"],
            input_metadata={},
            model_kwargs={},
            client_request_id=inp.client_request_id,
        )))
        for inp in pw.inputs
    }
    assert seeds == {_req_id_to_seed("")}


def _run_message_thread(s, pw, chunks=(), messages=(), until=lambda: True):
    """Run the API server's message thread over ``chunks`` from the data worker
    and ``messages`` from the conductor until ``until()`` holds."""
    chunks, messages = list(chunks), list(messages)
    pw.get_result_chunks = lambda: [chunks.pop(0)] if chunks else []
    pw.get_profile_updates = lambda: []
    pw.finished_reading = lambda rid, drained: None
    s.communicator = SimpleNamespace(
        get_all_new_messages=lambda: [messages.pop(0)] if messages else [],
    )
    s.conductor_proc = None
    s._liveness_interval_s = 0.5
    s.running = True
    thread = threading.Thread(target=s._process_messages, daemon=True)
    thread.start()
    try:
        _wait(until, timeout=10)
    finally:
        s.running = False
        thread.join(timeout=10)


def test_chunks_reach_the_client_under_its_own_request_id():
    """The engine's chunks name the request by its key; what the stream and
    collect hand back names it by the client's id."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)
    key = pw.inputs[0].request_id
    _run_message_thread(
        s, pw, chunks=[ResultChunk(request_id=key, modality="text", data=b"hi")],
        until=lambda: s.pending_requests[key].chunks,
    )
    s.pending_requests[key].event.set()

    chunks = asyncio.run(s.collect_results("X"))
    assert [(c.request_id, c.data) for c in chunks] == [("X", b"hi")]


@pytest.mark.parametrize("source", ["data_worker", "engine"])
def test_an_error_that_names_the_key_reaches_the_client_under_its_own_id(source):
    """An error raised below the API server can format the key it knows the
    request by (the sampler's min_p refusal does); the client sees its own id."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)
    key = pw.inputs[0].request_id
    text = f"request {key}: min_p must be in [0, 1]"
    if source == "data_worker":
        feed = dict(chunks=[ResultChunk(
            request_id=key, modality="error", data=text.encode(), metadata={"status": 400},
        )])
    else:
        feed = dict(messages=[APIServerMessage(
            message_type="request_failed",
            body=RequestFailed(request_id=key, error_message=text, status=400),
        )])
    _run_message_thread(s, pw, **feed, until=s.pending_requests[key].event.is_set)

    with pytest.raises(HTTPException) as failed:
        asyncio.run(s.collect_results("X"))
    assert (failed.value.status_code, failed.value.detail) == (
        400, "request X: min_p must be in [0, 1]",
    )


# ── the API server alone ─────────────────────────────────────────────────────

class _PreprocessWorker:
    def __init__(self):
        self.inputs = []
        self.aborted = []

    def new_request(self, inp):
        self.inputs.append(inp)

    def abort_request(self, request_id):
        self.aborted.append(request_id)


def test_a_closing_stream_does_not_abort_a_retry_under_its_id():
    """The request fails; the stream frees it and yields the error, the client
    retries under the same id, then closes the stream at the error. The
    stream's cleanup must abort only its own request."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)
    first = pw.inputs[0].request_id

    async def run():
        stream = s.iter_result_chunks("X")
        with s.request_lock:
            req = s.pending_requests[first]
            req.error, req.error_status = "boom", 500
            req.event.set()
        assert (await stream.__anext__()).modality == "error"
        s.submit_request(text="retry", request_id="X", **TEXT)
        await stream.aclose()

    asyncio.run(run())

    assert pw.aborted == [], "the old stream aborted the retry"
    assert pw.inputs[1].request_id in s.pending_requests


def test_a_submission_under_a_running_id_is_refused_and_the_running_one_is_untouched():
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)
    (first,) = (inp.request_id for inp in pw.inputs)

    with pytest.raises(HTTPException) as refused:
        s.submit_request(text="second", request_id="X", streaming=False, **TEXT)

    assert (refused.value.status_code, refused.value.detail) == (409, request_id_in_use("X"))
    assert len(pw.inputs) == 1 and pw.aborted == []
    assert list(s.pending_requests) == [first] and s._live == {"X": first}
    _complete(s, first, b"first")
    assert [c.data for c in asyncio.run(s.collect_results("X"))] == [b"first"]

    s.submit_request(text="reuse", request_id="X", streaming=False, **TEXT)
    assert len(pw.inputs) == 2, "the id was not free once its request was read"


def test_a_request_id_no_message_could_carry_is_refused():
    """A JSON escape can give an id a lone surrogate, which the wire codec
    cannot encode: refused up front, before anything is sent under it."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    with pytest.raises(HTTPException) as refused:
        s.submit_request(text="hi", request_id=json.loads('"\\udc80"'), **TEXT)

    assert refused.value.status_code == 400 and pw.inputs == []


def test_an_id_is_free_once_its_request_finished_even_before_it_is_read():
    """The first request's answer has landed but its client has not read it;
    a second submission under the id runs, and each client reads its own."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)

    async def run():
        first_reader = s.iter_result_chunks("X")
        _complete(s, pw.inputs[0].request_id, b"first")
        s.submit_request(text="second", request_id="X", **TEXT)
        second_reader = s.iter_result_chunks("X")
        _complete(s, pw.inputs[1].request_id, b"second")
        return [c.data async for c in first_reader], [c.data async for c in second_reader]

    assert asyncio.run(run()) == ([b"first"], [b"second"])
    assert pw.aborted == [] and not s.pending_requests


def test_an_immediate_retry_waits_for_the_dropped_request_s_abort():
    """The retry arrives before the server has aborted the stream its client
    dropped: it waits for that abort instead of being refused."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)

    async def run():
        asyncio.get_running_loop().call_later(0.02, s.abort_request, "X")
        await s.wait_for_request_id("X")
        s.submit_request(text="retry", request_id="X", **TEXT)

    asyncio.run(run())
    assert [inp.text for inp in pw.inputs] == ["first", "retry"]
    assert pw.aborted == [pw.inputs[0].request_id]


def test_the_wait_for_a_running_id_is_bounded(monkeypatch):
    monkeypatch.setattr(entrypoint, "DUPLICATE_ID_GRACE_S", 0.05)
    s = _api_server(_PreprocessWorker())
    s.submit_request(text="first", request_id="X", **TEXT)

    t0 = time.monotonic()
    asyncio.run(s.wait_for_request_id("X"))

    assert 0.05 <= time.monotonic() - t0 < 1.0
    assert s._id_waiters == set()


def test_one_caller_at_a_time_waits_on_a_running_id(monkeypatch):
    """N duplicates of a running id cost one poll loop: while one caller
    waits, another returns at once (and submit_request then refuses it)."""
    monkeypatch.setattr(entrypoint, "DUPLICATE_ID_GRACE_S", 0.3)
    s = _api_server(_PreprocessWorker())
    s.submit_request(text="first", request_id="X", **TEXT)

    async def run():
        first = asyncio.create_task(s.wait_for_request_id("X"))
        await _await(lambda: "X" in s._id_waiters)
        t0 = time.monotonic()
        await s.wait_for_request_id("X")
        second_s = time.monotonic() - t0
        assert not first.done(), "the first caller stopped waiting"
        await first
        return second_s

    assert asyncio.run(run()) < 0.1
    assert s._id_waiters == set()


def test_a_cancelled_wait_frees_the_id_for_the_next_caller(monkeypatch):
    monkeypatch.setattr(entrypoint, "DUPLICATE_ID_GRACE_S", 5.0)
    s = _api_server(_PreprocessWorker())
    s.submit_request(text="first", request_id="X", **TEXT)

    async def run():
        waiting = asyncio.create_task(s.wait_for_request_id("X"))
        await _await(lambda: "X" in s._id_waiters)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

    asyncio.run(run())
    assert s._id_waiters == set()


# ── a duplicate of a running id, through the routes ─────────────────────────

def _complete(s, key, data):
    """The engine's whole answer for ``key`` lands."""
    with s.request_lock:
        req = s.pending_requests[key]
        req.chunks.append(ResultChunk(request_id="X", modality="text", data=data))
        req.event.set()
    req.notify()


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline
        time.sleep(0.005)


async def _await(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline
        await asyncio.sleep(0.005)


def _routes(monkeypatch, s, tmp_path=None):
    monkeypatch.setattr(entrypoint, "api_server", s)
    monkeypatch.setattr(entrypoint, "DUPLICATE_ID_GRACE_S", 0.05)
    s.upload_dir = tmp_path
    s.model_name = "bagel"
    s.model = None


@pytest.mark.parametrize("streaming", [False, True])
def test_a_duplicate_generate_gets_a_409_and_the_running_one_completes(monkeypatch, streaming):
    """Two /generate calls under X; the second arrives while the first runs and
    is refused with an HTTP 409 before any stream opens. The first is not
    aborted and gets its whole answer."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    _routes(monkeypatch, s)
    form = {"text": "first", "request_id": "X", "streaming": str(streaming).lower()}

    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/generate", data=form))
            await _await(lambda: len(pw.inputs) == 1)
            second = await client.post("/generate", data={**form, "text": "second"})
            _complete(s, pw.inputs[0].request_id, b"first")
            return await first, second

    r1, r2 = asyncio.run(run())

    assert r2.status_code == 409, r2.text
    assert r2.json()["detail"] == request_id_in_use("X")
    assert r1.status_code == 200, r1.text
    if streaming:
        lines = [json.loads(line) for line in r1.text.splitlines()]
        assert [(line["modality"], base64.b64decode(line["data"])) for line in lines] == [
            ("text", b"first"),
        ]
    else:
        assert [base64.b64decode(o["data"]) for o in r1.json()["outputs"]["text"]] == [b"first"]
    assert len(pw.inputs) == 1 and pw.aborted == []
    assert not s.pending_requests and s._live == {}


def test_a_duplicate_websocket_request_gets_an_in_band_409(monkeypatch, tmp_path, caplog):
    pw = _PreprocessWorker()
    s = _api_server(pw)
    _routes(monkeypatch, s, tmp_path)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "first", "request_id": "X"}))
        _wait(lambda: len(pw.inputs) == 1)
        ws.send_text(json.dumps({"text": "second", "request_id": "X"}))

        reply = json.loads(ws.receive_text())
        assert reply == {"request_id": "X", "error": request_id_in_use("X"), "status": 409}

        _complete(s, pw.inputs[0].request_id, b"first")
        chunk, finish = json.loads(ws.receive_text()), json.loads(ws.receive_text())
    assert chunk["request_id"] == "X" and base64.b64decode(chunk["data"]) == b"first"
    assert finish == {"request_id": "X", "finish": True}
    assert len(pw.inputs) == 1 and pw.aborted == []
    assert not [r for r in caplog.records if r.exc_info], "an expected 409 logged a traceback"


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"model": "bagel", "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/videos/generations", {"model": "wan22", "prompt": "a cat"}),
], ids=["chat", "videos"])
@pytest.mark.parametrize("stream", [False, True])
def test_an_openai_route_answers_a_refused_id_with_a_409(monkeypatch, tmp_path, path, body, stream):
    """The OpenAI routes mint their own ids, so a client cannot pin one; if a
    minted id were still in flight, the route answers the 409 as its status."""
    from mstar.api_server.openai import serving_chat, serving_videos

    pw = _PreprocessWorker()
    s = _api_server(pw)
    _routes(monkeypatch, s, tmp_path)
    s.model_name = body["model"]
    for module in (serving_chat, serving_videos):
        monkeypatch.setattr(module, "rid", lambda prefix: "X")
    s.submit_request(text="first", request_id="X", **TEXT)
    body = {**body, "stream": True} if path.endswith("completions") and stream else body
    if path.endswith("generations") and stream:
        body = {**body, "stream_video": True}

    r = TestClient(entrypoint.app).post(path, json=body)

    assert r.status_code == 409, r.text
    assert r.json()["error"]["message"] == request_id_in_use("X")
    assert len(pw.inputs) == 1 and pw.aborted == []


def test_a_stream_is_bound_to_its_submission_before_it_is_iterated():
    """/generate hands the body to Starlette, which iterates it later; a
    submission under the same id in between (the first one has finished, so
    the id is free) is not this stream's."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)

    async def run():
        body = s.async_stream_results("X")
        _complete(s, pw.inputs[0].request_id, b"first")
        s.submit_request(text="second", request_id="X", **TEXT)
        _complete(s, pw.inputs[1].request_id, b"second")
        return [json.loads(line) async for line in body]

    lines = asyncio.run(run())

    assert [base64.b64decode(line["data"]) for line in lines] == [b"first"], (
        "the stream read the newer request")
    assert [c.data for c in s.pending_requests[pw.inputs[1].request_id].chunks] == [b"second"]


def test_an_openai_chat_stream_is_bound_before_it_is_iterated():
    """The chat stream is bound when it is made: Starlette first iterates it
    after the id is reused, and it still reads its own request's result."""
    from mstar.api_server.openai.serving_chat import _stream

    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)

    async def run():
        body = _stream(s, "m", "X", 24000)
        _complete(s, pw.inputs[0].request_id, b"first")
        s.submit_request(text="second", request_id="X", **TEXT)
        _complete(s, pw.inputs[1].request_id, b"second")
        return [frame async for frame in body]

    frames = asyncio.run(run())

    events = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: {")]
    contents = [e["choices"][0]["delta"].get("content") for e in events if e.get("choices")]
    assert "first" in contents and "second" not in contents, events
    assert [c.data for c in s.pending_requests[pw.inputs[1].request_id].chunks] == [b"second"]


def _bridge(s):
    from mstar.api_server.rust_frontend import RustFrontendBridge

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = s
    bridge._keys = {}
    sent = []
    bridge._send = sent.append
    return bridge, sent


def test_rust_bridge_relays_a_refused_submit_under_its_own_sub():
    """Two submits under X in one receive batch while the first runs: the
    second is refused and its 409 goes back in-band under its own sub (the
    form that carries a status); the first is untouched and answers."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    bridge, sent = _bridge(s)
    submit = {"t": "submit", "rid": "X", "sub": 1, "text": "first", **TEXT}

    async def run():
        bridge._submit(submit)
        bridge._submit({**submit, "sub": 2, "text": "second"})
        _complete(s, pw.inputs[0].request_id, b"first")
        await _await(lambda: [m["t"] for m in sent].count("done") == 1)

    asyncio.run(run())

    replies = [
        (m["sub"], m["t"], m.get("modality"), m.get("data"), (m.get("metadata") or {}).get("status"))
        for m in sent
    ]
    assert sorted(replies, key=str) == sorted([
        (2, "chunk", "error", request_id_in_use("X").encode(), 409),
        (1, "chunk", "text", b"first", None),
        (1, "done", None, None, None),
    ], key=str)
    assert len(pw.inputs) == 1 and pw.aborted == []
    assert bridge._keys == {}


def test_rust_bridge_abort_of_a_dropped_stream_spares_the_retry():
    """The client drops submission 1 of X and resubmits X as submission 2. The
    frontend sends 1's abort before it frees X, so the abort lands first; it
    reaches 1 only, the retry is not refused, and nothing of 1 is sent under
    2's sub."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    bridge, sent = _bridge(s)
    submit = {"t": "submit", "rid": "X", "sub": 1, "text": "first", **TEXT}
    abort = {"t": "abort", "rid": "X", "sub": 1}
    retry = {**submit, "sub": 2, "text": "retry"}

    async def run():
        bridge._submit(submit)
        await asyncio.sleep(0)
        bridge._abort(abort)
        bridge._submit(retry)
        await _await(lambda: [m["sub"] for m in sent if m["t"] == "done"] == [1])
        _complete(s, pw.inputs[1].request_id, b"retry")
        await _await(lambda: [m["sub"] for m in sent if m["t"] == "done"] == [1, 2])

    asyncio.run(run())

    first, second = (inp.request_id for inp in pw.inputs)
    assert pw.aborted == [first], "the dropped stream's abort reached the retry"
    assert [(m["t"], m.get("modality"), m.get("data")) for m in sent if m["sub"] == 2] == [
        ("chunk", "text", b"retry"),
        ("done", None, None),
    ]
    assert bridge._keys == {}
    assert second not in s.pending_requests and s._live == {}


def test_nothing_is_kept_once_the_request_is_done():
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.preprocess_worker.finished_reading = lambda rid, drained=True: None
    s.submit_request(text="first", request_id="X", streaming=False, **TEXT)
    key = pw.inputs[0].request_id
    with s.request_lock:
        s.recently_completed[key] = time.time()
        s.pending_requests[key].event.set()

    asyncio.run(s.collect_results("X"))
    assert s._live == {"X": key}  # its outputs are still being released
    with s.request_lock:
        s._prune_recently_completed()

    assert s._live == {} and s._client_ids == {}
    assert not s.pending_requests and not s.recently_completed


# ── a served retry right after a dropped stream ─────────────────────────────

class _StreamingWorker(_PreprocessWorker):
    """Each submission streams ``tokens`` text chunks, a few ms apart, until
    it is aborted, then finishes."""

    def __init__(self, tokens):
        super().__init__()
        self.server = None
        self.tokens = tokens

    def new_request(self, inp):
        super().new_request(inp)
        threading.Thread(target=self._run, args=(inp.request_id,), daemon=True).start()

    def _run(self, key):
        s = self.server
        for i in range(self.tokens + 1):
            time.sleep(0.003)
            with s.request_lock:
                req = s.pending_requests.get(key)
                if req is None or key in self.aborted:
                    return
                if i < self.tokens:
                    req.chunks.append(ResultChunk(request_id="X", modality="text", data=b"t"))
                else:
                    req.event.set()
            req.notify()

    def shutdown(self):
        pass


def test_a_retry_right_after_a_dropped_stream_is_served(monkeypatch):
    """The client reads 3 chunks of a /generate stream under X from a real
    server, drops the connection and resubmits X at once on a connection it
    opened earlier, before the server has seen the drop. The retry must get
    its whole answer rather than a 409, and so must the next reuse of X once
    that answer is read, round after round."""
    import socket

    import uvicorn

    pw = _StreamingWorker(tokens=60)
    s = _api_server(pw)
    pw.server = s
    monkeypatch.setattr(entrypoint, "api_server", s)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(entrypoint.app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait(lambda: server.started, timeout=15)
    url = f"http://127.0.0.1:{port}/generate"
    form = {"text": "count", "request_id": "X", "streaming": "true"}

    def answer(client):
        with client.stream("POST", url, data=form) as r:
            if r.status_code != 200:
                return f"{r.status_code} {r.read()[:120]!r}"
            lines = [json.loads(line) for line in r.iter_lines() if line.strip()]
        return [line["modality"] for line in lines]

    try:
        with httpx.Client(timeout=10) as warm:
            for _ in range(5):
                warm.get(f"http://127.0.0.1:{port}/health")
                with httpx.Client(timeout=10) as dropped, dropped.stream("POST", url, data=form) as r:
                    lines = r.iter_lines()
                    for _ in range(3):
                        assert next(lines)
                for request in ("retry", "reuse"):
                    got = answer(warm)
                    assert got == ["text"] * 60, f"the {request} got {got}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert len(pw.inputs) == 15
    assert pw.aborted == [inp.request_id for inp in pw.inputs[::3]], "a retry or reuse was cut short"
