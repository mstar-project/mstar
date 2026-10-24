"""An output failure in the data worker: one teardown and one error.

A failure turning a request's output into a chunk (starting the read,
postprocessing a result tensor or an inline token) sends the client its
error. The conductor still holds the request and would run it to the end, so
the data worker also stops it, once, the way a client's disconnect does:
ABORT_REQUEST to the conductor and its own drain. The conductor tears an
aborted request down without a ``request_failed``, so the data worker's error
stays the client's only answer, and every stream ends at that error.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_request_id_reuse import TEXT, _api_server, _PreprocessWorker, _routes, _wait
from test_teardown_barrier import PREPROCESS, _conductor, _remove_targets, _request_data

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import InlineToken, PreprocessWorkerThread, RequestOutputState
from mstar.api_server.request_types import PreprocessInput, ResultChunk, ResultTensors
from mstar.utils.ipc_format import ConductorMessageType, FailRequests, ReadsDone

# ── the data worker ─────────────────────────────────────────────────────────


class _Socket:
    def __init__(self, stop=None):
        self.sent = []
        self.stop = stop

    def get_all_new_messages(self):
        if self.stop is not None:
            self.stop.set()  # the loop runs this one pass to its end
        return []

    def send(self, entity, msg):
        self.sent.append((entity, msg))

    def to_conductor(self):
        return [(m.message_type, m.body.request_id) for e, m in self.sent if e == "conductor"]


class _TensorManager:
    def __init__(self, ready=None, read_raises=False):
        self.ready = ready or {}
        self.read_raises = read_raises

    def start_read_tensors(self, request_id, graph_edges):
        if self.read_raises:
            raise RuntimeError("arena full")
        return []

    def get_ready_tensors(self):
        ready, self.ready = self.ready, {}
        return ready

    def get_tensor(self, uuid):
        return object()

    def dereference(self, uuid):
        pass

    def has_inflight_reads(self, request_id):
        return False

    def force_cleanup_request(self, request_id):
        pass


class _Model:
    def __init__(self, postprocess_raises=False, prompt_raises=False):
        self.postprocess_raises = postprocess_raises
        self.prompt_raises = prompt_raises

    def process_prompt(self, *args, **kwargs):
        if self.prompt_raises:
            raise ValueError("bad prompt")

    def postprocess(self, output, modality, request_kwargs=None):
        if self.postprocess_raises:
            raise RuntimeError("decode failed")
        return b"ok"


def _thread(model, tensor_manager):
    stop = threading.Event()
    wt = PreprocessWorkerThread(
        in_queue=queue.Queue(),
        result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(),
        profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(),
        stop_event=stop,
        communicator=_Socket(stop),
        tensor_manager=tensor_manager,
        model=model,
    )
    # as _process_input does
    wt.in_flight_requests.add("r1")
    wt._live_requests["r1"] = 1
    return wt


def _edge(*uuids):
    return SimpleNamespace(name="text_output", tensor_info=[SimpleNamespace(uuid=u) for u in uuids])


def _errors(wt):
    chunks = []
    while not wt.out_queue.empty():
        chunks.append(wt.out_queue.get_nowait())
    return [c for c in chunks if c.modality == "error" and c.request_id == "r1"]


def _with_output(stage):
    """A data worker whose next loop pass fails r1's output at ``stage``."""
    if stage == "read start":
        wt = _thread(_Model(), _TensorManager(read_raises=True))
        wt.result_tensor_queue.put(ResultTensors(
            request_id="r1", modality="text", graph_edge=_edge("u1", "u2"), loop_indices=None,
        ))
    elif stage == "result tensor":
        wt = _thread(_Model(postprocess_raises=True), _TensorManager(ready={"r1": [_edge("u1", "u2")]}))
        wt.request_output_state["r1"] = RequestOutputState(
            order={"u1": (0, None), "u2": (1, None)}, next_sequence=2,
        )
        wt.tensor_uuid_to_metadata_per_request["r1"] = {"u1": {}, "u2": {}}
    else:
        wt = _thread(_Model(postprocess_raises=True), _TensorManager())
        wt.result_tensor_queue.put(InlineToken("r1", 7, None, "text"))
        wt.result_tensor_queue.put(InlineToken("r1", 8, None, "text"))
    return wt


@pytest.mark.parametrize("stage", ["read start", "result tensor", "inline token"])
def test_an_output_failure_stops_the_request_once(stage):
    """Two of r1's outputs fail in one pass: the client gets its error, and the
    conductor gets one ABORT_REQUEST and then the data worker's READS_DONE."""
    wt = _with_output(stage)
    wt.run()

    assert _errors(wt), "the client got no error"
    assert wt.communicator.to_conductor() == [
        (ConductorMessageType.ABORT_REQUEST, "r1"),
        (ConductorMessageType.READS_DONE, "r1"),
    ]


def _fail(wt):
    try:
        raise RuntimeError("decode failed")
    except RuntimeError as exc:
        wt._fail_request("r1", exc, "text output postprocessing")


@pytest.mark.parametrize("state", ["draining", "reads done sent", "torn down"])
def test_no_abort_for_a_request_already_on_its_way_out(state):
    """The conductor is tearing it down already, or the API server handed it
    back (its own finish or abort covers it)."""
    wt = _thread(_Model(), _TensorManager())
    if state == "draining":
        wt._draining_rids.add("r1")
    elif state == "reads done sent":
        wt._reads_done_sent.add("r1")
    else:
        wt.in_flight_requests.discard("r1")
    _fail(wt)
    assert len(_errors(wt)) == 1
    assert wt.communicator.to_conductor() == []


def test_no_abort_for_a_preprocessing_failure():
    """The conductor never saw the request."""
    wt = _thread(_Model(prompt_raises=True), _TensorManager())
    wt.in_flight_requests.clear()
    wt.in_queue.put(PreprocessInput(
        request_id="r1", text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs=None,
    ))
    wt.run()
    assert len(_errors(wt)) == 1
    assert wt.communicator.to_conductor() == []


def test_the_conductor_tears_the_failed_request_down_once_and_answers_nothing():
    """The data worker's messages for one output failure, delivered to the
    conductor: one drain, one removal, and no request_failed (the client
    already has the data worker's error). An abort or a worker's failure
    report that comes after it is ignored."""
    wt = _with_output("inline token")
    wt.run()
    c = _conductor({"r1": _request_data(workers=("w0",))})
    for _entity, msg in wt.communicator.sent:
        if msg.message_type == ConductorMessageType.ABORT_REQUEST:
            c._abort_request(msg.body.request_id)
        elif msg.message_type == ConductorMessageType.READS_DONE:
            c._handle_reads_done(msg.body)
    c._abort_request("r1")
    c._fail_requests(FailRequests(errors={"r1": "failed on a worker"}))
    c._handle_reads_done(ReadsDone(request_id="r1", entity_id="w0"))

    assert _remove_targets(c, "r1") == {"w0", PREPROCESS}
    assert not c.requests and not c.draining
    assert [m for e, m in c.sent if e == "api_server"] == []


# ── one error per failed request, on every surface ──────────────────────────


def _data_worker_failure(s, key, late=False):
    """What the message thread does with a data worker's error chunk, after
    one good chunk; ``late`` adds an output that arrives after the error."""
    with s.request_lock:
        req = s.pending_requests[key]
        req.chunks.append(ResultChunk(request_id="X", modality="text", data=b"a", metadata={}))
        req.chunks.append(ResultChunk(
            request_id="X", modality="error", data=b"text output postprocessing failed",
            metadata={"status": 500},
        ))
        if late:
            req.chunks.append(ResultChunk(request_id="X", modality="text", data=b"late", metadata={}))
        req.error, req.error_status = "text output postprocessing failed", 500
        req.event.set()
        s.recently_completed[key] = 0.0
    req.notify()


def _fail_when_submitted(s, pw, n=1):
    _wait(lambda: len(pw.inputs) >= n)
    _wait(lambda: pw.inputs[n - 1].request_id in s.pending_requests)
    _data_worker_failure(s, pw.inputs[n - 1].request_id)


def test_a_stream_ends_at_the_data_worker_s_error():
    """The stream used to append its own copy of the error after the data
    worker's, and to hand on whatever followed it."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)
    _data_worker_failure(s, pw.inputs[0].request_id, late=True)

    async def read():
        return [(c.modality, c.data) async for c in s.iter_result_chunks("X")]

    assert asyncio.run(read()) == [("text", b"a"), ("error", b"text output postprocessing failed")]
    assert pw.aborted == [], "the stream aborted a request that had finished"


def test_an_engine_failure_still_ends_the_stream_with_one_error():
    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)
    with s.request_lock:
        req = s.pending_requests[pw.inputs[0].request_id]
        req.error, req.error_status = "failed in the engine", 500
        req.event.set()

    async def read():
        return [(c.modality, c.data) async for c in s.iter_result_chunks("X")]

    assert asyncio.run(read()) == [("error", b"failed in the engine")]


def test_generate_streams_one_error_line(monkeypatch):
    pw = _PreprocessWorker()
    s = _api_server(pw)
    _routes(monkeypatch, s)
    failer = threading.Thread(target=_fail_when_submitted, args=(s, pw))
    failer.start()

    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # FastAPI 0.147+ reads an omitted bool form field as False, so
            # send streaming the way mstar.client does.
            return await client.post(
                "/generate", data={"text": "first", "request_id": "X", "streaming": "true"},
            )

    response = asyncio.run(run())
    failer.join()
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert [line["modality"] for line in lines] == ["text", "error"]


def test_websocket_replies_once_to_a_data_worker_failure(monkeypatch, tmp_path):
    """A second error reply under the id would read as the answer to the
    next request the client sends under it."""
    pw = _PreprocessWorker()
    s = _api_server(pw)
    _routes(monkeypatch, s, tmp_path)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        failer = threading.Thread(target=_fail_when_submitted, args=(s, pw))
        failer.start()
        ws.send_text(json.dumps({"text": "first", "request_id": "X"}))
        replies = [json.loads(ws.receive_text()) for _ in range(2)]
        failer.join()
        assert replies[1] == {
            "request_id": "X", "error": "text output postprocessing failed", "status": 500,
        }

        ws.send_text(json.dumps({"text": "second", "request_id": "X"}))
        _wait(lambda: len(pw.inputs) == 2)
        with s.request_lock:
            req = s.pending_requests[pw.inputs[1].request_id]
            req.chunks.append(ResultChunk(request_id="X", modality="text", data=b"b", metadata={}))
            req.event.set()
        req.notify()
        after = [json.loads(ws.receive_text()) for _ in range(2)]

    assert "data" in replies[0] and "error" not in after[0]
    assert after[1] == {"request_id": "X", "finish": True}


def test_the_bridge_relays_one_error():
    from mstar.api_server.rust_frontend import RustFrontendBridge

    pw = _PreprocessWorker()
    s = _api_server(pw)
    s.submit_request(text="first", request_id="X", **TEXT)
    _data_worker_failure(s, pw.inputs[0].request_id, late=True)
    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = s
    # as _submit does: submission 1's token names this submission's key
    bridge._keys = {1: s.submission_key("X")}
    assert bridge._keys[1] == pw.inputs[0].request_id
    sent = []
    bridge._send = sent.append

    asyncio.run(bridge._relay("X", 1, s.iter_result_chunks("X")))

    assert [(m["t"], m.get("modality"), m["rid"], m["sub"]) for m in sent] == [
        ("chunk", "text", "X", 1), ("chunk", "error", "X", 1), ("done", None, "X", 1),
    ]
    assert bridge._keys == {}, "the failed submission's token still names its key"

    # the frontend's late abort for token 1 must not reach a retry under the id
    s.submit_request(text="retry", request_id="X", **TEXT)
    bridge._abort({"rid": "X", "sub": 1})
    assert pw.aborted == []
    assert pw.inputs[1].request_id in s.pending_requests
