"""Exercise each error and response site that stamps an admitted weight version."""

import asyncio
import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import RequestOutputState
from mstar.api_server.request_types import ResultChunk, ResultTensors
from mstar.graph.base import GraphEdge


def test_ready_tensor_postprocess_error_keeps_its_frame_version():
    """The ready-tensor exception reads the version saved under its UUID."""
    from test_inline_tokens import _thread

    worker, _ = _thread()
    uuid = 71
    edge = GraphEdge(next_node="emit_to_client", name="image_output")
    edge.tensor_info = [SimpleNamespace(uuid=uuid)]
    state = RequestOutputState()
    state.order[uuid] = (0, None)
    state.next_sequence = 1
    worker.request_output_state["r1"] = state
    worker.tensor_uuid_to_metadata_per_request["r1"] = {uuid: {"weight_version": "v1"}}

    def fail_read(_):
        raise OSError("tensor gone")

    worker.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {"r1": [edge]}, take_failed_reads=lambda: [],
        get_tensor=fail_read, dereference=lambda _: None,
    )
    assert worker._process_read_tensors()
    chunk = worker.out_queue.get_nowait()
    assert chunk.modality == "error"
    assert chunk.metadata == {"status": 500, "weight_version": "v1"}


def test_transfer_start_error_keeps_result_frame_version():
    """A read that fails before it starts still tags each error chunk."""
    from test_inline_tokens import _thread

    worker, stop = _thread()
    edge = GraphEdge(next_node="emit_to_client", name="image_output")
    edge.tensor_info = [SimpleNamespace(uuid=72)]

    def fail_start(_):
        raise OSError("read did not start")

    worker._read_result_tensor = fail_start
    worker.result_tensor_queue.put(ResultTensors(
        request_id="r2", modality="image", graph_edge=edge, loop_indices=None,
        metadata={"weight_version": "v2"},
    ))
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        chunk = worker.out_queue.get(timeout=5)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert chunk.modality == "error"
    assert chunk.metadata == {"status": 500, "weight_version": "v2"}


def test_stream_tail_error_keeps_admitted_version():
    """A late failure after the stream opened adds a versioned terminal error."""
    from test_api_result_delivery import _api_server_stub, _pending_request, _register, _StubPreprocessWorker

    server = _api_server_stub(_StubPreprocessWorker())
    req = _pending_request()
    req.weight_version = "v3"
    req.error, req.error_status = "delivery timed out", 500
    req.event.set()
    _register(server, "r3", req)

    async def read():
        return [chunk async for chunk in server.iter_result_chunks("r3")]

    chunks = asyncio.run(read())
    assert len(chunks) == 1 and chunks[0].modality == "error"
    assert chunks[0].metadata == {"status": 500, "weight_version": "v3"}


def test_generate_ws_error_has_top_level_version(monkeypatch, tmp_path):
    """The WebSocket error reply needs its own version, not only chunk metadata."""
    from test_generate_ws import _FakeServer

    server = _FakeServer(tmp_path)

    async def chunks(request_id):
        yield ResultChunk(request_id, "error", b"generation failed", {
            "status": 500, "weight_version": "v4",
        })

    server.iter_result_chunks = chunks
    monkeypatch.setattr(entrypoint, "api_server", server)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "go", "request_id": "r4"}))
        reply = json.loads(ws.receive_text())
    assert reply == {
        "request_id": "r4", "error": "generation failed", "status": 500, "weight_version": "v4",
    }


def test_generate_nonstreaming_top_level_version(monkeypatch):
    """The non-streaming envelope repeats the first chunk's version."""
    from test_generate_route import _RecordingServer

    server = _RecordingServer()

    async def collect(_request_id, _request):
        return [ResultChunk("r", "text", b"hi", {"weight_version": "v5"})]

    server.collect_results = collect
    monkeypatch.setattr(entrypoint, "api_server", server)
    response = TestClient(entrypoint.app).post("/generate", data={"text": "hi", "streaming": "false"})
    assert response.status_code == 200
    assert response.json().get("weight_version") == "v5"
    assert response.json()["outputs"]["text"][0]["metadata"]["weight_version"] == "v5"
