"""``/generate/ws``: one WebSocket, many ``/generate``-shaped requests.

Drives the route through Starlette's in-process test client against a fake
API server: JSON and msgpack framings, media persisted under the upload dir
and laid out in order, pipelined requests told apart by ``request_id``, and
a rejected message answered in-band without dropping the socket.
"""

from __future__ import annotations

import asyncio
import base64
import json

import msgpack
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mstar.api_server import entrypoint
from mstar.api_server.request_types import ResultChunk


class _FakeServer:
    def __init__(self, upload_dir, chunks_per_request=2):
        self.upload_dir = upload_dir
        self.submitted = []
        self.chunks_per_request = chunks_per_request
        self.fail_for = set()

    def submit_request(self, **kwargs):
        rid = kwargs.get("request_id") or f"req-{len(self.submitted)}"
        if rid in self.fail_for:
            raise ValueError("bad request")
        kwargs["request_id"] = rid
        self.submitted.append(kwargs)
        return rid

    async def iter_result_chunks(self, request_id):
        for i in range(self.chunks_per_request):
            yield ResultChunk(
                request_id=request_id, modality="action",
                data=bytes([i]) * 4, metadata={"index": i},
            )


def _recv_until_finish(ws, binary):
    msgs = []
    while True:
        payload = msgpack.unpackb(ws.receive_bytes(), raw=False) if binary else json.loads(ws.receive_text())
        msgs.append(payload)
        if payload.get("finish") or payload.get("error"):
            return msgs


def test_json_frame_round_trip(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({
            "text": "turn left", "output_modalities": "action",
            "files": [{"name": "obs.png", "data": base64.b64encode(b"PNG").decode()}],
            "model_kwargs": {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10},
            "request_id": "r1",
        }))
        msgs = _recv_until_finish(ws, binary=False)
    assert [m.get("modality") for m in msgs[:-1]] == ["action", "action"]
    assert base64.b64decode(msgs[0]["data"]) == b"\x00" * 4 and msgs[0]["metadata"] == {"index": 0}
    assert msgs[-1] == {"request_id": "r1", "finish": True}
    (sub,) = fake.submitted
    assert sub["text"] == "turn left" and sub["output_modalities"] == ["action"]
    assert sub["input_modalities"] == ["image", "text"]
    assert [p.modality for p in sub["prompt_parts"]] == ["image", "text"]
    assert sub["model_kwargs"]["action_mode"] == "policy" and sub["streaming"] is True
    saved = sub["file_paths"]["image"][0]
    assert saved.startswith(str(tmp_path)) and open(saved, "rb").read() == b"PNG"


def test_msgpack_frames_and_pipelining(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path, chunks_per_request=1)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        for i in range(3):
            ws.send_bytes(msgpack.packb({
                "files": [{"name": f"obs{i}.jpg", "data": b"\xff\xd8" + bytes([i])}],
                "output_modalities": ["action"], "input_modalities": ["image"], "request_id": f"p{i}",
            }, use_bin_type=True))
        got = {}
        for _ in range(6):
            payload = msgpack.unpackb(ws.receive_bytes(), raw=False)
            got.setdefault(payload["request_id"], []).append(payload)
    assert set(got) == {"p0", "p1", "p2"}
    for rid, msgs in got.items():
        assert msgs[0]["modality"] == "action" and isinstance(msgs[0]["data"], bytes)
        assert msgs[-1] == {"request_id": rid, "finish": True}
    assert [s["input_modalities"] for s in fake.submitted] == [["image"]] * 3
    assert all(s["text"] is None for s in fake.submitted)


def test_rejected_message_keeps_the_socket(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path)
    fake.fail_for.add("bad")
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "x", "request_id": "bad"}))
        (err,) = _recv_until_finish(ws, binary=False)
        assert err["request_id"] == "bad" and "bad request" in err["error"]
        ws.send_text(json.dumps({"files": [{"name": "clip.xyz", "data": ""}], "request_id": "unk"}))
        (err,) = _recv_until_finish(ws, binary=False)
        assert "Cannot determine modality" in err["error"]
        ws.send_text(json.dumps({"text": "fine", "request_id": "ok"}))
        msgs = _recv_until_finish(ws, binary=False)
    assert msgs[-1] == {"request_id": "ok", "finish": True}
    assert fake.submitted[-1]["request_id"] == "ok"


def test_not_ready_closes(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", None)
    with pytest.raises(WebSocketDisconnect) as info:
        with TestClient(entrypoint.app).websocket_connect("/generate/ws"):
            pass
    assert info.value.code == 1013


def test_undecodable_and_empty_frames_are_reported(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text("not json")
        err = json.loads(ws.receive_text())
        assert err["request_id"] is None and "undecodable" in err["error"]
        ws.send_text(json.dumps([1, 2]))
        assert "must be an object" in json.loads(ws.receive_text())["error"]
        ws.send_text(json.dumps({"request_id": "empty"}))
        err = json.loads(ws.receive_text())
        assert err["request_id"] == "empty" and "neither text nor files" in err["error"]
        ws.send_text(json.dumps({"text": "still alive", "request_id": "ok"}))
        msgs = _recv_until_finish(ws, binary=False)
    assert msgs[-1] == {"request_id": "ok", "finish": True} and not fake.submitted[:-1]


def test_disconnect_mid_stream_cancels(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path)
    aborted = []

    async def slow_chunks(request_id):
        try:
            yield ResultChunk(request_id=request_id, modality="text", data=b"first", metadata={})
            await asyncio.sleep(30)
            yield ResultChunk(request_id=request_id, modality="text", data=b"never", metadata={})
        finally:
            aborted.append(request_id)

    fake.iter_result_chunks = slow_chunks
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "long", "request_id": "slow"}))
        first = json.loads(ws.receive_text())
        assert base64.b64decode(first["data"]) == b"first"
    # Leaving the block closes the socket; the pending task is cancelled and
    # the chunk iterator's cleanup (the engine abort, in production) runs.
    assert aborted == ["slow"]
