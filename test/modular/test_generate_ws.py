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
import multiprocessing as mp
import os
import signal
import threading
import time
from pathlib import Path

import msgpack
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mstar.api_server import entrypoint, ws_frames
from mstar.api_server.request_types import ResultChunk


class _FakeServer:
    takes_stream_input = False

    def __init__(self, upload_dir, chunks_per_request=2):
        self.upload_dir = upload_dir
        self.submitted = []
        self.chunks_per_request = chunks_per_request
        self.fail_for = set()
        # request_id -> (message, status): fails in-band after one chunk
        self.error_for: dict = {}

    async def wait_for_request_id(self, request_id):
        pass

    def submission_key(self, request_id):
        return request_id

    def abort_submission(self, key):
        self.aborted = [*getattr(self, "aborted", []), key]

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
            if request_id in self.error_for:
                message, status = self.error_for[request_id]
                yield ResultChunk(
                    request_id=request_id, modality="error",
                    data=message.encode(), metadata={"status": status},
                )
                return


@pytest.fixture
def frame_pool():
    ws_frames.start()
    yield
    ws_frames.stop()


def _every_frame_off_the_loop(monkeypatch):
    monkeypatch.setattr(ws_frames, "OFF_LOOP_TEXT_BYTES", 0)
    monkeypatch.setattr(ws_frames, "OFF_LOOP_BINARY_BYTES", 0)


@pytest.fixture(params=[False, True], ids=["on the loop", "in the pool"])
def off_loop(request, monkeypatch):
    """The test twice: every frame parsed on the loop, then in the pool."""
    if request.param:
        request.getfixturevalue("frame_pool")
        _every_frame_off_the_loop(monkeypatch)


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


def test_a_null_field_counts_as_absent(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "hi", "request_id": "r", "append": None, "files": None}))
        msgs = _recv_until_finish(ws, binary=False)
    assert msgs[-1] == {"request_id": "r", "finish": True}
    assert [s["request_id"] for s in fake.submitted] == ["r"]


def test_not_ready_closes(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", None)
    with pytest.raises(WebSocketDisconnect) as info:
        with TestClient(entrypoint.app).websocket_connect("/generate/ws"):
            pass
    assert info.value.code == 1013


def test_undecodable_and_empty_frames_are_reported(monkeypatch, tmp_path, off_loop):
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


def test_unpadded_base64_is_accepted(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path, chunks_per_request=1)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({
            "files": [{"name": "obs.png", "data": base64.b64encode(b"PN").decode().rstrip("=")}],
            "output_modalities": "action", "request_id": "r1",
        }))
        msgs = _recv_until_finish(ws, binary=False)
    assert msgs[-1] == {"request_id": "r1", "finish": True}
    assert open(fake.submitted[0]["file_paths"]["image"][0], "rb").read() == b"PN"


def test_in_band_failure_replaces_the_finish(monkeypatch, tmp_path):
    """A request that fails after it was accepted answers with the error and
    its status, and nothing else: no finish frame follows."""
    fake = _FakeServer(tmp_path, chunks_per_request=2)
    fake.error_for["mid"] = ("Error in worker: ValueError: bad domain", 400)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "go", "request_id": "mid"}))
        msgs = _recv_until_finish(ws, binary=False)
        assert [m.get("modality") for m in msgs[:-1]] == ["action"]
        assert msgs[-1] == {"request_id": "mid", "error": "Error in worker: ValueError: bad domain", "status": 400}
        # the next reply on the socket belongs to the next request
        ws.send_text(json.dumps({"text": "again", "request_id": "ok"}))
        assert json.loads(ws.receive_text())["request_id"] == "ok"


@pytest.mark.parametrize("message, field", [
    ({"text": 5}, "text"),
    ({"text": "x", "request_id": 7}, "request_id"),
    ({"text": "x", "files": "obs.png"}, "files"),
    ({"text": "x", "files": [{"name": 3, "data": ""}]}, "name"),
    ({"text": "x", "output_modalities": [1]}, "output_modalities"),
    ({"text": "x", "input_modalities": {"a": 1}}, "input_modalities"),
    ({"text": "x", "model_kwargs": [1]}, "model_kwargs"),
])
def test_wrongly_typed_fields_are_rejected_by_name(monkeypatch, tmp_path, message, field):
    fake = _FakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps(message))
        err = json.loads(ws.receive_text())
    assert "error" in err and field in err["error"], err
    assert not fake.submitted


def test_uploads_are_cleaned_up(monkeypatch, tmp_path, off_loop):
    """Files a message carried are scheduled for deletion once the request is
    done, as the HTTP route does; a message rejected halfway, or before its
    files are taken, leaves none. Also for a frame parsed in the pool, which
    saves its files as it parses."""
    fake = _FakeServer(tmp_path, chunks_per_request=1)
    scheduled = []
    monkeypatch.setattr(entrypoint, "_schedule_upload_cleanup", lambda paths, delay_s=60.0: scheduled.append(paths))
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({
            "files": [{"name": "a.png", "data": base64.b64encode(b"A").decode()},
                      {"name": "b.wav", "data": base64.b64encode(b"B").decode()}],
            "output_modalities": "action", "request_id": "two",
        }))
        _recv_until_finish(ws, binary=False)
        ws.send_text(json.dumps({
            "files": [{"name": "c.png", "data": base64.b64encode(b"C").decode()},
                      {"name": "clip.xyz", "data": ""}],
            "request_id": "half",
        }))
        err = json.loads(ws.receive_text())
        ws.send_bytes(msgpack.packb({"files": [{"name": "d.png", "data": b"D"}], "request_id": 5}))
        refused = msgpack.unpackb(ws.receive_bytes(), raw=False)
    assert "Cannot determine modality" in err["error"]
    assert refused["error"] == "request_id must be a string"
    assert scheduled == [fake.submitted[0]["file_paths"]]
    assert [open(p, "rb").read() for p in fake.submitted[0]["file_paths"]["image"]] == [b"A"]
    saved = sorted(f.name.split("_", 1)[1] for f in tmp_path.iterdir())
    assert saved == ["a.png", "b.wav"], saved


def test_a_large_frame_parses_while_the_loop_keeps_serving(frame_pool, tmp_path):
    """json.loads and base64 decoding hold the GIL, so a thread can't take a
    large frame off the loop: a 100 MiB message stalled every request for
    about half a second. A frame past OFF_LOOP_TEXT_BYTES parses in the pool, and
    the loop keeps running meanwhile."""
    data = bytes(range(256)) * (24 << 12)  # 24 MiB of file, 32 MiB as base64
    frame = json.dumps({"request_id": "r", "files": [{"name": "a.png", "data": base64.b64encode(data).decode()}]})

    async def parse_beside_a_ticker():
        ticks, done = 0, asyncio.Event()

        async def tick():
            nonlocal ticks
            while not done.is_set():
                await asyncio.sleep(0.001)
                ticks += 1

        ticker = asyncio.create_task(tick())
        await asyncio.sleep(0.01)
        ticks = 0
        message = await ws_frames.parse_frame(frame, tmp_path)
        done.set()
        await ticker
        return message, ticks

    message, ticks = asyncio.run(parse_beside_a_ticker())
    assert ticks >= 10, ticks  # on the loop this is 1: the parse is one blocking call
    (entry,) = message["files"]
    assert open(entry["data"].claim(), "rb").read() == data


def test_a_frame_in_flight_when_a_parser_process_dies_fails_and_a_new_pool_takes_the_next(
    frame_pool, monkeypatch, tmp_path,
):
    _every_frame_off_the_loop(monkeypatch)

    async def kill_a_process_then_parse():
        broken = ws_frames._pool
        broken.submit(ws_frames.parse, b"\x80").result()  # a process is up
        for process in broken._processes.values():
            process.kill()
        with pytest.raises(OSError, match="parser process died"):
            await ws_frames.parse_frame('{"text": "lost"}', tmp_path)
        assert ws_frames._pool not in (None, broken)
        return await ws_frames.parse_frame('{"text": "next"}', tmp_path)

    assert asyncio.run(kill_a_process_then_parse()) == {"text": "next"}
    assert list(tmp_path.iterdir()) == []


def test_a_file_that_cannot_be_saved_refuses_its_message_and_leaves_nothing(monkeypatch, tmp_path, off_loop):
    fake = _FakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "r", "text": "x", "files": [
            {"name": "a.png", "data": "QUJD"}, {"name": "b" * 300 + ".png", "data": "QUJD"},
        ]}))
        reply = json.loads(ws.receive_text())
    assert reply["request_id"] == "r" and str(tmp_path) not in reply["error"]
    assert list(tmp_path.iterdir()) == [] and not fake.submitted


def test_a_message_the_pool_cannot_send_back_takes_its_saved_files_with_it(frame_pool, monkeypatch, tmp_path):
    """Pickling the message can fail after its files are saved, on a nesting
    json takes, where the worker can no longer remove them."""
    _every_frame_off_the_loop(monkeypatch)
    frame = '{"files": [{"name": "a.png", "data": "QUJD"}], "model_kwargs": ' + "[" * 8000 + "]" * 8000 + "}"

    with pytest.raises(RecursionError):
        asyncio.run(ws_frames.parse_frame(frame, tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_a_parse_its_socket_stopped_waiting_for_still_takes_its_files_with_it(frame_pool, monkeypatch, tmp_path):
    """The frame's task is cancelled once the worker has saved its first file,
    and the worker then dies on the second: nothing is left."""
    _every_frame_off_the_loop(monkeypatch)
    frame = json.dumps({"files": [
        {"name": "a.png", "data": "QUJD"},
        {"name": "b.png", "data": base64.b64encode(bytes(64 << 20)).decode()},
    ]})

    async def cancel_once_saved():
        task = asyncio.create_task(ws_frames.parse_frame(frame, tmp_path))
        deadline = time.monotonic() + 30
        while not any(p.name.endswith("_a.png") for p in tmp_path.iterdir()):
            assert not task.done() and time.monotonic() < deadline, "the first file was never saved"
            await asyncio.sleep(0.001)
        task.cancel()
        for process in ws_frames._pool._processes.values():
            process.kill()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_once_saved())
    deadline = time.monotonic() + 10
    while list(tmp_path.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(tmp_path.iterdir()) == []


def _stopped_while_saving(frame, upload_dir):
    ws_frames._worker_init()

    def stop_once_the_second_file_is_open():
        while not any(p.name.endswith("_b.png") for p in Path(upload_dir).iterdir()):
            time.sleep(0.0005)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=stop_once_the_second_file_is_open, daemon=True).start()
    ws_frames._parse_saving_files(frame, False, upload_dir)
    time.sleep(30)


def test_a_pool_process_stopped_mid_frame_removes_what_it_saved(tmp_path):
    """When one pool process dies the pool stops the others, whose frames'
    futures have already failed and been swept; one still saving, the file it
    is writing included, must not leave its files."""
    frame = tmp_path / "f.wsframe"
    frame.write_text(json.dumps({"files": [
        {"name": "a.png", "data": "QUJD"},
        {"name": "b.png", "data": base64.b64encode(bytes(128 << 20)).decode()},
    ]}))
    process = mp.get_context("spawn").Process(target=_stopped_while_saving, args=(str(frame), str(tmp_path)))
    process.start()
    process.join(60)
    assert process.exitcode == 0
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".png"] == []


def test_a_frame_the_server_could_not_hand_off_gets_a_500_not_undecodable(monkeypatch, tmp_path):
    fake = _FakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)

    async def fails(payload, upload_dir):
        raise OSError("disk full")

    monkeypatch.setattr(ws_frames, "parse_frame", fails)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "x", "request_id": "r"}))
        reply = json.loads(ws.receive_text())
    assert reply["status"] == 500 and "undecodable" not in reply["error"]


def test_data_that_is_not_base64_does_not_travel_back_from_the_pool(frame_pool, monkeypatch, tmp_path):
    """The request refuses it either way; carrying it back would put its bytes
    through the loop again."""
    _every_frame_off_the_loop(monkeypatch)
    frame = json.dumps({"files": [{"name": "a.png", "data": "QUJD" * (1 << 18) + "\u00e9"}]})
    message = asyncio.run(ws_frames.parse_frame(frame, tmp_path))
    assert len(message["files"][0]["data"]) < 16


def test_a_socket_closed_while_its_files_decode_leaves_none(monkeypatch, tmp_path):
    """Closing the socket cancels the message's task while the threadpool is
    still writing its files, which the task never got to schedule for
    cleanup: they are removed once the write is done."""
    monkeypatch.setattr(entrypoint, "api_server", _FakeServer(tmp_path))
    decoding, closed, unlinked = threading.Event(), threading.Event(), threading.Event()
    decode, unlink = entrypoint._decode_ws_files, entrypoint.unlink_files

    def slow(files, upload_dir):
        decoding.set()
        closed.wait(10)  # the socket closes, and its task is cancelled, first
        return decode(files, upload_dir)

    def unlink_files(file_paths):
        unlink(file_paths)
        unlinked.set()

    monkeypatch.setattr(entrypoint, "_decode_ws_files", slow)
    monkeypatch.setattr(entrypoint, "unlink_files", unlink_files)

    class _Socket:
        def __init__(self):
            self.frames = [{"type": "websocket.receive", "text": json.dumps(
                {"text": "hi", "files": [_OBS], "request_id": "r"})}]

        async def accept(self):
            pass

        async def receive(self):
            if self.frames:
                return self.frames.pop()
            await asyncio.to_thread(decoding.wait)
            return {"type": "websocket.disconnect"}

    async def run():
        await entrypoint.generate_ws(_Socket())
        await asyncio.sleep(0)  # the task takes its cancel
        closed.set()
        assert await asyncio.to_thread(unlinked.wait, 10), "nothing removed the decoded files"

    asyncio.run(run())
    assert list(tmp_path.iterdir()) == []


class _ListeningFakeServer(_FakeServer):
    """A model with a client stream: requests stay open unless they end their input."""
    takes_stream_input = True

    def __init__(self, upload_dir):
        super().__init__(upload_dir, chunks_per_request=1)
        self.appends = []

    def taking_input(self, key):
        return True

    def stream_input(self, key, file_paths=None, end_input=False, **append):
        self.appends.append((key, sorted(file_paths or {}), end_input, append))
        return True

    async def iter_result_chunks(self, request_id):
        # a call finishes once its input has ended, at its opening or by an append
        while not (
            any(s["request_id"] == request_id and s["end_input"] is True for s in self.submitted)
            or any(a[0] == request_id and a[2] is True for a in self.appends)
        ):
            await asyncio.sleep(0.01)
        async for chunk in super().iter_result_chunks(request_id):
            yield chunk


def test_an_opening_with_input_and_settings_is_never_an_append(monkeypatch, tmp_path):
    """Routing is by shape at the top level: text or files open a request even
    with ``end_input`` and settings beside them, and an ``append`` key alone
    makes an append, whatever its record holds."""
    fake = _ListeningFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "one", "text": "hi", "end_input": True,
                                 "model_kwargs": {"gain": 2}}))
        assert _recv_until_finish(ws, binary=False)[-1] == {"request_id": "one", "finish": True}
        ws.send_text(json.dumps({"request_id": "stray", "append": {"text": "hi", "model_kwargs": {}}}))
        stray = json.loads(ws.receive_text())

    assert [s["request_id"] for s in fake.submitted] == ["one"]
    assert fake.submitted[0]["model_kwargs"] == {"gain": 2} and fake.submitted[0]["end_input"] is True
    assert stray["append"] is True and "not taking input on this socket" in stray["error"]
    assert fake.appends == []


@pytest.mark.parametrize("record", [
    [{"name": "c.wav", "data": base64.b64encode(b"RIFF").decode()}],
    {"files": [{"name": "c.wav", "data": base64.b64encode(b"RIFF").decode()}]},
], ids=["shorthand", "record"])
def test_a_live_call_takes_both_append_forms(monkeypatch, tmp_path, record):
    fake = _ListeningFakeServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "call", "output_modalities": "text"}))
        ws.send_text(json.dumps({"request_id": "call", "append": record, "end_input": True}))
        msgs = _recv_until_finish(ws, binary=False)

    assert not any("error" in m for m in msgs)
    [(key, streams, end_input, append)] = fake.appends
    assert callable(append.pop("on_late"))  # how the data worker answers a late append
    assert (key, streams, end_input, append) == (
        "call", ["audio"], True, {"text": None, "file_streams": None, "model_kwargs": None},
    )


def _slow_decode(monkeypatch, seconds=0.2):
    """A message with files takes ``seconds`` longer to decode, like a large upload."""
    decode = entrypoint._decode_ws_files

    def slow(files, upload_dir):
        if files:
            time.sleep(seconds)
        return decode(files, upload_dir)

    monkeypatch.setattr(entrypoint, "_decode_ws_files", slow)


def _recv_replies(ws, request_ids):
    """Every reply until each of ``request_ids`` has finished or failed."""
    open_ids, replies = set(request_ids), []
    while open_ids:
        reply = json.loads(ws.receive_text())
        replies.append(reply)
        if reply.get("finish") or reply.get("error"):
            open_ids.discard(reply["request_id"])
    return replies


_OBS = {"name": "obs.png", "data": base64.b64encode(b"PNG").decode()}


@pytest.mark.parametrize("slow_first", ["file-decode", "request-id-wait"])
def test_pipelined_messages_are_submitted_in_send_order(monkeypatch, tmp_path, slow_first):
    """A message that takes longer before its submission (a large file, an id
    still in flight) does not let the next message on the socket overtake it."""
    fake = _FakeServer(tmp_path, chunks_per_request=1)
    if slow_first == "file-decode":
        _slow_decode(monkeypatch)
    else:
        async def wait_for_first(request_id):
            if request_id == "first":
                await asyncio.sleep(0.2)

        fake.wait_for_request_id = wait_for_first
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "obs 0", "files": [_OBS], "request_id": "first"}))
        ws.send_text(json.dumps({"text": "obs 1", "request_id": "second"}))
        _recv_replies(ws, ["first", "second"])
    assert [s["request_id"] for s in fake.submitted] == ["first", "second"]


def test_a_reused_request_id_refuses_the_later_message(monkeypatch, tmp_path):
    """Two pipelined messages under one id: the one sent first is served and
    the one sent second gets the 409, whichever decodes faster."""
    fake = _FakeServer(tmp_path, chunks_per_request=1)
    submit = fake.submit_request

    def refuse_a_live_id(**kwargs):
        if any(s["request_id"] == kwargs["request_id"] for s in fake.submitted):
            raise HTTPException(status_code=409, detail="request_id 'step' is already in use")
        return submit(**kwargs)

    fake.submit_request = refuse_a_live_id
    _slow_decode(monkeypatch)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "obs 0", "files": [_OBS], "request_id": "step"}))
        ws.send_text(json.dumps({"text": "obs 1", "request_id": "step"}))
        replies = _recv_replies(ws, ["step"])
        while not any(r.get("error") for r in replies):
            replies.append(json.loads(ws.receive_text()))
    (served,) = fake.submitted
    assert served["text"] == "obs 0" and served["file_paths"]
    assert [r["status"] for r in replies if r.get("error")] == [409]


def test_a_rejected_message_keeps_the_order_of_the_others(monkeypatch, tmp_path):
    """A message refused before submission answers at once, and the one after
    it still waits for the slow one before it."""
    fake = _FakeServer(tmp_path, chunks_per_request=1)
    _slow_decode(monkeypatch)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "obs 0", "files": [_OBS], "request_id": "first"}))
        ws.send_text(json.dumps({"request_id": "empty"}))
        ws.send_text(json.dumps({"text": "obs 2", "request_id": "third"}))
        replies = _recv_replies(ws, ["first", "empty", "third"])
    assert [s["request_id"] for s in fake.submitted] == ["first", "third"]
    assert any(r["request_id"] == "empty" and "neither text nor files" in r.get("error", "") for r in replies)


def test_a_long_stream_does_not_hold_up_the_next_request(monkeypatch, tmp_path):
    """Only submissions keep the send order: a request still streaming does not
    delay the next one's submission or replies."""
    fake = _FakeServer(tmp_path, chunks_per_request=1)
    stream = fake.iter_result_chunks
    release = asyncio.Event()

    async def held_first(request_id):
        async for chunk in stream(request_id):
            yield chunk
        if request_id == "long":
            await release.wait()

    fake.iter_result_chunks = held_first
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "obs 0", "request_id": "long"}))
        assert json.loads(ws.receive_text())["request_id"] == "long"
        ws.send_text(json.dumps({"text": "obs 1", "request_id": "short"}))
        replies = _recv_replies(ws, ["short"])
    assert replies[-1] == {"request_id": "short", "finish": True}
    assert [s["request_id"] for s in fake.submitted] == ["long", "short"]


def test_an_append_is_not_held_up_by_another_message_waiting_its_turn(monkeypatch, tmp_path):
    """Openings submit in send order, so a live call opened behind a slow
    upload waits for it, and so do its appends; another call's appends go
    through meanwhile."""
    fake = _ListeningFakeServer(tmp_path)
    log = []
    submit, stream_input = fake.submit_request, fake.stream_input

    def logged_submit(**kwargs):
        log.append(("submit", kwargs["request_id"]))
        return submit(**kwargs)

    def logged_stream_input(key, *args, **kwargs):
        log.append(("append", key))
        return stream_input(key, *args, **kwargs)

    fake.submit_request, fake.stream_input = logged_submit, logged_stream_input
    _slow_decode(monkeypatch, seconds=0.5)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"request_id": "early", "output_modalities": "text"}))
        ws.send_text(json.dumps({"request_id": "upload", "text": "hi", "files": [_OBS], "end_input": True}))
        ws.send_text(json.dumps({"request_id": "late", "output_modalities": "text"}))
        ws.send_text(json.dumps({"request_id": "late", "append": {"text": "hi"}}))
        ws.send_text(json.dumps({"request_id": "early", "append": {"text": "hi"}, "end_input": True}))
        ws.send_text(json.dumps({"request_id": "late", "end_input": True}))
        replies = _recv_replies(ws, ["early", "upload", "late"])

    assert not any(r.get("error") for r in replies)
    assert log.index(("append", "early")) < log.index(("submit", "upload"))
    assert [e for e in log if e[1] == "late"] == [("submit", "late"), ("append", "late"), ("append", "late")]
