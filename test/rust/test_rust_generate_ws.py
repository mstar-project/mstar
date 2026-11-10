"""``/generate/ws`` on mstar-server against the Python route.

Each case sends the same frames (JSON text, or msgpack binary) to both
frontends, served by one scripted data plane that acts as ``APIServer`` does
for a submission and for a live call's appends, and compares every reply
(grouped by request, in order) and what each submission and append carried."""

import asyncio
import base64
import contextlib
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")
pytest.importorskip("websockets")
msgpack = pytest.importorskip("msgpack")

import websockets.exceptions
import websockets.sync.client
from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _Chunk, _free_port, _wait_healthy

from mstar.api_server import entrypoint
from mstar.api_server.rust_frontend import RustFrontendBridge

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

STREAMS = ("audio", "text")


class _Server:
    """A submission answers with what it and its appends carried; a live call
    answers once its input has ended, or with the error that rejected it."""

    def __init__(self, upload_dir: Path, streams=()):
        self.upload_dir = upload_dir
        self.client_streams = frozenset(streams)
        self.takes_stream_input = bool(streams)
        self.metrics = None
        self.fatal_error = None
        self.reset()

    def reset(self):
        self.submitted: list[dict] = []
        self.appended: list[dict] = []
        self.calls: dict[str, dict] = {}

    def cleanup(self):
        pass

    async def wait_for_request_id(self, request_id, timeout=None):
        return None

    @staticmethod
    def _files(file_paths):
        return {m: [(Path(p).name.split("_", 1)[1], Path(p).read_bytes()) for p in ps]
                for m, ps in (file_paths or {}).items()}

    def submit_request(self, *, text=None, file_paths=None, input_modalities, output_modalities, model_kwargs=None,
                       prompt_parts=None, streaming=True, request_id=None, model=None, end_input=True,
                       file_streams=None, raw_inputs=None):
        if text == "refuse me":
            raise ValueError("the model refuses this text")
        rid = request_id or f"gen-{len(self.submitted)}"
        self.submitted.append({
            # None and {} alike: submit_request reads either as no kwargs
            "text": text, "in": list(input_modalities), "out": list(output_modalities), "mk": model_kwargs or {},
            "parts": [(p.modality, p.text, p.index, p.role) for p in prompt_parts or []],
            "files": self._files(file_paths), "streaming": streaming, "end_input": end_input,
            "file_streams": file_streams,
        })
        opened = set()
        if self.client_streams and end_input is not True:
            opened = set(self.client_streams) - set(end_input or ())
        self.calls[rid] = {"open": opened, "inputs": [], "error": None, "text": text}
        return rid

    def submission_key(self, request_id):
        return request_id

    def relay_chunks_to(self, key, sink, tag):
        pass

    def abort_submission(self, key):
        pass

    def taking_input(self, key):
        call = self.calls.get(key)
        return bool(call and call["open"] and call["error"] is None)

    def stream_input(self, key, file_paths=None, end_input=False, *, text=None, file_streams=None,
                     model_kwargs=None, on_late=None):
        named = {n for names in (file_streams or {}).values() for n in names if n is not None}
        unknown = named - self.client_streams
        if unknown:
            raise ValueError(f"no client stream named {', '.join(map(repr, sorted(unknown)))}; "
                             f"this model takes {sorted(self.client_streams)}")
        if not self.taking_input(key):
            return False
        call = self.calls[key]
        self.appended.append({"key": key, "text": text, "files": self._files(file_paths),
                              "file_streams": file_streams, "mk": model_kwargs, "end_input": end_input})
        for paths in (file_paths or {}).values():
            for p in paths:
                Path(p).unlink(missing_ok=True)
        call["inputs"].append(text)
        call["open"] -= set(call["open"]) if end_input is True else set(end_input or ())
        if text == "too late" and on_late is not None:
            on_late("the model's text stream has ended")
        return True

    def reject_input(self, key, error, status):
        if not self.taking_input(key):
            return False
        call = self.calls[key]
        call["error"], call["open"] = (error, status), set()
        return True

    async def iter_result_chunks(self, request_id):
        call = self.calls[request_id]
        deadline = time.monotonic() + 10
        while call["open"] and call["error"] is None and time.monotonic() < deadline:
            await asyncio.sleep(0.002)
        if call["error"] is not None:
            error, status = call["error"]
            yield _Chunk("error", error.encode(), {"status": status})
            return
        if call["text"] == "fail later":
            yield _Chunk("text", b"partial")
            yield _Chunk("error", b"the engine gave up", {"status": 503})
            return
        yield _Chunk("text", json.dumps({"inputs": call["inputs"]}).encode())
        yield _Chunk("audio", b"\x00\x01\xff", {"sample_rate": 24000})


@contextlib.contextmanager
def _rust(server: _Server, **extra_env):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_genws_")
    env = dict(os.environ, MSTAR_CLIENT_STREAMS=json.dumps(sorted(server.client_streams)), **extra_env)
    proc = subprocess.Popen([BINARY, "qwen3_5_0.8b", str(port), bridge_dir, str(server.upload_dir)], env=env)
    bridge = RustFrontendBridge(server, bridge_dir)
    threading.Thread(target=bridge.run, daemon=True).start()
    try:
        assert _wait_healthy(port)
        yield port
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def stacks():
    with contextlib.ExitStack() as exits:
        made = {}

        def get(live: bool):
            if live not in made:
                server = _Server(Path(tempfile.mkdtemp(prefix="mstar_genws_up_")), STREAMS if live else ())
                made[live] = (server, exits.enter_context(_rust(server)))
            return made[live]
        yield get


def _decode(reply):
    return msgpack.unpackb(reply, raw=False) if isinstance(reply, bytes) else json.loads(reply)


def _ended(reply: dict) -> bool:
    return reply.get("finish") is True or ("error" in reply and not reply.get("append"))


def _python(server, frames, settle, ending):
    entrypoint.api_server = server
    replies = []
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        for frame in frames:
            (ws.send_bytes if isinstance(frame, bytes) else ws.send_text)(frame)
            if settle:
                time.sleep(0.05)
        while len([r for r in replies if _ended(r) or r.get("append")]) < ending:
            replies.append(_decode(ws.receive_bytes() if isinstance(frames[0], bytes) else ws.receive_text()))
    return replies


def _rust_session(port, frames, settle, ending):
    replies = []
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/generate/ws", compression=None, max_size=None) as ws:
        for frame in frames:
            ws.send(frame)
            if settle:
                time.sleep(0.05)
        while len([r for r in replies if _ended(r) or r.get("append")]) < ending:
            replies.append(_decode(ws.recv(timeout=15)))
    return replies


def _by_request(replies):
    grouped: dict = {}
    for r in replies:
        grouped.setdefault(str(r.get("request_id")), []).append(r)
    return grouped


def _both(stacks, monkeypatch, live, messages, *, binary=False, ending=None, settle=False):
    """Both frontends' replies, grouped by request, and their submissions and appends.
    ``ending`` is how many ending replies (finishes, errors, late appends) the session gets."""
    server, port = stacks(live)
    pack = (lambda m: msgpack.packb(m, use_bin_type=True)) if binary else json.dumps
    frames = [m if isinstance(m, (str, bytes)) else pack(m) for m in messages]
    ending = len(frames) if ending is None else ending
    monkeypatch.setattr(entrypoint, "api_server", server)
    server.reset()
    py = _python(server, frames, settle, ending)
    py_record = (list(server.submitted), list(server.appended))
    server.reset()
    rust = _rust_session(port, frames, settle, ending)
    rust_record = (list(server.submitted), list(server.appended))
    return (_by_request(py), py_record), (_by_request(rust), rust_record)


PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489")


def _file(name, data, binary, **extra):
    return {"name": name, "data": data if binary else base64.b64encode(data).decode(), **extra}


@pytest.mark.parametrize("binary", [False, True], ids=["json", "msgpack"])
def test_requests_and_their_files_reach_the_engine_alike(stacks, monkeypatch, binary):
    messages = [
        {"request_id": "a", "text": "hello", "model_kwargs": {"temperature": 0.2}},
        {"request_id": "b", "text": "what is it", "output_modalities": ["text", "audio"],
         "files": [_file("cat.png", PNG, binary), _file("x.wav", b"RIFF", binary)]},
        {"text": "no id", "input_modalities": "image,text", "files": [_file("p.png", PNG, binary)]},
        {"request_id": "c", "files": [_file("only.png", PNG, binary)], "model_kwargs": json.dumps({"seed": 3})},
    ]
    (py, (ps, _)), (rust, (rs, _)) = _both(stacks, monkeypatch, False, messages, binary=binary)
    assert rs == ps and len(ps) == 4
    for rid in ("a", "b", "c"):
        assert rust[rid] == py[rid]
        assert rust[rid][-1] == {"request_id": rid, "finish": True}
    # the message with no id runs under one the server makes, on each side its own
    (py_other,), (rust_other,) = py.keys() - {"a", "b", "c"}, rust.keys() - {"a", "b", "c"}
    unnamed = lambda replies: [{k: v for k, v in r.items() if k != "request_id"} for r in replies]  # noqa: E731
    assert unnamed(rust[rust_other]) == unnamed(py[py_other])


@pytest.mark.parametrize("binary", [False, True], ids=["json", "msgpack"])
def test_a_refused_message_reads_as_pythons(stacks, monkeypatch, binary):
    messages = [
        {"request_id": "e1"},
        {"request_id": "e2", "text": 5},
        {"request_id": "e3", "text": "x", "files": "not a list"},
        {"request_id": "e4", "files": [{"name": "a.xyz", "data": "AAAA"}]},
        {"request_id": "e5", "files": [{"data": "AAAA"}]},
        {"request_id": "e6", "files": [{"name": "a.png"}]},
        {"request_id": "e7", "text": "x", "output_modalities": 5},
        {"request_id": "e8", "text": "x", "model_kwargs": [1]},
        {"request_id": 7, "text": "x"},
        {"request_id": "x" * 200, "text": "x"},
        [1, 2],
        {"request_id": "e9", "text": "refuse me"},
        # refused before it had an id: its error names none
        {"text": "refuse me"},
        {"request_id": "e10", "text": "fail later"},
        {"request_id": "e11", "text": "x", "end_input": "yes"},
        {"request_id": "", "text": "empty id is no id"},
    ]
    (py, (ps, _)), (rust, (rs, _)) = _both(stacks, monkeypatch, False, messages, binary=binary)
    assert rs == ps
    for rid in [f"e{i}" for i in range(1, 12)] + ["x" * 200]:
        assert rust.get(rid) == py.get(rid), rid
    assert rust["None"] == py["None"]


def test_an_undecodable_frame_is_answered_and_the_socket_goes_on(stacks, monkeypatch):
    messages = ["{not json", {"request_id": "ok", "text": "fine"}]
    (py, _), (rust, _) = _both(stacks, monkeypatch, False, messages)
    assert rust["ok"] == py["ok"]
    (bad_py,), (bad_rust,) = py["None"], rust["None"]
    assert bad_rust.keys() == bad_py.keys() == {"request_id", "error"}
    assert bad_rust["error"].startswith("undecodable frame: ")


@pytest.mark.parametrize("binary", [False, True], ids=["json", "msgpack"])
def test_a_live_call_takes_appends_until_its_input_ends(stacks, monkeypatch, binary):
    messages = [
        {"request_id": "call", "text": "start", "end_input": False},
        {"request_id": "call", "append": {"text": "more", "files": [_file("a.wav", b"RIFF1", binary, stream="audio")]}},
        {"request_id": "call", "append": [_file("b.wav", b"RIFF2", binary)], "end_input": ["audio"]},
        {"request_id": "call", "append": {"model_kwargs": {"k": 2}}},
        {"request_id": "call", "append": {"text": "last", "model_kwargs": {"k": 1}}, "end_input": True},
        {"request_id": "call", "append": {"text": "after the end"}},
    ]
    (py, (ps, pa)), (rust, (rs, ra)) = _both(stacks, monkeypatch, True, messages, binary=binary, ending=2, settle=True)
    assert (rs, ra) == (ps, pa) and len(pa) == 4
    assert rust["call"] == py["call"]


@pytest.mark.parametrize("case", ["unknown-field", "misplaced", "unknown-stream", "bad-text", "empty"])
def test_a_rejected_append_ends_the_call_with_its_error(stacks, monkeypatch, case):
    append = {
        "unknown-field": {"request_id": "r", "append": {"audio": "x"}},
        "misplaced": {"request_id": "r", "append": {"text": "a"}, "text": "b"},
        "unknown-stream": {"request_id": "r", "append": [_file("a.wav", b"RIFF", False, stream="video")]},
        "bad-text": {"request_id": "r", "append": {"text": 3}},
        "empty": {"request_id": "r", "append": {}},
    }[case]
    messages = [{"request_id": "r", "text": "start", "end_input": False}, append]
    (py, (ps, pa)), (rust, (rs, ra)) = _both(stacks, monkeypatch, True, messages, ending=1, settle=True)
    assert (rs, ra) == (ps, pa)
    assert rust["r"] == py["r"]
    assert py["r"][-1]["error"].startswith("append rejected: ") and py["r"][-1]["status"] == 400


def test_live_call_refusals_read_as_pythons(stacks, monkeypatch):
    messages = [
        {"text": "a live call with no id", "end_input": False},
        {"request_id": "nobody", "append": {"text": "x"}},
        {"request_id": "once", "text": "x", "end_input": True},
        # an opening is one-shot unless it says otherwise, so an empty one is refused, with a hint
        {"request_id": "empty"},
        {"request_id": "late", "text": "start", "end_input": False},
        {"request_id": "late", "append": {"text": "too late"}, "end_input": True},
    ]
    (py, (ps, pa)), (rust, (rs, ra)) = _both(stacks, monkeypatch, True, messages, ending=6, settle=True)
    assert (rs, ra) == (ps, pa)
    for rid in ("None", "nobody", "once", "empty"):
        assert rust.get(rid) == py.get(rid), rid
    assert py["empty"][0]["error"].endswith("(open a live call with end_input: false)")
    # the late notice and the call's own replies race on either frontend
    as_set = lambda replies: sorted(json.dumps(r, sort_keys=True) for r in replies)  # noqa: E731
    assert as_set(rust["late"]) == as_set(py["late"])
    assert {"request_id": "late", "append": True, "error": "the model's text stream has ended"} in py["late"]


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("path", ["/", "/generate/ws", "/v1/realtime"])
def test_a_plain_request_to_a_socket_route_reads_as_pythons(stacks, monkeypatch, method, path):
    server, port = stacks(False)
    monkeypatch.setattr(entrypoint, "api_server", server)
    py = TestClient(entrypoint.app).request(method, path)
    body = b"" if method == "POST" else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=body)
    try:
        with urllib.request.urlopen(request, timeout=15) as r:
            rust = (r.status, json.loads(r.read()))
    except urllib.error.HTTPError as e:
        rust = (e.code, json.loads(e.read()))
    assert rust == (py.status_code, py.json())



def test_a_client_that_closes_gets_the_close_handshake(stacks):
    """A client's Close is answered with one, as uvicorn answers it; without the
    answer the client sees the connection drop (1006)."""
    _, port = stacks(False)
    ws = websockets.sync.client.connect(f"ws://127.0.0.1:{port}/generate/ws", compression=None)
    ws.close()
    assert ws.close_code == 1000


# what uvicorn (websockets) closes a socket with when a frame passes ws_max_message_mb
TOO_BIG = (1009, "frame with 2000 bytes exceeds limit of 1000 bytes")


def test_a_message_past_the_limit_closes_the_socket_as_uvicorn_does():
    with _rust(_Server(Path(tempfile.mkdtemp(prefix="mstar_genws_up_")), ()), MSTAR_WS_MAX_BYTES="1000") as port:
        url = f"ws://127.0.0.1:{port}/generate/ws"
        with websockets.sync.client.connect(url, compression=None, max_size=None) as ws:
            ws.send("x" * 2000)
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                ws.recv(timeout=10)
    assert (closed.value.rcvd.code, closed.value.rcvd.reason) == TOO_BIG
