"""Wire-contract test for the Rust frontend: the real
``mstar-server`` binary + the real ``RustFrontendBridge``, with ``APIServer``
stubbed — proves the HTTP surface, the msgpack bridge protocol, and the
error path end to end without GPUs. Skipped unless the ``mstar_rust``
extension is installed and the server binary is built
(``cargo build --release`` in ``rust/server/``)."""
import contextlib
import json
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("mstar_rust")

from mstar.api_server.rust_frontend import (
    RustFrontendBridge,
    find_server_binary,
    launch_rust_server,
)

try:
    BINARY = find_server_binary()
except FileNotFoundError:
    BINARY = None

pytestmark = pytest.mark.skipif(
    BINARY is None, reason="mstar-server binary not built")


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality = modality
        self.data = data
        self.metadata = metadata or {}


class _StubAPIServer:
    """Scripted data plane: echoes the prompt back in two text chunks."""

    def __init__(self):
        self.submitted = []
        self.aborted = []

    def submit_request(self, *, text=None, file_paths=None,
                       input_modalities, output_modalities,
                       model_kwargs=None, streaming=True, request_id=None):
        if text and "boom" in text:
            raise ValueError("scripted ingest failure")
        self.submitted.append({
            "rid": request_id, "text": text,
            "in": input_modalities, "out": output_modalities,
            "mk": model_kwargs, "files": file_paths,
            "streaming": streaming,
        })
        return request_id

    async def iter_result_chunks(self, request_id):
        yield _Chunk("text", b"Hello ")
        yield _Chunk("text", b"world")

    def abort_request(self, request_id):
        self.aborted.append(request_id)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_healthy(port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3)
            return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture()
def stack():
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_rf_test_")
    upload_dir = tempfile.mkdtemp(prefix="mstar_rf_up_")
    proc = launch_rust_server(BINARY, "qwen3_omni", port, bridge_dir,
                              upload_dir)
    stub = _StubAPIServer()
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    # /health is DEEP: passing this gate proves the ping/pong round-trip.
    if not _wait_healthy(port):
        proc.terminate()
        raise RuntimeError("server never became healthy")
    yield port, stub, bridge, proc
    bridge.stop()
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)


def _chat(port, content, timeout=30):
    body = {"model": "qwen3_omni", "messages":
            [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def test_chat_roundtrip(stack):
    port, stub, _bridge, _proc = stack
    out = _chat(port, "Say hello.")
    assert out["choices"][0]["message"]["content"] == "Hello world"
    (sub,) = stub.submitted
    assert "Say hello." in sub["text"]
    assert sub["out"] == ["text"]


def _post_raw(port, path, raw_bytes, timeout=15):
    """POST a raw body (possibly malformed) as application/json; return
    (status, parsed_json_or_None)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=raw_bytes,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, None


def test_malformed_json_body_is_422_with_detail(stack):
    """A malformed / schema-invalid JSON body returns FastAPI's 422 +
    {"detail": [{...}]} shape, not axum's default 400 + text/plain."""
    port, _stub, _bridge, _proc = stack
    # Broken JSON syntax.
    code, body = _post_raw(port, "/v1/chat/completions", b"{not json")
    assert code == 422, (code, body)
    assert isinstance(body["detail"], list) and body["detail"]
    # Missing required field `role` (Python's pydantic ChatMessage requires it).
    code, body = _post_raw(
        port, "/v1/chat/completions",
        json.dumps({"model": "qwen3_omni",
                    "messages": [{"content": "hi"}]}).encode())
    assert code == 422, (code, body)
    assert isinstance(body["detail"], list) and body["detail"]


def test_ingest_failure_is_a_500_not_a_hang(stack):
    port, stub, _bridge, _proc = stack
    with pytest.raises(urllib.error.HTTPError) as e:
        _chat(port, "boom", timeout=15)
    assert e.value.code == 500
    # and the server keeps serving afterwards
    assert _chat(port, "again")["choices"][0]["message"]["content"] == \
        "Hello world"


def test_health_goes_red_when_the_bridge_dies(stack):
    port, _stub, bridge, _proc = stack
    bridge.stop()
    time.sleep(0.1)  # let the loop exit
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10)
    assert e.value.code == 503


def test_sigterm_drains_and_exits(stack):
    import signal

    port, _stub, _bridge, proc = stack
    assert _chat(port, "warm")["choices"][0]["message"]["content"]
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=15) is not None


class _GatedStub(_StubAPIServer):
    """Holds each request open until a shared gate is released, so a request's
    admission permit is observably held for its whole (streaming) body."""

    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    async def iter_result_chunks(self, request_id):
        import asyncio
        while not self.gate.is_set():
            await asyncio.sleep(0.02)
        yield _Chunk("text", b"Hello ")
        yield _Chunk("text", b"world")


def test_admission_permit_is_held_through_streaming_body():
    """The concurrency cap must bound *streaming* requests for their entire
    body, not just until the Response is constructed. With cap=1, an open
    streaming request holds the only permit, so a concurrent request 503s until
    it finishes — then the freed permit lets a later request through. The
    original permit-released-at-construction bug would let the concurrent
    request straight in (a cap=0 test would pass either way, so can't catch it)."""
    import os
    import subprocess

    gate = threading.Event()
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_rf_adm_")
    upload_dir = tempfile.mkdtemp(prefix="mstar_rf_admup_")
    env = dict(os.environ, MSTAR_MAX_CONCURRENT_REQUESTS="1")
    proc = subprocess.Popen(
        [BINARY, "qwen3_omni", str(port), bridge_dir, upload_dir], env=env)
    stub = _GatedStub(gate)
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    resp_a = None
    try:
        assert _wait_healthy(port)
        # A: a streaming chat. The role-delta head event flushes immediately;
        # the body then blocks on the gate, so A holds its permit open.
        body = {"model": "qwen3_omni", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        req_a = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        resp_a = urllib.request.urlopen(req_a, timeout=15)
        assert resp_a.readline()  # head event received: A is being served
        # B: A holds the only permit, so B is refused with 503.
        with pytest.raises(urllib.error.HTTPError) as e:
            _chat(port, "second", timeout=5)
        assert e.value.code == 503
        # Release A; its permit frees and a subsequent request succeeds.
        gate.set()
        assert b"world" in resp_a.read()
        assert _chat(port, "third")["choices"][0]["message"]["content"] == \
            "Hello world"
    finally:
        gate.set()
        if resp_a is not None:
            resp_a.close()
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


def test_tokenizer_env_fails_fast(tmp_path, monkeypatch):
    """MSTAR_TOKENIZER set -> the bridge refuses at construction (the model
    side owns tokenization), instead of 500-ing every request."""
    from mstar.api_server.rust_frontend import RustFrontendBridge

    monkeypatch.setenv("MSTAR_TOKENIZER", "/some/tokenizer.json")
    with pytest.raises(RuntimeError, match="MSTAR_TOKENIZER"):
        RustFrontendBridge(_StubAPIServer(), str(tmp_path))


def test_upload_filename_traversal_is_contained(tmp_path):
    """A multipart filename with `../` must not escape the upload dir
    (arbitrary write, amplified to arbitrary delete by cleanup)."""
    import os

    from mstar.api_server import media_io

    # save_base64's format field is client-controlled: a traversal format
    # must be sanitized to a plain extension, keeping the file in upload_dir.
    modality, path = media_io.save_base64(
        "AAAA", "../../../../tmp/evil", "audio", tmp_path)
    resolved = os.path.realpath(path)
    assert resolved.startswith(os.path.realpath(str(tmp_path)) + os.sep), path
    assert ".." not in os.path.basename(path)


def _generate(port, urlencoded: bool, timeout=30):
    """POST /generate with either content type; returns (status, ndjson)."""
    import urllib.parse

    fields = {"text": "hi", "output_modalities": "text",
              "streaming": "false", "model_kwargs": '{"foo": 1}'}
    if urlencoded:
        data = urllib.parse.urlencode(fields).encode()
        ctype = "application/x-www-form-urlencoded"
    else:
        boundary = "----genboundary"
        data = ("".join(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{k}"\r\n\r\n{v}\r\n' for k, v in fields.items())
            + f"--{boundary}--\r\n").encode()
        ctype = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=data, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def test_generate_accepts_both_content_types(stack):
    """The FastAPI /generate accepts multipart AND urlencoded form bodies
    (the latter is what `requests.post(data=...)` sends with no files); the
    Rust handler must too, or file-less clients regress to 400."""
    port, stub, _bridge, _proc = stack
    for urlencoded in (False, True):
        status, _ = _generate(port, urlencoded)
        assert status == 200, ("urlencoded" if urlencoded else "multipart")
    # both landed as real submits with the text + model_kwargs parsed
    texts = [s["text"] for s in stub.submitted]
    assert texts.count("hi") == 2, stub.submitted
    assert all(s["mk"] == {"foo": 1} for s in stub.submitted)


_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42FAKE-MP4-BYTES"
_TINY_PNG_DATA_URI = "data:image/png;base64,iVBORw0KGgo="


class _VideoStubAPIServer(_StubAPIServer):
    """Data plane for Cosmos3: emits a single (fake) mp4 video chunk (plus a
    PNG image chunk for the images-on-cosmos3 surface)."""

    async def iter_result_chunks(self, request_id):
        yield _Chunk("video", _FAKE_MP4)


class _ImageStubAPIServer(_StubAPIServer):
    async def iter_result_chunks(self, request_id):
        yield _Chunk("image", b"\x89PNG\r\n\x1a\nFAKE")


@contextlib.contextmanager
def _model_stack(model, stub):
    """Launch the real mstar-server for `model` wired to `stub`."""
    import os
    import subprocess

    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_rf_ms_")
    upload_dir = tempfile.mkdtemp(prefix="mstar_rf_msup_")
    proc = subprocess.Popen(
        [BINARY, model, str(port), bridge_dir, upload_dir], env=dict(os.environ))
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        assert _wait_healthy(port)
        yield port, upload_dir
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


def _post_json(port, path, body, timeout=30):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def test_videos_generations_roundtrip():
    """Text-to-video on Cosmos3: request maps to output_modalities=["video"]
    and the video chunk comes back b64-encoded in the `b64_json` shape. Cosmos3
    was added to the adapter registry after the frontend PR opened."""
    import base64

    stub = _VideoStubAPIServer()
    with _model_stack("cosmos3", stub) as (port, _up):
        out = _post_json(port, "/v1/videos/generations", {
            "model": "cosmos3", "prompt": "a cat surfing",
            "num_frames": 24, "fps": 12.0, "guidance_scale": 7.5})
        assert base64.b64decode(out["data"][0]["b64_json"]) == _FAKE_MP4
        (sub,) = stub.submitted
        assert sub["in"] == ["text"] and sub["out"] == ["video"]
        assert not sub["files"]
        assert sub["mk"]["num_frames"] == 24 and sub["mk"]["fps"] == 12.0
        assert sub["mk"]["guidance_scale"] == 7.5


def test_videos_conditioning_branches():
    """Image-to-video and video-to-video: the conditioning reference routes
    through `resolve_media_ref` (persisted under upload_dir) and sets the right
    input modalities."""
    for field, in_mods in (("image", ["image", "text"]),
                           ("video", ["video", "text"])):
        stub = _VideoStubAPIServer()
        with _model_stack("cosmos3", stub) as (port, upload_dir):
            _post_json(port, "/v1/videos/generations", {
                "model": "cosmos3", "prompt": "x", field: _TINY_PNG_DATA_URI})
            (sub,) = stub.submitted
            assert sub["in"] == in_mods, (field, sub["in"])
            assert sub["out"] == ["video"]
            # the reference was persisted under upload_dir and passed by path
            (path,) = sub["files"][field]
            assert path.startswith(upload_dir), path


def test_videos_both_conditioning_is_400():
    """Providing both `image` and `video` conditioning is rejected (mirrors the
    Python adapter's ValueError → 400)."""
    stub = _VideoStubAPIServer()
    with _model_stack("cosmos3", stub) as (port, _up):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post_json(port, "/v1/videos/generations", {
                "model": "cosmos3", "prompt": "x",
                "image": _TINY_PNG_DATA_URI, "video": _TINY_PNG_DATA_URI})
        assert e.value.code == 400
        assert not stub.submitted  # rejected before submit


class _ErrorChunkStub(_StubAPIServer):
    """Deliver a backend failure the way the data plane does: a terminal chunk
    with modality 'error' + metadata.status that ends cleanly (no raise) — the
    path collect_results re-raises on the Python side."""

    def __init__(self, status=500):
        super().__init__()
        self.status = status

    async def iter_result_chunks(self, request_id):
        yield _Chunk("error", b"Error in worker: boom", {"status": self.status})


def _generate_nonstream(port, timeout=15):
    import urllib.parse
    data = urllib.parse.urlencode(
        {"text": "hi", "output_modalities": "text", "streaming": "false"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_nonstreaming_backend_error_maps_to_real_status():
    """A non-streaming request whose backend fails (in-band 'error' chunk with
    metadata.status) must return that HTTP status, not a 200 with the error
    silently filtered (chat/speech/images/videos) or buried in outputs
    (/generate). Regression for the 200-on-failure bug."""
    # /v1 chat: OpenAI error envelope with the real status.
    with _model_stack("qwen3_omni", _ErrorChunkStub(500)) as (port, _up):
        code, body = _post_raw(
            port, "/v1/chat/completions",
            json.dumps({"model": "qwen3_omni",
                        "messages": [{"role": "user", "content": "hi"}]}).encode())
        assert code == 500, (code, body)
        assert "boom" in body["error"]["message"]
    # /generate: FastAPI {"detail": ...}, not 200 with the error in outputs.
    with _model_stack("qwen3_omni", _ErrorChunkStub(500)) as (port, _up):
        code, body = _generate_nonstream(port)
        assert code == 500, (code, body)
        assert "boom" in body["detail"]
    # Status is carried in-band (400 vs 500 distinction), not hardcoded.
    with _model_stack("qwen3_omni", _ErrorChunkStub(400)) as (port, _up):
        code, body = _post_raw(
            port, "/v1/chat/completions",
            json.dumps({"model": "qwen3_omni",
                        "messages": [{"role": "user", "content": "hi"}]}).encode())
        assert code == 400, (code, body)


def test_images_generations_on_cosmos3():
    """Cosmos3 also serves `/v1/images/generations` (text-to-image)."""
    import base64

    stub = _ImageStubAPIServer()
    with _model_stack("cosmos3", stub) as (port, _up):
        out = _post_json(port, "/v1/images/generations",
                         {"model": "cosmos3", "prompt": "a red circle", "n": 2})
        # n=2 → two engine submits, two images aggregated
        assert len(stub.submitted) == 2
        assert len(out["data"]) == 2
        assert base64.b64decode(out["data"][0]["b64_json"]) == b"\x89PNG\r\n\x1a\nFAKE"
