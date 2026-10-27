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
import types
import urllib.error
import urllib.request

import pytest

pytest.importorskip("mstar_rust")

from mstar.api_server.rust_frontend import (
    OPENAI_ADAPTER_ENV,
    RustFrontendBridge,
    find_server_binary,
    launch_rust_server,
    openai_adapter_class,
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
        # APIServer.metrics: None until --enable-metrics turns them on.
        self.metrics = None

    def submit_request(self, *, text=None, file_paths=None,
                       input_modalities, output_modalities,
                       model_kwargs=None, prompt_parts=None, streaming=True,
                       request_id=None):
        if text and "boom" in text:
            raise ValueError("scripted ingest failure")
        self.submitted.append({
            "rid": request_id, "text": text,
            "in": input_modalities, "out": output_modalities,
            "mk": model_kwargs, "files": file_paths, "parts": prompt_parts,
            "streaming": streaming,
        })
        return request_id

    async def iter_result_chunks(self, request_id):
        yield _Chunk("text", b"Hello ")
        yield _Chunk("text", b"world")

    def abort_request(self, request_id):
        self.aborted.append(request_id)

    def submission_key(self, request_id):
        return request_id

    def abort_submission(self, key):
        if key is not None:
            self.aborted.append(key)


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


@contextlib.contextmanager
def _running_stack(api_key=None):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_rf_test_")
    upload_dir = tempfile.mkdtemp(prefix="mstar_rf_up_")
    proc = launch_rust_server(BINARY, "qwen3_omni", port, bridge_dir,
                              upload_dir, api_key=api_key)
    stub = _StubAPIServer()
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    # /health is DEEP: passing this gate proves the ping/pong round-trip.
    if not _wait_healthy(port):
        proc.terminate()
        raise RuntimeError("server never became healthy")
    try:
        yield port, stub, bridge, proc
    finally:
        bridge.stop()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


@pytest.fixture()
def stack():
    with _running_stack() as s:
        yield s


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


def test_a_chats_roles_reach_the_bridge(stack):
    """The chat renders as its own turns, as it does on the default server."""
    port, stub, _bridge, _proc = stack
    _post_json(port, "/v1/chat/completions", {"model": "qwen3_omni", "messages": [
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Another."},
    ]})
    (sub,) = stub.submitted
    assert [(p.role, p.text) for p in sub["parts"] or []] == [
        ("user", "Name a color."), ("assistant", "Blue."), ("user", "Another."),
    ], "the Rust frontend joined the chat into one user turn"


def test_a_single_user_message_arrives_as_one_user_part(stack):
    port, stub, _bridge, _proc = stack
    _chat(port, "Say hello.")
    (sub,) = stub.submitted
    assert [(p.modality, p.role, p.text) for p in sub["parts"] or []] == [
        ("text", "user", "Say hello."),
    ], "a one-message chat did not arrive as the one user turn it renders as"


def test_generate_arrives_without_parts(stack):
    """``/generate`` has no messages, so the model takes its legacy layout."""
    port, stub, _bridge, _proc = stack
    _generate(port, urlencoded=True)
    (sub,) = stub.submitted
    assert sub["parts"] is None, "a /generate submit carried chat parts"


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


def _get(port, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def _without_status_age(text: bytes) -> list[bytes]:
    return [line for line in text.splitlines()
            if not line.startswith(b"mstar_conductor_status_age_seconds{")]


def test_metrics_relay_the_api_server_registry_open_like_health():
    """``/metrics`` on ``mstar-server`` is the text the API server's registry
    renders, with its content type, and needs no key while the request
    routes keep it."""
    pytest.importorskip("prometheus_client")
    from mstar.observability.metrics import ServingMetrics
    from mstar.profile.format import RequestTiming

    with _running_stack(api_key="s3cret") as (port, stub, _bridge, _proc):
        stub.metrics = ServingMetrics(
            "qwen3_omni", in_flight=lambda: 3, data_worker_queue_depth=lambda: 0,
        )
        now = time.perf_counter()
        timing = RequestTiming(recv_time=now - 0.4, first_chunk_time=now - 0.3)
        stub.metrics.record(types.SimpleNamespace(
            output_modalities=["text"], profile=types.SimpleNamespace(timing=timing),
            first_chunk_times={"text": now - 0.3},
        ), "success")
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            status, resp_headers, body = _get(port, "/metrics", headers)
            assert status == 200, body
            assert resp_headers["Content-Type"] == stub.metrics.content_type
            # The status age is computed at scrape, so it differs between renders.
            assert _without_status_age(body) == _without_status_age(stub.metrics.render())
        from prometheus_client.parser import text_string_to_metric_families

        samples = {(s.name, tuple(sorted(s.labels.items()))): s.value
                   for f in text_string_to_metric_families(body.decode()) for s in f.samples}
        model = ("model_name", "qwen3_omni")
        assert samples[("mstar_requests_in_flight", (model,))] == 3
        assert samples[("mstar_requests_total", (model, ("outcome", "success"),
                                                 ("output_modalities", "text")))] == 1
        assert samples[("mstar_e2e_request_latency_seconds_count",
                        (model, ("output_modalities", "text")))] == 1
        assert _get(port, "/v1/models")[0] == 401


def test_metrics_are_404_when_off_and_503_when_the_bridge_dies(stack):
    port, _stub, bridge, _proc = stack
    status, _, body = _get(port, "/metrics")
    assert status == 404
    assert "--enable-metrics" in json.loads(body)["detail"]
    bridge.stop()
    time.sleep(0.1)  # let the loop exit
    assert _get(port, "/metrics")[0] == 503


def test_sigterm_drains_and_exits(stack):
    import signal

    port, _stub, _bridge, proc = stack
    assert _chat(port, "warm")["choices"][0]["message"]["content"]
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=15) is not None


class _GatedStub(_StubAPIServer):
    """Sends the first chunk, then holds each request open until a shared gate
    is released, so a request's admission permit is observably held for its
    whole (streaming) body."""

    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    async def iter_result_chunks(self, request_id):
        import asyncio
        yield _Chunk("text", b"Hello ")
        while not self.gate.is_set():
            await asyncio.sleep(0.02)
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
    env = dict(os.environ, MSTAR_MAX_CONCURRENT_REQUESTS="1",
               **{OPENAI_ADAPTER_ENV: openai_adapter_class("qwen3_omni")})
    proc = subprocess.Popen(
        [BINARY, "qwen3_omni", str(port), bridge_dir, upload_dir], env=env)
    stub = _GatedStub(gate)
    # What the bridge reads of APIServer.metrics.
    stub.metrics = types.SimpleNamespace(
        render=lambda: b"mstar_up 1\n", content_type="text/plain; version=0.0.4",
        refusals=types.SimpleNamespace(update_rust=lambda routes, statuses, counts: None))
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    resp_a = None
    try:
        assert _wait_healthy(port)
        # A: a streaming chat. Its first delta flushes immediately; the body
        # then blocks on the gate, so A holds its permit open.
        body = {"model": "qwen3_omni", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        req_a = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        resp_a = urllib.request.urlopen(req_a, timeout=15)
        # Not just any line: an SSE keep-alive comment would also be one.
        assert b"Hello" in resp_a.readline()  # first delta: A is being served
        # B: A holds the only permit, so B is refused with 503.
        with pytest.raises(urllib.error.HTTPError) as e:
            _chat(port, "second", timeout=5)
        assert e.value.code == 503
        # A saturated server is still scrapeable: /metrics bypasses the cap.
        assert _get(port, "/metrics")[:3:2] == (200, b"mstar_up 1\n")
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


def _status(port, method, path, headers=None, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method, data=body,
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.headers  # case-insensitive lookups
    except urllib.error.HTTPError as e:
        return e.code, e.headers


def test_api_key_guards_every_route_but_health(monkeypatch):
    """The launcher hands the key to mstar-server, which then refuses every
    route but /health (passing ``_wait_healthy`` keyless proves that one)
    without ``Authorization: Bearer <key>``."""
    key = "s3cret-key"
    # The launcher passes the resolved key, not whatever the parent inherited.
    monkeypatch.setenv("MSTAR_API_KEY", "inherited-and-ignored")
    chat = json.dumps({"model": "qwen3_omni",
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    with _running_stack(api_key=key) as (port, stub, _bridge, _proc):
        for headers in ({}, {"Authorization": "Bearer wrong"},
                        {"Authorization": "Bearer inherited-and-ignored"},
                        {"Authorization": f"Basic {key}"}):
            assert _status(port, "GET", "/v1/models", headers)[0] == 401
            status, resp_headers = _status(port, "POST", "/v1/chat/completions", headers, chat)
            assert status == 401
            assert resp_headers.get("www-authenticate") == "Bearer"
        assert not stub.submitted, "a request without the key reached the backend"
        good = {"Authorization": f"Bearer {key}"}
        assert _status(port, "GET", "/v1/models", good)[0] == 200
        assert _status(port, "POST", "/v1/chat/completions", good, chat)[0] == 200
        assert len(stub.submitted) == 1
        # CORS answers a browser preflight, which carries no credentials.
        status, resp_headers = _status(port, "OPTIONS", "/v1/chat/completions", {
            "Origin": "https://app.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"})
        assert status == 200 and "access-control-allow-origin" in resp_headers


def test_no_api_key_ignores_an_inherited_env_key(monkeypatch):
    monkeypatch.setenv("MSTAR_API_KEY", "inherited")
    with _running_stack(api_key=None) as (port, _stub, _bridge, _proc):
        assert _status(port, "GET", "/v1/models")[0] == 200


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


def _generate(port, urlencoded: bool, fields=None, timeout=30):
    """POST /generate with either content type; returns (status, ndjson)."""
    import urllib.parse

    if fields is None:
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


# The Python /generate's 400s (test_video_frame_protocol checks the same text there).
_STREAMING_ONLY = ("Output modality video_frame requires streaming=true; raw frame "
                   "chunks cannot be returned as an aggregated response.")
_OUTPUT_ONLY = "'video_frame' is an output-only modality"


@pytest.mark.parametrize("fields,detail", [
    ({"output_modalities": "video_frame", "streaming": "false"}, _STREAMING_ONLY),
    ({"input_modalities": "video_frame", "output_modalities": "text"}, _OUTPUT_ONLY),
], ids=["output-not-streamed", "input"])
def test_generate_refuses_video_frame_with_a_400_before_submitting(stack, fields, detail):
    """The API server's own check would raise a ValueError, which reaches
    the client as a 500."""
    import urllib.parse

    port, stub, _bridge, _proc = stack
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=urllib.parse.urlencode({"text": "hi", **fields}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=15)
    assert (e.value.code, json.loads(e.value.read())) == (400, {"detail": detail})
    assert stub.submitted == []


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


@pytest.mark.parametrize("urlencoded", [False, True])
def test_generate_treats_an_empty_request_id_as_none(stack, urlencoded):
    """Like the Python /generate: each request gets a fresh id, so two with
    ``request_id=""`` never collide on ``""``."""
    port, stub, _bridge, _proc = stack
    for streaming in ("true", "false"):
        fields = {"text": "hi", "streaming": streaming, "request_id": ""}
        assert _generate(port, urlencoded, fields)[0] == 200
    rids = [s["rid"] for s in stub.submitted]
    assert len(rids) == 2 and "" not in rids and rids[0] != rids[1]


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
def _model_stack(model, stub, env=None):
    """Launch the real mstar-server for `model` wired to `stub`."""
    import os
    import subprocess

    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_rf_ms_")
    upload_dir = tempfile.mkdtemp(prefix="mstar_rf_msup_")
    env = {OPENAI_ADAPTER_ENV: openai_adapter_class(model), **(env or {})}
    proc = subprocess.Popen(
        [BINARY, model, str(port), bridge_dir, upload_dir], env=dict(os.environ, **env))
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


def _server_file_refs():
    secret = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    secret.write(b"\x89PNG\r\n\x1a\n")
    secret.close()
    return [secret.name, "../" * 8 + secret.name.lstrip("/"), f"file://{secret.name}"]


def test_chat_refuses_server_file_paths(stack):
    port, stub, _bridge, _proc = stack
    for ref in _server_file_refs():
        body = {"model": "qwen3_omni", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": ref}}]}]}
        status, out = _post_raw(port, "/v1/chat/completions", json.dumps(body).encode())
        assert status == 400, (ref, status, out)
    assert not stub.submitted


def test_video_conditioning_refuses_server_file_paths():
    stub = _VideoStubAPIServer()
    with _model_stack("cosmos3", stub) as (port, _up):
        for ref in _server_file_refs():
            status, _ = _post_raw(port, "/v1/videos/generations", json.dumps(
                {"model": "cosmos3", "prompt": "x", "image": ref}).encode())
            assert status == 400, ref
    assert not stub.submitted


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


class _MidStreamErrorStub(_StubAPIServer):
    """A text chunk, then the data plane's terminal error chunk."""

    def __init__(self, status):
        super().__init__()
        self.status = status

    async def iter_result_chunks(self, request_id):
        yield _Chunk("text", b"Hello ")
        yield _Chunk("error", b"Error in worker: boom", {"status": self.status})


@pytest.mark.parametrize("status", [400, 500])
def test_a_failed_generate_stream_ends_on_the_same_line_on_both_frontends(status, monkeypatch):
    """The Python /generate ends the stream with the error chunk itself, its
    status in ``metadata``, which the SDK raises on; mstar-server must send the
    same line. Both frontends encode the same chunks."""
    pytest.importorskip("fastapi")
    import base64
    import urllib.parse

    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    stub = _MidStreamErrorStub(status)

    class _PythonServer(entrypoint.APIServer):
        """The real /generate stream encoding over the stub's chunks."""

        def __init__(self):
            self.enable_nvtx = False

        async def wait_for_request_id(self, request_id):
            pass

        def submit_request(self, **kwargs):
            return kwargs["request_id"]

        def iter_result_chunks(self, request_id):
            return stub.iter_result_chunks(request_id)

    form = {"text": "hi", "output_modalities": "text", "streaming": "true"}
    monkeypatch.setattr(entrypoint, "api_server", _PythonServer())
    py = TestClient(entrypoint.app).post("/generate", data=form)
    with _model_stack("qwen3_omni", stub) as (port, _up):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate", data=urllib.parse.urlencode(form).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=15) as r:
            rust_status, rust_lines = r.status, [json.loads(line) for line in r.read().splitlines()]

    assert (rust_status, rust_lines) == (py.status_code, [json.loads(line) for line in py.content.splitlines()])
    assert rust_lines[-1] == {
        "modality": "error", "data": base64.b64encode(b"Error in worker: boom").decode(),
        "metadata": {"status": status},
    }


_EDIT_BOUNDARY = "----editboundary"
_EDIT_FORM = (
    f'--{_EDIT_BOUNDARY}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nmake it neon\r\n'
    f'--{_EDIT_BOUNDARY}\r\nContent-Disposition: form-data; name="image"; filename="in.png"\r\n'
    f"Content-Type: image/png\r\n\r\nPNG\r\n--{_EDIT_BOUNDARY}--\r\n"
).encode()


def _error_of(port, path, body):
    """POST ``body`` (JSON, or the image edit's form as bytes); return the
    OpenAI error object, from the stream's error event when it streams."""
    if isinstance(body, bytes):
        data, ctype = body, f"multipart/form-data; boundary={_EDIT_BOUNDARY}"
    else:
        data, ctype = json.dumps(body).encode(), "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            lines = r.read().decode().splitlines()
            return [json.loads(ln[6:]) for ln in lines if ln.startswith("data: {")][-1]["error"]
    except urllib.error.HTTPError as e:
        error = json.loads(e.read())["error"]
        assert e.code == error["code"], (e.code, error)
        return error


_CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def test_body_over_limit_is_413_like_the_python_frontend(monkeypatch):
    """MSTAR_MAX_BODY_MB: a JSON body and a multipart upload over the limit
    both get a 413 naming it, as `mstar/api_server/body_limit.py` answers
    (axum's own rejections were a 422 and a 400)."""
    monkeypatch.setenv("MSTAR_MAX_BODY_MB", "1")
    big = b"x" * (2 * 2**20)
    message = "request body is larger than the 1 MiB limit (MSTAR_MAX_BODY_MB)"

    def post(port, path, data, ctype):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": ctype})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        return e.value.code, json.loads(e.value.read())

    boundary = "----limitboundary"

    def form(name, filename):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode() \
            + big + f"\r\n--{boundary}--\r\n".encode()

    ctype = f"multipart/form-data; boundary={boundary}"
    with _model_stack("bagel", _ImageStubAPIServer()) as (port, _up):
        chat = json.dumps({"model": "bagel", "messages": [{"role": "user", "content": "a" * len(big)}]})
        code, body = post(port, "/v1/chat/completions", chat.encode(), "application/json")
        assert code == 413 and body["error"]["message"] == message
        code, body = post(port, "/v1/images/edits", form("image", "a.png"), ctype)
        assert code == 413 and body["error"]["message"] == message
        code, body = post(port, "/generate", form("files", "a.wav"), ctype)
        assert code == 413 and body["detail"] == message


@pytest.mark.parametrize("status,error_type", [(400, "invalid_request_error"), (500, "server_error")])
@pytest.mark.parametrize("model,path,body", [
    ("qwen3_omni", "/v1/chat/completions", {"model": "qwen3_omni", **_CHAT}),
    ("qwen3_omni", "/v1/chat/completions", {"model": "qwen3_omni", **_CHAT, "stream": True}),
    ("qwen3_omni", "/v1/audio/speech", {"model": "qwen3_omni", "input": "hi"}),
    ("cosmos3", "/v1/images/generations", {"model": "cosmos3", "prompt": "a cat"}),
    ("cosmos3", "/v1/videos/generations", {"model": "cosmos3", "prompt": "a cat"}),
    ("bagel", "/v1/images/edits", _EDIT_FORM),
], ids=["chat", "chat-stream", "speech", "images", "videos", "edits"])
def test_a_backend_error_is_typed_by_its_status(model, path, body, status, error_type):
    """A 400 the backend reports (a ``ValueError`` in the data worker) is the
    request's fault, as the Python server types it; ``server_error`` says the
    opposite."""
    stub = _ErrorChunkStub(status)
    with _model_stack(model, stub) as (port, _up):
        error = _error_of(port, path, body)
    assert stub.submitted, "the frontend refused the request before the backend saw it"
    assert (error["code"], error["type"]) == (status, error_type), "the error type ignores the status"


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


class _RefusedError(Exception):
    """An ``HTTPException`` without fastapi (the CI Rust job installs M* bare)."""

    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


class _KeyedStub(_StubAPIServer):
    """Submissions as ``APIServer`` runs them: each under a key of its own, a
    stream bound to the newest submission under its id when it is created,
    ``abort_request`` acting on the newest, ``abort_submission`` on one, and a
    submission while the previous one under the id still runs refused with a
    409. Each submission streams ``tokens`` text chunks, a few ms apart,
    unless aborted."""

    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.n = 0
        self.live = {}
        self.running = set()
        self.refused = 0

    def submit_request(self, *, request_id=None, **kwargs):
        if self.live.get(request_id) in self.running:
            self.refused += 1
            raise _RefusedError(409, f"request_id {request_id!r} already has a request in flight")
        self.n += 1
        key = f"{request_id}#{self.n}"
        self.live[request_id] = key
        self.running.add(key)
        return request_id

    def submission_key(self, request_id):
        return self.live.get(request_id)

    def iter_result_chunks(self, request_id):
        return self._stream(self.live[request_id])

    async def _stream(self, key):
        import asyncio

        for _ in range(self.tokens):
            if key not in self.running:
                return
            yield _Chunk("text", b"t")
            await asyncio.sleep(0.003)
        self.running.discard(key)

    def abort_request(self, request_id):
        self.abort_submission(self.live.get(request_id))

    def abort_submission(self, key):
        if key in self.running:
            self.running.discard(key)
            self.aborted.append(key)


def _generate_form(streaming=True):
    import urllib.parse

    return urllib.parse.urlencode({
        "text": "count", "request_id": "X", "streaming": str(streaming).lower(),
        "output_modalities": "text"}).encode()


def test_a_retry_right_after_a_dropped_stream_is_served():
    """The client reads 3 chunks of a /generate stream under X, drops it and
    resubmits X at once on a connection it opened earlier, before the frontend
    has noticed the drop. The retry must get its whole answer, not a 409, and
    so must the next reuse of X once that answer is read, round after round:
    the frontend waits for the drop and sends its abort before the retry's
    submit."""
    import http.client

    stub = _KeyedStub(tokens=200)
    form = _generate_form()

    def answer(conn):
        conn.request("POST", "/generate", body=form,
                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        r = conn.getresponse()
        lines = [json.loads(line) for line in r.read().splitlines() if line.strip()]
        return [line.get("modality") for line in lines], lines[-1:]

    with _model_stack("qwen3_omni", stub) as (port, _up):
        url = f"http://127.0.0.1:{port}/generate"
        for _ in range(5):
            warm = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            warm.connect()
            with urllib.request.urlopen(url, data=form, timeout=10) as r:
                for _ in range(3):
                    assert r.readline()
            for request in ("retry", "reuse"):
                modalities, last = answer(warm)
                assert modalities == ["text"] * 200, (
                    f"the {request} got {len(modalities)} chunks, ending {last}")
            warm.close()
    dropped = [f"X#{n}" for n in (1, 4, 7, 10, 13)]
    assert stub.aborted == dropped, "a retry or reuse was cut short"
    assert stub.refused == 0


@pytest.mark.parametrize("streaming", [True, False])
def test_a_duplicate_of_a_running_id_gets_a_409_and_the_running_one_completes(streaming):
    """A second /generate under X while the first still streams is answered
    with an HTTP 409 by the frontend itself, after its grace; the first is
    not aborted and gets every chunk."""
    stub = _KeyedStub(tokens=400)

    with _model_stack("qwen3_omni", stub) as (port, _up):
        url = f"http://127.0.0.1:{port}/generate"
        with urllib.request.urlopen(url, data=_generate_form(), timeout=10) as r:
            head = [r.readline() for _ in range(3)]
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(url, data=_generate_form(streaming), timeout=10)
            lines = head + r.read().splitlines()

    assert refused.value.code == 409
    assert json.loads(refused.value.read()) == {
        "detail": "request_id 'X' already has a request in flight; "
                  "wait for it to finish or abort it first"}
    assert [json.loads(line)["modality"] for line in lines if line.strip()] == ["text"] * 400
    assert stub.n == 1 and stub.refused == 0, "the duplicate reached the backend"
    assert stub.aborted == []


class _GatedKeyedStub(_KeyedStub):
    """``_KeyedStub`` whose streams wait for ``gate``; a prompt of ``fail``
    gets the backend's 400 once it was submitted."""

    def __init__(self, gate):
        super().__init__(tokens=1)
        self.gate = gate
        self.texts = {}

    def submit_request(self, *, request_id=None, text=None, **kwargs):
        request_id = super().submit_request(request_id=request_id, text=text, **kwargs)
        self.texts[self.live[request_id]] = text
        return request_id

    async def _stream(self, key):
        import asyncio

        if self.texts[key] == "fail":
            self.running.discard(key)
            yield _Chunk("error", b"bad input", {"status": 400})
            return
        while not self.gate.is_set():
            await asyncio.sleep(0.01)
        async for chunk in super()._stream(key):
            yield chunk


def test_refusals_reach_the_api_server_registry():
    """Requests ``mstar-server`` refuses before submitting them (no key, body
    over the limit, invalid JSON, a bad form, a ``request_id`` in flight,
    the admission cap, an unknown route, a wrong method) are counted by route
    template and status and exported by the API server's registry through
    the bridge; a request the backend fails after submission is not."""
    pytest.importorskip("prometheus_client")
    import urllib.parse

    from mstar.observability.metrics import ServingMetrics

    key = {"Authorization": "Bearer s3cret"}
    gate = threading.Event()
    stub = _GatedKeyedStub(gate)
    stub.metrics = ServingMetrics("qwen3_omni", in_flight=lambda: 0, data_worker_queue_depth=lambda: 0)
    env = {"MSTAR_API_KEY": "s3cret", "MSTAR_MAX_CONCURRENT_REQUESTS": "2", "MSTAR_MAX_BODY_MB": "1"}
    chat = json.dumps({"model": "qwen3_omni", "messages": [{"role": "user", "content": "hi"}]}).encode()

    def call(method, path, body=None, headers=key, ctype="application/json"):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=body,
                                     headers={"Content-Type": ctype, **headers})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def generate(request_id, text="hi", streaming="true", **fields):
        return urllib.parse.urlencode({"text": text, "request_id": request_id,
                                       "streaming": streaming, **fields}).encode()

    def open_stream(request_id):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=generate(request_id),
                                     headers={"Content-Type": "application/x-www-form-urlencoded", **key})
        return urllib.request.urlopen(req, timeout=15)

    form = "application/x-www-form-urlencoded"
    with _model_stack("qwen3_omni", stub, env=env) as (port, _up):
        assert call("POST", "/v1/chat/completions", chat, headers={}) == 401
        assert call("GET", "/no/such/route", headers={}) == 401
        assert call("GET", "/no/such/route") == 404
        assert call("GET", "/generate") == 405
        assert call("POST", "/v1/chat/completions", b"x" * 2**21) == 413
        assert call("POST", "/v1/chat/completions", b"{not json") == 422
        assert call("POST", "/generate", generate("B", model_kwargs="{"), ctype=form) == 400
        assert call("POST", "/generate", generate("F", text="fail", streaming="false"), ctype=form) == 400
        streams = [open_stream("X")]
        try:
            assert call("POST", "/generate", generate("X"), ctype=form) == 409
            streams.append(open_stream("Y"))
            assert call("POST", "/v1/chat/completions", chat) == 503
        finally:
            gate.set()
            for s in streams:
                s.read()
                s.close()
        status, _, body = _get(port, "/metrics")

    assert status == 200, body
    from prometheus_client.parser import text_string_to_metric_families

    refusals = {(s.labels["route"], s.labels["status"]): s.value
                for f in text_string_to_metric_families(body.decode()) for s in f.samples
                if s.name == "mstar_http_refusals_total" and s.labels["model_name"] == "qwen3_omni"}
    assert refusals == {
        ("/v1/chat/completions", "401"): 1, ("unmatched", "401"): 1, ("unmatched", "404"): 1,
        ("/generate", "405"): 1, ("/v1/chat/completions", "413"): 1, ("/v1/chat/completions", "422"): 1,
        ("/generate", "400"): 1, ("/generate", "409"): 1, ("/v1/chat/completions", "503"): 1,
    }
    assert stub.n == 3, "a refused request reached the backend"


def _image_n_error(port, n):
    return _error_of(port, "/v1/images/generations", {"model": "cosmos3", "prompt": "x", "n": n})


def test_images_n_outside_one_to_the_cap_is_400_before_submit():
    """`n` below 1 or above the cap (default 10) is the Python frontend's 400,
    and nothing reaches the backend; `n` at the cap still returns that many."""
    stub = _ImageStubAPIServer()
    with _model_stack("cosmos3", stub) as (port, _up):
        for n in (0, -1, 11):
            assert _image_n_error(port, n) == {
                "message": f"n must be between 1 and 10, got {n}",
                "type": "invalid_request_error", "code": 400}
        assert not stub.submitted
        out = _post_json(port, "/v1/images/generations",
                         {"model": "cosmos3", "prompt": "x", "n": 10})
        assert len(out["data"]) == 10 and len(stub.submitted) == 10


def test_images_n_cap_follows_the_env():
    stub = _ImageStubAPIServer()
    with _model_stack("cosmos3", stub, env={"MSTAR_MAX_IMAGES_PER_REQUEST": "2"}) as (port, _up):
        assert _image_n_error(port, 3)["message"] == "n must be between 1 and 2, got 3"
        assert not stub.submitted
        out = _post_json(port, "/v1/images/generations",
                         {"model": "cosmos3", "prompt": "x", "n": 2})
        assert len(out["data"]) == 2
