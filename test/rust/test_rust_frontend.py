"""Wire-contract test for the Rust frontend: the real
``mstar-server`` binary + the real ``RustFrontendBridge``, with ``APIServer``
stubbed — proves the HTTP surface, the msgpack bridge protocol, and the
error path end to end without GPUs. Skipped unless the ``mstar_rust``
extension is installed and the server binary is built
(``cargo build --release`` in ``rust/server/``)."""
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
            "mk": model_kwargs,
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


def test_admission_control_returns_503_when_saturated():
    import os
    import subprocess

    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_rf_sat_")
    upload_dir = tempfile.mkdtemp(prefix="mstar_rf_satup_")
    env = dict(os.environ, MSTAR_MAX_CONCURRENT_REQUESTS="0")
    proc = subprocess.Popen(
        [BINARY, "qwen3_omni", str(port), bridge_dir, upload_dir], env=env)
    stub = _StubAPIServer()
    bridge = RustFrontendBridge(stub, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        # health bypasses admission: alive even at zero capacity
        assert _wait_healthy(port)
        with pytest.raises(urllib.error.HTTPError) as e:
            _chat(port, "hi", timeout=10)
        assert e.value.code == 503
    finally:
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
