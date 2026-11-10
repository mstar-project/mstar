"""LoRA on both frontends: ``/v1/models`` lists the served adapters, and a
request's ``model`` field (or its ``lora_adapter`` kwarg) selects the adapter it
runs under, or is refused with the same text. mstar-server resolves it itself
(``served_models.rs``) and submits the adapter as a ``lora_adapter`` kwarg; the
Python routes resolve ``model`` in ``submit_request`` (``ServedModels.select``),
which also checks that kwarg and adds the adapter's digest. These tests apply it
to what each frontend submitted and compare what the engine would run."""
import json
import urllib.error
import urllib.request

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _Chunk, _model_stack, _StubAPIServer
from test_rust_openai_adapters import _model_of

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import APIServer
from mstar.api_server.rust_frontend import SERVED_MODELS_ENV, served_models_json
from mstar.api_server.served_models import ServedModels

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

MODEL = "qwen3_5_0.8b"
# each adapter's weights digest, which submit_request adds to the kwargs
DIGESTS = {"sql": b"\x01" * 32, "chat": b"\x02" * 32}
SERVED = ServedModels(base=[MODEL, "Qwen/Qwen3.5-0.8B"], digests=DIGESTS, lora=True)


class _Rust(_StubAPIServer):
    async def iter_result_chunks(self, request_id):
        for c in (_Chunk("text", b"ok"), _Chunk("image", b"png"), _Chunk("video", b"mp4"), _Chunk("audio", b"\0\0")):
            yield c


class _Python:
    """What the Python routes call on APIServer: a submit records its kwargs."""

    open_result_stream = APIServer.open_result_stream
    lora_kwargs = APIServer.lora_kwargs

    def __init__(self, model_name, upload_dir, served):
        self.model_name, self.upload_dir, self.model, self.served_models = model_name, upload_dir, None, served
        self.submitted = []

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs["request_id"]

    async def iter_result_chunks(self, request_id):
        for c in (_Chunk("text", b"ok"), _Chunk("image", b"png"), _Chunk("video", b"mp4"), _Chunk("audio", b"\0\0")):
            yield c

    async def collect_results(self, request_id, raw_request=None):
        return [c async for c in self.iter_result_chunks(request_id)]

    async def iter_result_chunk_batches(self, request_id):
        async for c in self.iter_result_chunks(request_id):
            yield [c]

    def release_request(self, request_id):
        pass

    def request_usage(self, request_id):
        return None


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _python(model, served, tmp_path, monkeypatch):
    api = _Python(model, tmp_path, served)
    monkeypatch.setattr(entrypoint, "api_server", api)
    return api, TestClient(entrypoint.app)


def _resolved(submit: dict, served: ServedModels) -> tuple[dict, bytes]:
    """The kwargs and prefix root a Python submit runs under once ``submit_request``
    resolves its ``model``."""
    kwargs, root = served.select(submit.get("model"), submit.get("model_kwargs"))
    return kwargs or {}, root


def _rust_resolved(rust_kwargs: dict | None, served: ServedModels) -> tuple[dict, bytes]:
    """The same for mstar-server's submit, which the bridge makes without a ``model``."""
    kwargs, root = served.select(None, rust_kwargs)
    return kwargs or {}, root


def test_v1_models_lists_the_adapters_beside_the_base_model(tmp_path, monkeypatch):
    _, client = _python(MODEL, SERVED, tmp_path, monkeypatch)
    want = client.get("/v1/models").json()
    with _model_stack(MODEL, _Rust(), env={SERVED_MODELS_ENV: served_models_json(SERVED)}) as (port, _upload):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
            got = json.loads(r.read())
    blank = lambda cards: [{k: v for k, v in c.items() if k != "created"} for c in cards["data"]]  # noqa: E731
    assert blank(got) == blank(want) == [
        {"id": MODEL, "object": "model", "owned_by": "mstar"},
        {"id": "sql", "object": "model", "owned_by": "mstar", "parent": MODEL},
        {"id": "chat", "object": "model", "owned_by": "mstar", "parent": MODEL},
    ]


CHATS = {
    "adapter": {"model": "sql"},
    "base-name": {"model": MODEL},
    "repo-name": {"model": "Qwen/Qwen3.5-0.8B"},
    "no-model": {},
    "unknown": {"model": "nope"},
    "kwarg": {"lora_adapter": "chat"},
    "kwarg-agrees": {"model": "chat", "lora_adapter": "chat"},
    "kwarg-disagrees": {"model": "sql", "lora_adapter": "chat"},
    "kwarg-unknown": {"lora_adapter": "nope"},
}


@pytest.mark.parametrize("stream", [False, True], ids=["full", "stream"])
@pytest.mark.parametrize("case", sorted(CHATS))
def test_a_chat_runs_under_the_adapter_its_model_names(case, stream, tmp_path, monkeypatch):
    """The same adapter in the submit, the same refusal, and the response naming
    the adapter that served it, as vLLM's does."""
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": stream, **CHATS[case]}
    api, client = _python(MODEL, SERVED, tmp_path, monkeypatch)
    py = client.post("/v1/chat/completions", json=body)
    stub = _Rust()
    with _model_stack(MODEL, stub, env={SERVED_MODELS_ENV: served_models_json(SERVED)}) as (port, _upload):
        status, data = _post(port, "/v1/chat/completions", body)
    assert status == py.status_code
    if status != 200:
        assert json.loads(data)["error"]["message"] == py.json()["error"]["message"]
        assert stub.submitted == api.submitted == []
        return
    (sub,) = stub.submitted
    assert _rust_resolved(sub["mk"], SERVED) == _resolved(api.submitted[0], SERVED)
    if stream:
        models = {json.loads(line[6:])["model"] for line in data.decode().splitlines()
                  if line.startswith("data: {")}
        want = {json.loads(line[6:])["model"] for line in py.text.splitlines() if line.startswith("data: {")}
    else:
        models, want = {json.loads(data)["model"]}, {py.json()["model"]}
    assert models == want


@pytest.mark.parametrize("route,body,cls", [
    ("/v1/audio/speech", {"input": "Hi."}, "Qwen3TTSAdapter"),
    ("/v1/images/generations", {"prompt": "a cat"}, "DiffusionImageAdapter"),
    ("/v1/videos/generations", {"prompt": "a cat"}, "Wan22Adapter"),
], ids=["speech", "images", "videos"])
@pytest.mark.parametrize("case", ["adapter", "unknown", "kwarg", "kwarg-disagrees", "no-model"])
def test_speech_images_and_videos_run_under_the_adapter_too(route, body, cls, case, tmp_path, monkeypatch):
    model = _model_of(cls)
    served = ServedModels(base=[model], digests=DIGESTS, lora=True)
    body = {**body, **CHATS[case]}
    api, client = _python(model, served, tmp_path, monkeypatch)
    try:
        py_status = client.post(route, json=body).status_code
        py_error = None
    except ValueError as e:  # a submit_request refusal the route lets through
        py_status, py_error = 400, str(e)
    stub = _Rust()
    with _model_stack(model, stub, env={SERVED_MODELS_ENV: served_models_json(served)}) as (port, _upload):
        status, data = _post(port, route, body)
    want_error = None
    if api.submitted:
        try:
            _resolved(api.submitted[0], served)
        except ValueError as e:
            want_error = str(e)
    want_error = want_error or py_error
    if want_error is not None:
        assert (status, json.loads(data)["error"]["message"]) == (400, want_error)
        assert stub.submitted == []
        return
    assert status == py_status == 200, data
    assert _rust_resolved(stub.submitted[0]["mk"], served) == _resolved(api.submitted[0], served)


# A 1x1 PNG, the edit's input image.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360f8cfc0f01f0005000201e7a5b6"
    "2a0000000049454e44ae426082")


def _edit_form(fields: dict) -> tuple[bytes, str]:
    boundary = "----mstarlora"
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    )
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="in.png"\r\n'
             "Content-Type: image/png\r\n\r\n").encode() + PNG + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


@pytest.mark.parametrize("case", ["adapter", "unknown", "kwarg", "kwarg-disagrees", "no-model"])
def test_an_image_edit_runs_under_the_adapter_its_form_names(case, tmp_path, monkeypatch):
    """``/v1/images/edits`` reads ``model`` from a form field, not JSON."""
    model = _model_of("DiffusionImageAdapter")
    served = ServedModels(base=[model], digests=DIGESTS, lora=True)
    body, ctype = _edit_form({"prompt": "make it blue", **CHATS[case]})
    api, client = _python(model, served, tmp_path, monkeypatch)
    py = client.post("/v1/images/edits", content=body, headers={"Content-Type": ctype})
    stub = _Rust()
    with _model_stack(model, stub, env={SERVED_MODELS_ENV: served_models_json(served)}) as (port, _upload):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/images/edits", data=body,
                                     headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status, data = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, data = e.code, e.read()
    want_error = None
    if api.submitted:
        try:
            _resolved(api.submitted[0], served)
        except ValueError as e:
            want_error = str(e)
    elif py.status_code != 200:
        want_error = py.json()["error"]["message"]
    if want_error is not None:
        assert (status, json.loads(data)["error"]["message"]) == (400, want_error)
        assert stub.submitted == []
        return
    assert status == py.status_code == 200, data
    assert _rust_resolved(stub.submitted[0]["mk"], served) == _resolved(api.submitted[0], served)
