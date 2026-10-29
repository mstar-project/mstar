"""How a request reaches a LoRA adapter, and loading and unloading adapters while serving.

The OpenAI ``model`` field names an adapter (vLLM's semantics). A runtime load is selectable
only once every worker acknowledged it, and a failure anywhere is undone everywhere; the
end-to-end tests run the real admin, conductor ops, worker handlers and pools over the wire codec.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import types
from concurrent.futures import wait
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("safetensors")
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

try:
    from .test_lora_pool import ingest, make_spec, run_step, write_adapter
    from .test_lora_tp_lockstep import _Barrier
except ImportError:  # collected without package context
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_lora_pool import ingest, make_spec, run_step, write_adapter
    from test_lora_tp_lockstep import _Barrier

import mstar.communication.wire_types  # noqa: E402,F401  (registers the tags)
from mstar.api_server.data_worker import adapter_prefix_roots  # noqa: E402
from mstar.api_server.entrypoint import APIServer  # noqa: E402
from mstar.api_server.lora_admin import LoraAdmin  # noqa: E402
from mstar.api_server.openai.adapters import SubmitArgs  # noqa: E402
from mstar.api_server.request_types import APIServerMessage, LoraAdapterResult, RequestFailed  # noqa: E402
from mstar.api_server.served_models import ServedModels  # noqa: E402
from mstar.communication.wire import decode, encode  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.lora_ops import LoraAdapterOps  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig  # noqa: E402
from mstar.engine.resources.lora import LORA_ADAPTER_KWARG, LORA_RESOURCE_KEY, LoraReqConfig  # noqa: E402
from mstar.engine.resources.lora.config import ADAPTER_WEIGHTS_FILE  # noqa: E402
from mstar.engine.resources.lora.identity import (  # noqa: E402
    AdapterPrefixRoots,
    adapter_prefix_root,
    read_adapter_digest,
)
from mstar.engine.resources.lora.pool import LoraAdapterPool  # noqa: E402
from mstar.engine.resources.lora.serving import lora_specs  # noqa: E402
from mstar.engine.resources.lora.store import check_adapter_file, check_adapters  # noqa: E402
from mstar.engine.resources.spec import apply_yaml_overrides  # noqa: E402
from mstar.engine.resources.step import AdmitRuntimeError  # noqa: E402
from mstar.model.base import KVLabel, PrefixKeys  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    LoadLoraAdapter,
    LoraAdapterAck,
    UnloadLoraAdapter,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.worker import Worker  # noqa: E402

BASE = "qwen3_5_0.8b"
LORA_CONFIG = {"resources": {"lora": {"adapters": {"sql": "/a"}}}}
# a refused load or unload: a ValueError is a 400, an HTTPException carries its status
REFUSED = (ValueError, HTTPException)


def _status(exc: Exception) -> int:
    return getattr(exc, "status_code", 400 if isinstance(exc, (ValueError, TypeError)) else 500)


def _message(exc: Exception) -> str:
    return str(getattr(exc, "detail", exc))


def _wire(message):
    return decode(encode(message))


@pytest.fixture
def paths(tmp_path):
    out = {}
    for i, name in enumerate(("a1", "a2", "a3")):
        out[name] = str(tmp_path / f"{name}.safetensors")
        write_adapter(out[name], rank=2 + i, seed=i + 1)
    return out


# --- the OpenAI `model` field ------------------------------------------------------------

class _StubAPI:
    def __init__(self, config):
        self.model_name = BASE
        self.model = SimpleNamespace(get_output_sample_rate=lambda modality="audio": 24000)
        self.served_models = ServedModels.of(BASE, config)
        self.lora_admin = None
        self.upload_dir = Path(tempfile.mkdtemp())
        self.last_submit = None

    def submit_request(self, **kw):
        # the real APIServer's adapter selection, on this stub's ServedModels
        kw["model_kwargs"] = APIServer.lora_kwargs(self, kw.pop("model", None), kw.get("model_kwargs"))
        self.last_submit = kw
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return [SimpleNamespace(modality="text", data=b"ok", metadata={})]


def _client(monkeypatch, config, adapter=None):
    import mstar.api_server
    import mstar.api_server.openai.router as router_module

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = fake_ep.api_server = _StubAPI(config)
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    if adapter is not None:
        monkeypatch.setattr(router_module, "get_adapter", lambda name: adapter)
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app), stub


class _EveryRoute:
    """An OpenAI adapter on every surface, passing the request's extra fields through."""

    supports_chat = supports_speech = supports_images = supports_videos = True

    @staticmethod
    def _args(req, output):
        return SubmitArgs(text="hi", input_modalities=["text"], output_modalities=[output],
                          model_kwargs=dict(req.model_extra or {}))

    def speech_to_request(self, req, upload_dir):
        return self._args(req, "audio")

    def image_to_request(self, req, upload_dir):
        return self._args(req, "image")

    def video_to_request(self, req, upload_dir):
        return self._args(req, "video")

    def image_edit_to_request(self, prompt, image_path, model_kwargs):
        return SubmitArgs(text=prompt, input_modalities=["text", "image"], model_kwargs=dict(model_kwargs))


_ROUTES = {
    "chat": lambda c, b: c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], **b}),
    "speech": lambda c, body: c.post("/v1/audio/speech", json={"input": "hi", **body}),
    "images": lambda c, body: c.post("/v1/images/generations", json={"prompt": "hi", **body}),
    "videos": lambda c, body: c.post("/v1/videos/generations", json={"prompt": "hi", **body}),
    "image_edits": lambda c, body: c.post(
        "/v1/images/edits", data={"prompt": "hi", **body}, files={"image": ("a.png", b"png")},
    ),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_every_route_maps_the_model_field_to_an_adapter(monkeypatch, route):
    client, stub = _client(monkeypatch, LORA_CONFIG, None if route == "chat" else _EveryRoute())
    post = _ROUTES[route]
    response = post(client, {"model": "sql"})
    assert stub.last_submit["model_kwargs"][LORA_ADAPTER_KWARG] == "sql"
    if route == "chat":
        assert response.json()["model"] == "sql"
    refused = [{"model": "nope"}]
    if route in ("chat", "speech"):  # an extra-body adapter is checked like the model field
        post(client, {LORA_ADAPTER_KWARG: "sql"})
        assert stub.last_submit["model_kwargs"][LORA_ADAPTER_KWARG] == "sql"
        refused += [{LORA_ADAPTER_KWARG: "nope"}, {"model": "sql", LORA_ADAPTER_KWARG: "other"}]
    for body in refused:
        stub.last_submit = None
        response = post(client, body)
        assert response.status_code == 400 and stub.last_submit is None, body
    assert "unknown LoRA adapter" in post(client, {"model": "nope"}).text
    response = post(client, {"model": BASE})
    assert LORA_ADAPTER_KWARG not in stub.last_submit["model_kwargs"]
    if route == "chat":
        assert response.json()["model"] == BASE


def test_models_lists_adapters_with_their_parent(monkeypatch):
    client, stub = _client(monkeypatch, {})
    cards = client.get("/v1/models").json()["data"]
    assert [c["id"] for c in cards] == [BASE] and "parent" not in cards[0]
    assert stub.served_models.adapter_for("anything") is None  # no LoRA: the field is not read
    client, stub = _client(monkeypatch, LORA_CONFIG)
    stub.served_models.add("loaded")  # a runtime load
    cards = client.get("/v1/models").json()["data"]
    assert [(c["id"], c.get("parent")) for c in cards] == [(BASE, None), ("sql", BASE), ("loaded", BASE)]


@pytest.mark.parametrize("resources", [{"typo": {}}, ["kv"], ["lora"]], ids=["typo", "list", "list_naming_lora"])
def test_a_deployment_without_lora_leaves_its_resources_block_to_the_conductor(resources):
    """The API server reads the config before the conductor's ``validate_deployment``,
    which reports a bad ``resources:`` block with the deployment's other problems."""
    config = {"resources": resources}
    served = ServedModels.of(BASE, config)
    assert not served.lora and served.adapters == ()
    assert lora_specs(object(), config) == []
    keyed = SimpleNamespace(
        kv_labels=lambda: {"kv": {"main": KVLabel(keys=PrefixKeys("ids", ("prefill",)))}},
        get_node_resources=list,
    )
    assert adapter_prefix_roots(keyed, config) is None


@pytest.mark.parametrize("block", [None, ["sql"]], ids=["bare", "list"])
def test_a_lora_block_that_is_not_a_mapping_is_left_to_the_conductor(block):
    """``validate_deployment`` reports it: `resources: lora:` must be a mapping."""
    assert lora_specs(object(), {"resources": {LORA_RESOURCE_KEY: block}}) == []


def test_submit_request_refuses_an_unserved_adapter_kwarg():
    """``/generate`` and the WebSocket route reach ``submit_request`` with the kwarg; an
    unloaded adapter is out of ``ServedModels`` at once, so new requests on it are a 400."""
    from api_server_helpers import make_test_api_server

    submitted = []
    served = ServedModels(base=(BASE,), adapters=("sql",), lora=True)
    server = make_test_api_server(
        served_models=served, preprocess_worker=SimpleNamespace(new_request=submitted.append),
    )
    common = dict(input_modalities=["text"], output_modalities=["text"], text="hi")
    server.submit_request(**common, model_kwargs={LORA_ADAPTER_KWARG: "sql"}, request_id="a")
    server.submit_request(**common, model="sql", request_id="b")
    assert [s.model_kwargs[LORA_ADAPTER_KWARG] for s in submitted] == ["sql", "sql"]
    served.remove("sql")
    for kwargs in ({LORA_ADAPTER_KWARG: "sql"}, {LORA_ADAPTER_KWARG: 3}):
        with pytest.raises(HTTPException) as err:
            server.submit_request(**common, model_kwargs=kwargs, request_id="c")
        assert err.value.status_code == 400
    assert len(submitted) == 2 and "c" not in server._live


# --- the admin endpoints -----------------------------------------------------------------

class _Admin:
    """The router's view of ``LoraAdmin``: what it was asked, and what it answers."""

    def __init__(self, error=None):
        self.calls, self.error = [], error

    async def _answer(self, call, verb):
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        return f"Success: LoRA adapter '{call[1]}' {verb} successfully."

    async def load(self, name, path):
        return await self._answer(("load", name, path), "added")

    async def unload(self, name):
        return await self._answer(("unload", name), "removed")


_ENDPOINTS = (("/v1/load_lora_adapter", {"lora_name": "a", "lora_path": "/p"}),
              ("/v1/unload_lora_adapter", {"lora_name": "a"}))


@pytest.mark.parametrize("error,status,message", [
    (None, 200, None),
    (ValueError("why"), 400, "why"),
    (ValueError("bad /srv/x/adapter.safetensors"), 400, "bad adapter.safetensors"),
    (HTTPException(status_code=409, detail="why"), 409, "why"),
    (RuntimeError("/srv/secret"), 500, "internal error (error id"),
])
def test_the_lora_endpoints(monkeypatch, error, status, message):
    """vLLM's bodies; errors take every route's envelope: a ValueError is a 400 with
    server paths cut to base names, anything else a 500 naming only an error id."""
    client, stub = _client(monkeypatch, LORA_CONFIG)
    stub.lora_admin = _Admin(error)
    for (path, body), verb in zip(_ENDPOINTS, ("added", "removed"), strict=True):
        r = client.post(path, json=body)
        if error is None:
            assert r.status_code == 200 and r.text == f"Success: LoRA adapter 'a' {verb} successfully."
        else:
            got = r.json()["error"]
            assert r.status_code == status and got["code"] == status
            assert got["type"] == ("invalid_request_error" if status < 500 else "server_error")
            assert got["message"].startswith(message) and "/srv" not in got["message"]
    # a missing field never reaches the admin
    client.post("/v1/load_lora_adapter", json={"lora_name": "b"})
    client.post("/v1/unload_lora_adapter", json={})
    assert stub.lora_admin.calls == [("load", "a", "/p"), ("unload", "a")]


def test_the_lora_endpoints_are_404_without_runtime_updates(monkeypatch):
    client, stub = _client(monkeypatch, LORA_CONFIG)
    sent = []
    for spec in (None, SimpleNamespace(runtime_updates=False)):
        stub.lora_admin = LoraAdmin(spec, ServedModels(base=(BASE,)), None, sent.append)
        for path, body in _ENDPOINTS:
            assert client.post(path, json=body).status_code == 404
    assert sent == []
    # nor does the conductor act on one
    out = _Outbox()
    for spec in (None, make_spec({})):
        LoraAdapterOps(spec, ["worker_0"], out).load(LoadLoraAdapter(op_id="l", lora_name="a", lora_path="/p"))
    assert [a.status for a in out.answers()] == [404, 404] and out.to("worker_0") == []
    # with them on, a malformed name or path is a 400
    admin = LoraAdmin(SimpleNamespace(runtime_updates=True), ServedModels(base=(BASE,)), None, sent.append)
    for call in (admin.load("", "/p"), admin.load("a", ""), admin.unload(""), admin.load(BASE, "/p")):
        with pytest.raises(ValueError):
            asyncio.run(call)


def test_the_new_messages_round_trip_the_wire():
    load = LoadLoraAdapter(op_id="op", lora_name="sql", lora_path="/a.safetensors", lora_digest="ab" * 32)
    unload = UnloadLoraAdapter(op_id="op", lora_name="sql")
    for message in (
        ConductorMessage(ConductorMessageType.LOAD_LORA_ADAPTER, load),
        ConductorMessage(ConductorMessageType.UNLOAD_LORA_ADAPTER, unload),
        WorkerMessage(WorkerMessageType.LOAD_LORA_ADAPTER, load),
        WorkerMessage(WorkerMessageType.UNLOAD_LORA_ADAPTER, unload),
        ConductorMessage(ConductorMessageType.LORA_ADAPTER_ACK,
                         LoraAdapterAck(op_id="op", worker_id="worker_1", error="boom")),
        ConductorMessage(ConductorMessageType.LORA_ADAPTER_ACK, LoraAdapterAck(op_id="op", worker_id="w")),
        APIServerMessage(message_type="lora_adapter_result", body=LoraAdapterResult(op_id="op")),
        APIServerMessage(message_type="lora_adapter_result",
                         body=LoraAdapterResult(op_id="op", error_message="failed", status=400)),
        # the union decodes by required fields: a body before it still decodes as itself
        APIServerMessage(message_type="lora_adapter_result",
                         body=RequestFailed(request_id="r", error_message="e", status=400)),
    ):
        got = _wire(message)
        assert got == message and type(got.body) is type(message.body)
    # a request's adapter and prefix root, as the conductor sends them to a worker
    root = adapter_prefix_root(b"\x01" * 32)
    for kv in (KVReqConfig(prefix_root=root), KVReqConfig()):
        info = CurrentForwardPassInfo(request_id="r", fwd_index=0, random_seed=1, max_tokens=8,
                                      graph_walk="prefill_text", partition_name="p")
        info.resource_configs = {LORA_RESOURCE_KEY: LoraReqConfig(adapter="sql"), "kv": kv}
        got = _wire(WorkerMessage(WorkerMessageType.INPUT_SIGNALS, InputSignals(
            request_id="r", partition_name="p", request_info=info, inputs=[],
        ))).body.request_info.resource_configs
        assert got[LORA_RESOURCE_KEY] == LoraReqConfig(adapter="sql") and got["kv"].prefix_root == kv.prefix_root


# --- runtime loads -----------------------------------------------------------------------

class _Outbox:
    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def __call__(self, peer, message):
        self.sent.append((peer, _wire(message)))

    def to(self, peer):
        return [m for p, m in self.sent if p == peer]

    def answers(self):
        return [m.body for m in self.to("api_server")]


def test_a_load_is_selectable_only_once_every_worker_acknowledged(paths):
    spec = make_spec({"a1": paths["a1"]}, runtime_updates=True)
    out = _Outbox()
    ops = LoraAdapterOps(spec, ["worker_0", "worker_1"], out)
    ops.load(LoadLoraAdapter(op_id="op1", lora_name="a2", lora_path=paths["a2"]))
    for worker in ("worker_0", "worker_1"):
        assert [m.message_type for m in out.to(worker)] == [WorkerMessageType.LOAD_LORA_ADAPTER]
    ops.ack(LoraAdapterAck(op_id="op1", worker_id="worker_0"))
    with pytest.raises(ValueError, match="unknown LoRA adapter 'a2'"):
        LoraReqConfig(adapter="a2").validate(spec)
    assert out.answers() == []
    ops.ack(LoraAdapterAck(op_id="op1", worker_id="worker_1"))
    LoraReqConfig(adapter="a2").validate(spec)
    assert out.answers() == [LoraAdapterResult(op_id="op1", status=200, served=True)]


def test_a_load_a_worker_never_answers_times_out_and_is_undone(paths):
    spec = make_spec({}, runtime_updates=True, runtime_update_timeout_s=10.0)
    out, clock = _Outbox(), SimpleNamespace(now=0.0)
    ops = LoraAdapterOps(spec, ["worker_0", "worker_1"], out, clock=lambda: clock.now)
    ops.load(LoadLoraAdapter(op_id="op1", lora_name="a2", lora_path=paths["a2"]))
    ops.ack(LoraAdapterAck(op_id="op1", worker_id="worker_0"))  # worker_1 never answers
    clock.now = 9.0
    ops.expire()
    assert out.answers() == []
    clock.now = 10.0
    ops.expire()
    (answer,) = out.answers()
    assert answer.status == 504 and "1 worker(s) did not answer" in answer.error_message
    assert answer.served is False and "a2" not in spec.adapters
    # undone on every worker: the silent one may still be reading it
    for worker in ("worker_0", "worker_1"):
        assert out.to(worker)[-1].message_type == WorkerMessageType.UNLOAD_LORA_ADAPTER
    # a late ack is ignored, and the name is free for another try
    ops.ack(LoraAdapterAck(op_id="op1", worker_id="worker_1"))
    ops.load(LoadLoraAdapter(op_id="op2", lora_name="a2", lora_path=paths["a2"]))
    assert len(out.answers()) == 1 and out.to("worker_0")[-1].body.op_id == "op2"


def _finish_when_read(worker):
    if worker._lora_loads:
        wait([f for load in worker._lora_loads._loads for _, f in load.reads], timeout=30)
        worker._finish_lora_loads()


class _Deployment:
    """API server, conductor and two workers (TP ranks 0 and 1, each with its own pool)."""

    def __init__(self, paths, fail_on: str | None = None):
        self.roots = AdapterPrefixRoots()
        self.served = ServedModels(base=("base",), lora=True)
        self.admin = LoraAdmin(make_spec({}, runtime_updates=True), self.served, self.roots, self._to_conductor)
        self.ops = LoraAdapterOps(make_spec({}, runtime_updates=True), ["worker_0", "worker_1"], self._from_conductor)
        self.pools, self.workers, self.unload_seen = {}, {}, []
        for rank in range(2):
            worker_id = f"worker_{rank}"
            pool = self.pools[worker_id] = LoraAdapterPool(make_spec({}), "cpu", rank=rank, world_size=2)
            if fail_on == worker_id:
                def broken(name, path):
                    raise OSError(f"{name}: cannot read /srv/secret on this rank")
                pool.store._read = broken  # the worker's background read fails
            w = self.workers[worker_id] = Worker.__new__(Worker)
            w.worker_id = worker_id
            w.engine_manager = SimpleNamespace(engine=SimpleNamespace(lora_pools=lambda p=pool: [p]))
            w.communicator = SimpleNamespace(send=self._from_worker)

    def _to_conductor(self, message: ConductorMessage):
        message = _wire(message)
        if message.message_type == ConductorMessageType.LOAD_LORA_ADAPTER:
            self.ops.load(message.body)
        else:
            self.ops.unload(message.body)

    def _from_conductor(self, peer, message):
        message = _wire(message)
        if peer == "api_server":
            self.admin.on_result(message.body)
            return
        if message.message_type == WorkerMessageType.UNLOAD_LORA_ADAPTER:
            name = message.body.lora_name
            self.unload_seen.append((peer, name in self.served.adapters, self.roots.root(name)))
        self.workers[peer]._process_message_list([message])
        _finish_when_read(self.workers[peer])

    def _from_worker(self, peer, message):
        assert peer == "conductor"
        self.ops.ack(_wire(message).body)

    def load(self, name, path):
        return asyncio.run(self.admin.load(name, path))

    def unload(self, name):
        return asyncio.run(self.admin.unload(name))


def test_a_runtime_load_and_unload_end_to_end(paths):
    d = _Deployment(paths)
    assert d.load("a1", paths["a1"]) == "Success: LoRA adapter 'a1' added successfully."
    assert all(pool.serves("a1") for pool in d.pools.values())
    assert d.served.adapter_for("a1") == "a1"
    LoraReqConfig(adapter="a1").validate(d.ops._spec)
    root = d.roots.root("a1")
    assert root == adapter_prefix_root(read_adapter_digest(make_spec({}), paths["a1"]))
    with pytest.raises(REFUSED) as err:
        d.load("a1", paths["a2"])
    assert _status(err.value) == 400 and "already been loaded" in _message(err.value)

    assert d.unload("a1") == "Success: LoRA adapter 'a1' removed successfully."
    # out of selection before any worker hears of it; keyed until the conductor confirmed
    assert d.unload_seen == [("worker_0", False, root), ("worker_1", False, root)]
    assert d.roots.root("a1") is None and d.served.adapters == ()
    assert not any(pool.serves("a1") for pool in d.pools.values())
    with pytest.raises(REFUSED) as err:
        d.unload("a1")
    assert _status(err.value) == 404

    d.load("a1", paths["a1"])  # the same weights key under the same root
    assert d.roots.root("a1") == root
    d.unload("a1")
    d.load("a1", paths["a2"])  # other weights under the name: another namespace
    assert d.roots.root("a1") not in (None, root)


def test_a_failed_read_on_one_worker_is_undone_everywhere_without_leaking_the_io_error(paths):
    d = _Deployment(paths, fail_on="worker_1")
    with pytest.raises(REFUSED) as err:
        d.load("a1", paths["a1"])
    assert _status(err.value) == 400 and "could not be read" in _message(err.value)
    # the I/O error and the worker ids stay in the server log
    assert "/srv/secret" not in _message(err.value) and "worker_1" not in _message(err.value)
    assert not any(pool.serves("a1") for pool in d.pools.values())
    assert "a1" not in d.pools["worker_0"].store and "a1" not in d.ops._spec.adapters
    assert d.served.adapters == () and d.roots.root("a1") is None


def test_a_refused_path_tells_the_client_nothing_about_the_servers_files(paths, tmp_path):
    d = _Deployment(paths)
    other = tmp_path / "notes.txt"
    other.write_text("not an adapter")
    answers = set()
    for path in (str(tmp_path / "missing.safetensors"), str(other), str(tmp_path)):
        with pytest.raises(REFUSED) as err:
            d.load("x", path)
        assert _status(err.value) == 400 and str(tmp_path) not in _message(err.value)
        answers.add(_message(err.value))
    # a missing file, a file of another kind and a directory read the same
    assert len(answers) == 1 and "could not be read" in answers.pop()
    # an adapter the deployment cannot serve is refused with its reason, before any worker
    wide = tmp_path / "wide.safetensors"
    write_adapter(wide, rank=8, seed=9)  # above make_spec's max_rank of 4
    with pytest.raises(REFUSED) as err:
        d.load("wide", str(wide))
    assert _status(err.value) == 400 and "max_rank" in _message(err.value)
    assert not any(pool.serves(n) for pool in d.pools.values() for n in ("x", "wide"))


def test_requests_on_an_unloaded_adapter_finish_and_new_ones_are_refused(paths):
    d = _Deployment(paths)
    d.load("a1", paths["a1"])
    for pool in d.pools.values():
        ingest(pool, "r1", "a1")
        assert run_step(pool, ["r1"]).ok
    slots = {w: pool.slot_of("r1") for w, pool in d.pools.items()}
    d.unload("a1")
    for w, pool in d.pools.items():
        assert run_step(pool, ["r1"]).ok  # still running on its slot
        assert pool.slot_state(slots[w]).refcount == 1 and "a1" in pool.store
        with pytest.raises(ValueError, match="unknown LoRA adapter"):
            ingest(pool, "r2", "a1")
    # a reload before they finish is refused on every worker, and rolled back
    with pytest.raises(REFUSED, match="still running"):
        d.load("a1", paths["a1"])
    for w, pool in d.pools.items():
        pool.remove_request("r1")
        assert pool.slot_state(slots[w]).refcount == 0
        assert "a1" not in pool.store and pool.resident_slot("a1") is None


# --- startup checks and digests --------------------------------------------------------------

def test_startup_refuses_every_bad_configured_adapter(tmp_path):
    peft = tmp_path / "peft"
    peft.mkdir()
    good = str(peft / ADAPTER_WEIGHTS_FILE)
    write_adapter(good, rank=2, seed=2)
    wide = str(tmp_path / "wide.safetensors")
    write_adapter(wide, rank=8, seed=3)  # above max_rank 4
    misfit = str(tmp_path / "misfit.safetensors")  # q_proj's A with the wrong input width
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 5),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 2)}, misfit)
    bad = {"wide": wide, "misfit": misfit, "gone": str(tmp_path / "gone.safetensors")}

    spec = make_spec({})
    apply_yaml_overrides([spec], {"resources": {"lora": {"adapters": {"good": str(peft), **bad}}}})
    assert spec.adapters["good"] == good  # a PEFT directory names its safetensors
    with pytest.raises(ValueError) as err:
        check_adapters(spec)
    assert all(repr(name) in str(err.value) for name in bad) and "'good'" not in str(err.value)
    assert "rank 8" in str(err.value)
    # mechanism off: the pool alone preloads only cpu_capacity adapters, and starts
    LoraAdapterPool(make_spec({"good": good, "wide": wide}, cpu_capacity=1), "cpu")
    # the conductor runs the check when the API server did not
    conductor = object.__new__(Conductor)
    conductor.model = SimpleNamespace(get_node_resources=lambda: [], get_lora_specs=lambda: [make_spec({})])
    conductor.model_config = {"resources": {"lora": {"adapters": {"good": good, "wide": wide}}}}
    with pytest.raises(ValueError, match="'wide'"):
        conductor._settle_lora_adapters(None)


@pytest.mark.parametrize("pinned", [True, False])
def test_a_file_changed_on_disk_is_refused_on_every_rank(paths, pinned):
    """a1's file is overwritten after it was checked and before a worker first reads it.
    Mechanism off (no digest pinned): the new weights are served."""
    path = paths["a1"]
    digest = read_adapter_digest(make_spec({}), path)
    barrier, pools = _Barrier(2), []
    for rank in range(2):
        # a2 first: the preload reads only it, so a1's first read comes after the change
        spec = make_spec({"a2": path, "a1": path}, max_slots=1, cpu_capacity=1)
        spec.adapter_digests = {"a1": digest, "a2": digest} if pinned else {}
        pools.append(LoraAdapterPool(spec, "cpu", rank=rank, world_size=2, agree=barrier.for_rank(rank)))
    write_adapter(path, rank=2, seed=7)  # same name and shapes, other weights
    outcomes = [None, None]

    def run(i):
        ingest(pools[i], "r1", "a1")
        outcomes[i] = run_step(pools[i], ["r1"])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    if not pinned:
        assert all(outcome.ok for outcome in outcomes)
        return
    for outcome in outcomes:
        assert not outcome.ok and isinstance(outcome.reason, AdmitRuntimeError)
        assert outcome.reason.request_ids == frozenset({"r1"})
    assert "changed on disk" in outcomes[0].reason.message + outcomes[1].reason.message
    assert all(pool.resident_slot("a1") is None for pool in pools)


def test_a_runtime_load_whose_file_changes_before_the_workers_read_it_is_refused(paths):
    d = _Deployment(paths)
    forward = d.ops.load

    def changed_in_between(body):
        assert body.lora_digest  # the API server's check travels with the load
        write_adapter(paths["a1"], rank=2, seed=7)
        forward(body)

    d.ops.load = changed_in_between
    with pytest.raises(REFUSED) as err:
        d.load("a1", paths["a1"])
    assert _status(err.value) == 400 and "changed on disk" in _message(err.value)
    assert not any(pool.serves("a1") for pool in d.pools.values())
    assert d.served.adapters == () and d.roots.root("a1") is None


def test_specs_that_digest_an_adapter_differently_refuse_it(paths):
    """One digest pins an adapter on every spec's pools, so specs reading a file into
    different weights refuse it."""
    one = make_spec({"a": paths["a1"]})
    other = make_spec({"a": paths["a1"]}, convert_keys=lambda sd: {k: v * 2 for k, v in sd.items()})
    other.resource_key = "lora_other"
    with pytest.raises(ValueError, match="digests differently"):
        check_adapter_file(one, "a", paths["a1"], [other])
