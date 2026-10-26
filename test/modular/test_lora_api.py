"""How a request reaches a LoRA adapter, and how the serving config's adapters are checked.

The OpenAI ``model`` field names an adapter (vLLM's semantics). Every configured adapter is
checked at startup and pinned to the digest of the weights checked, on every rank.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import types
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
from mstar.api_server.openai.adapters import SubmitArgs  # noqa: E402
from mstar.api_server.served_models import ServedModels  # noqa: E402
from mstar.communication.wire import decode, encode  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig  # noqa: E402
from mstar.engine.resources.lora import LORA_ADAPTER_KWARG, LORA_RESOURCE_KEY, LoraReqConfig  # noqa: E402
from mstar.engine.resources.lora.config import ADAPTER_WEIGHTS_FILE  # noqa: E402
from mstar.engine.resources.lora.identity import adapter_prefix_root, read_adapter_digest  # noqa: E402
from mstar.engine.resources.lora.pool import LoraAdapterPool  # noqa: E402
from mstar.engine.resources.lora.serving import lora_specs  # noqa: E402
from mstar.engine.resources.lora.store import check_adapter_file, check_adapters  # noqa: E402
from mstar.engine.resources.spec import apply_yaml_overrides  # noqa: E402
from mstar.engine.resources.step import AdmitRuntimeError  # noqa: E402
from mstar.model.base import KVLabel, PrefixKeys  # noqa: E402
from mstar.utils.ipc_format import InputSignals, WorkerMessage, WorkerMessageType  # noqa: E402

BASE = "qwen3_5_0.8b"
LORA_CONFIG = {"resources": {"lora": {"adapters": {"sql": "/a"}}}}


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
    cards = client.get("/v1/models").json()["data"]
    assert [(c["id"], c.get("parent")) for c in cards] == [(BASE, None), ("sql", BASE)]


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
    adapter the deployment does not serve is a 400."""
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
    for kwargs in ({LORA_ADAPTER_KWARG: "nope"}, {LORA_ADAPTER_KWARG: 3}):
        with pytest.raises(HTTPException) as err:
            server.submit_request(**common, model_kwargs=kwargs, request_id="c")
        assert err.value.status_code == 400
    assert len(submitted) == 2 and "c" not in server._live


# --- the wire ------------------------------------------------------------------------------

def test_a_requests_adapter_and_prefix_root_round_trip_the_wire():
    """As the conductor sends them to a worker."""
    root = adapter_prefix_root(b"\x01" * 32)
    for kv in (KVReqConfig(prefix_root=root), KVReqConfig()):
        info = CurrentForwardPassInfo(request_id="r", fwd_index=0, random_seed=1, max_tokens=8,
                                      graph_walk="prefill_text", partition_name="p")
        info.resource_configs = {LORA_RESOURCE_KEY: LoraReqConfig(adapter="sql"), "kv": kv}
        got = _wire(WorkerMessage(WorkerMessageType.INPUT_SIGNALS, InputSignals(
            request_id="r", partition_name="p", request_info=info, inputs=[],
        ))).body.request_info.resource_configs
        assert got[LORA_RESOURCE_KEY] == LoraReqConfig(adapter="sql") and got["kv"].prefix_root == kv.prefix_root


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


def test_specs_that_digest_an_adapter_differently_refuse_it(paths):
    """One digest pins an adapter on every spec's pools, so specs reading a file into
    different weights refuse it."""
    one = make_spec({"a": paths["a1"]})
    other = make_spec({"a": paths["a1"]}, convert_keys=lambda sd: {k: v * 2 for k, v in sd.items()})
    other.resource_key = "lora_other"
    with pytest.raises(ValueError, match="digests differently"):
        check_adapter_file(one, "a", paths["a1"], [other])
