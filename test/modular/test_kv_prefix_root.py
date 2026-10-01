"""One root per KV resource, folded from everything its pages depend on.

A page is only reusable under the build that wrote it, so the index hangs from a
root that folds the checkpoint, the preprocessing, and the configuration of the
KV resource and of every resource planned against it. Anything left out of that
fold is a way for one deployment to match pages another one wrote.

The other half is where the cache stays shut: above one rank, because the ranks
walk their indexes independently and would match different lengths; and for a
model that names no checkpoint, until the deployment sets a salt, because two
builds that differ only in their weights would share every key.

Where it opens, the engine says so, and warns a cache with no host pages: a
cached prompt reserves none of its pages, so nothing bounds how many requests
decode at once, and with nothing to offload a full pool holds them until they
time out.
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources import Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.base import PrefixStream

KV = "kv"
ROPE = "rope"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the host pool pins its memory"
)


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


class _Model:
    """Names a checkpoint and a preprocessing identity, nothing else."""

    def __init__(self, checkpoint=None, name="_Model"):
        self._checkpoint = checkpoint
        self._name = name

    def checkpoint_path(self):
        return self._checkpoint

    def preprocess_fingerprint(self):
        return self._name

    def prefix_key_streams(self):
        return {}


class _Declaring(_Model):
    """Keys ``main`` on the KV resource from its text walk."""

    def prefix_key_streams(self):
        return {KV: {"main": PrefixStream("text_inputs", "ids", "prefill_text", "decode")}}


def _checkpoint(root, config: bytes) -> str:
    root.mkdir()
    (root / "config.json").write_bytes(config)
    (root / "model-00001.safetensors").write_bytes(b"x" * 100)
    return str(root)


def _kv(dtype=torch.float32, **overrides) -> KVManager:
    cfg = dict(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
        max_num_pages=16, page_size=16,
    )
    cfg.update(overrides)
    return KVManager(
        cfg=PagedKVConfig(**cfg), name=KV, joint_comm_group=None,
        transfer_engine_info=None, device=torch.device("cpu"), dtype=dtype,
    )


def _rope(**overrides) -> RopeManager:
    rope = RopeManager.__new__(RopeManager)
    rope._config = PositionConfig(kv_cache=KV, **overrides)
    return rope


def _root(kv=None, rope=None, model=None) -> bytes:
    """Run the engine's fold over one KV resource and one resource above it."""
    kv = kv or _kv()
    rope = rope or _rope()
    engine = Engine.__new__(Engine)
    engine._resources = {KV: kv, ROPE: rope}
    specs = {
        KV: KVSpec(resource_key=KV, nodes={"LLM"}, config=kv.config),
        ROPE: PositionSpec(
            resource_key=ROPE, nodes={"LLM"}, config=rope._config,
        ),
    }
    engine._open_prefix_caches(specs, model or _Model())
    return kv._prefix_root


# ── what moves the root ─────────────────────────────────────────────────


def test_the_same_build_roots_the_same_way_twice():
    assert _root() == _root(), (
        "two builds of one deployment rooted differently, so neither could "
        "match what the other wrote"
    )


def test_the_root_changes_with_the_kv_resources_own_config():
    assert _root() != _root(kv=_kv(page_size=32)), "page size is not in the root"
    assert _root() != _root(kv=_kv(prefix_cache_salt="tenant-b")), (
        "the salt is not in the root"
    )


def test_the_root_changes_with_a_resource_planned_against_the_cache():
    assert _root() != _root(rope=_rope(rope_theta=500000.0)), (
        "the position config is not in the root, so a re-scaled rope would "
        "match pages written under the old one"
    )


def test_the_root_changes_with_the_preprocessing_identity():
    assert _root() != _root(model=_Model(name="_Other")), (
        "two models that build a prompt differently share a root"
    )


def test_the_root_changes_with_the_checkpoint(tmp_path):
    (tmp_path / "config.json").write_bytes(b'{"hidden_size": 8}')
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 100)
    before = _root(model=_Model(checkpoint=str(tmp_path)))

    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 200)

    assert _root(model=_Model(checkpoint=str(tmp_path))) != before, (
        "the weights are not in the root"
    )


# ── where the cache stays shut ──────────────────────────────────────────


def test_a_rank_above_one_never_opens_its_index():
    kv = _kv()
    kv._world_size = 2

    _root(kv=kv)

    assert kv._index is None and kv._prefix_root is None, (
        "the index opened at a world size that cannot agree on a match length"
    )


def test_a_deployment_can_turn_the_cache_off():
    kv = _kv(prefix_cache=False)

    _root(kv=kv)

    assert kv._index is None and kv._prefix_root is None, (
        "a deployment that turned the cache off opened an index anyway"
    )


def test_a_model_naming_no_checkpoint_keeps_its_cache_shut(caplog):
    kv = _kv()

    with caplog.at_level(logging.INFO, logger=Engine.__module__):
        _root(kv=kv, model=_Declaring())

    assert kv._index is None and kv._prefix_root is None, (
        "the cache opened under a root without the weights, so two builds that "
        "differ only in their weights would share every key"
    )
    lines = [r.getMessage() for r in caplog.records if "prefix cache off" in r.getMessage()]
    assert len(lines) == 1 and KV in lines[0] and "prefix_cache_salt" in lines[0], (
        "the cache stayed shut without naming the resource or what opens it"
    )


def test_a_salt_opens_the_cache_of_a_model_naming_no_checkpoint():
    kv = _kv(prefix_cache_salt="deployment")

    _root(kv=kv, model=_Declaring())

    assert kv._index is not None, "a deployment that set a salt still had its cache shut"


def test_a_model_naming_a_checkpoint_opens_its_cache(tmp_path):
    kv = _kv()

    _root(kv=kv, model=_Declaring(checkpoint=_checkpoint(tmp_path / "a", b"{}")))

    assert kv._index is not None, "a model that named its weights had its cache shut"


def test_two_checkpoints_root_the_cache_apart(tmp_path):
    a = _checkpoint(tmp_path / "a", b'{"hidden_size": 8}')
    b = _checkpoint(tmp_path / "b", b'{"hidden_size": 16}')

    assert _root(model=_Declaring(checkpoint=a)) != _root(model=_Declaring(checkpoint=b)), (
        "two checkpoints rooted the cache the same way, so each would match "
        "pages the other wrote"
    )


def test_the_index_opens_empty():
    kv = _kv()

    _root(kv=kv)

    assert kv._index is not None, "the cache was asked to open and did not"
    assert kv._index.lookup([b"anything"]) == [], "the index opened with entries"


# ── what the engine says when it opens ──────────────────────────────────


def _said_at_load(caplog, kv, model) -> tuple[list[str], list[str]]:
    """The engine's info and warning lines from opening ``kv``."""
    with caplog.at_level(logging.INFO, logger=Engine.__module__):
        _root(kv=kv, model=model)
    said = [r for r in caplog.records if r.name == Engine.__module__]
    return (
        [r.getMessage() for r in said if r.levelno == logging.INFO],
        [r.getMessage() for r in said if r.levelno == logging.WARNING],
    )


@requires_cuda
def test_a_keyed_node_with_host_pages_says_what_opened_and_no_more(caplog):
    info, warnings = _said_at_load(
        caplog, _kv(prefix_cache_salt="deployment", cpu_offload_pages=4), _Declaring(),
    )

    assert len(info) == 1 and "main" in info[0] and "LLM" in info[0], (
        "the cache opened without saying which label it keys on which node"
    )
    assert warnings == [], "a cache with host pages to offload to was warned it had none"


def test_a_keyed_node_without_host_pages_is_warned_its_decode_can_hold(caplog):
    info, warnings = _said_at_load(caplog, _kv(prefix_cache_salt="deployment"), _Declaring())

    assert len(info) == 1 and "main" in info[0] and "LLM" in info[0], (
        "the cache opened without saying which label it keys on which node"
    )
    assert len(warnings) == 1 and "cpu_offload_pages" in warnings[0], (
        "a cache with nothing to offload opened without saying a full pool "
        "holds its decode until the requests time out"
    )
    assert "max_concurrent_requests" in warnings[0], (
        "the warning names one way out of a full pool and not the other"
    )


@pytest.mark.parametrize("overrides, model", [
    ({}, _Model), ({"prefix_cache": False}, _Declaring),
], ids=["nothing declared", "cache off"])
def test_an_uncached_node_says_nothing_at_load(caplog, overrides, model):
    info, warnings = _said_at_load(caplog, _kv(**overrides), model())

    assert info == [] and warnings == [], (
        "a node with no cache in use logged one, or a warning about one"
    )


# ── the keys the request brought ────────────────────────────────────────


def test_ingest_puts_the_requests_keys_on_the_stream():
    kv = _kv()

    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": [b"k0", b"k1"]}))

    assert kv._streams["r0"]["main"].chain.keys == [b"k0", b"k1"], (
        "the chain the request brought never reached its stream"
    )


def test_a_label_the_request_did_not_key_carries_none():
    kv = _kv()
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": [b"k0"]}))

    stream = kv._ensure_label("r0", "cfg_text")

    assert stream.chain is None, "an unkeyed label was given another label's keys"


def test_a_request_can_opt_out_of_the_cache():
    kv = _kv()

    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": [b"k0"]}, prefix_cache=False,
    ))

    assert kv._streams["r0"]["main"].chain is None, (
        "an opted-out request still carried its keys onto the stream"
    )


def test_the_engine_hands_each_cache_the_walks_its_streams_name():
    kv = _kv(prefix_cache_salt="deployment")
    _root(kv=kv, model=_Declaring())

    assert kv._keyed_walks == {"main": {"prefill_text", "decode"}}, (
        "the cache was opened without the walks its keys describe, so every "
        "walk's pages would be filed under them"
    )


# ── which resources are offered the root ────────────────────────────────


class _Recording(Resource):
    """Keeps state across requests of its own, and records the root it is handed."""

    def __init__(self):
        self.roots: list[bytes] = []

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def enable_prefix_cache(self, root, walks=None):
        self.roots.append(root)


class _Plain(Resource):
    """Keeps nothing across requests, so has nothing to open."""

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError


def _open(resources: dict) -> None:
    engine = Engine.__new__(Engine)
    engine._resources = resources
    kv = resources[KV]
    engine._open_prefix_caches(
        {KV: KVSpec(resource_key=KV, nodes={"LLM"}, config=kv.config)}, _Model(),
    )


def test_every_resource_is_offered_the_root_not_just_the_kv_cache():
    other = _Recording()

    _open({KV: _kv(), "other": other})

    assert len(other.roots) == 1 and isinstance(other.roots[0], bytes), (
        "a resource that is not a KVManager was never asked to open its cache"
    )


def test_a_resource_with_nothing_to_open_leaves_the_cache_beside_it_open():
    kv = _kv()

    _open({KV: kv, "plain": _Plain()})

    assert kv._index is not None, (
        "a resource with nothing to open kept the cache beside it shut"
    )
