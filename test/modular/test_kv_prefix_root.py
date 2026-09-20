"""One root per KV resource, folded from everything its pages depend on.

A page is only reusable under the build that wrote it, so the index hangs from a
root that folds the checkpoint, the preprocessing, and the configuration of the
KV resource and of every resource planned against it. Anything left out of that
fold is a way for one deployment to match pages another one wrote.

The other half is where the cache stays shut: above one rank, because the ranks
walk their indexes independently and would match different lengths.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVSpec
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.position.manager import RopeManager

KV = "kv"
ROPE = "rope"


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


def _kv(dtype=torch.float32, **overrides) -> KVManager:
    cfg = dict(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
        max_num_pages=16, page_size=16,
    )
    cfg.update(overrides)
    return KVManager(
        cfg=KVConfig(**cfg), name=KV, joint_comm_group=None,
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
    assert _root() == _root()


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

    assert kv._index is None and kv._prefix_root is None


def test_the_index_opens_empty():
    kv = _kv()

    _root(kv=kv)

    assert kv._index is not None
    assert kv._index.lookup([b"anything"]) == [], "the index opened with entries"


# ── the keys the request brought ────────────────────────────────────────


def test_ingest_puts_the_requests_keys_on_the_stream():
    kv = _kv()

    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": [b"k0", b"k1"]}))

    assert kv._streams["r0"]["main"].keys == [b"k0", b"k1"]


def test_a_label_the_request_did_not_key_carries_none():
    kv = _kv()
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": [b"k0"]}))

    stream = kv._ensure_label("r0", "cfg_text")

    assert stream.keys is None, "an unkeyed label was given another label's keys"


def test_a_request_can_opt_out_of_the_cache():
    kv = _kv()

    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": [b"k0"]}, prefix_cache=False,
    ))

    assert kv._streams["r0"]["main"].keys is None, (
        "an opted-out request still carried its keys onto the stream"
    )
