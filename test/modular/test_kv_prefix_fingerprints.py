"""The root names everything a cached page depends on except its own tokens.

A page holds the attention state of a prefix under one set of weights, one head
slice, one dtype, one page size, one attention kernel and one position scheme.
If a fingerprint leaves one of those out, two deployments that differ only in
that thing share a root, and the second one matches pages it never wrote. So
each of these tests changes one input and demands a different fingerprint.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import checkpoint_identity
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionConfig, PosScheme
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.base import Model


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _kv(dtype=torch.float32, **overrides) -> KVManager:
    cfg = dict(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
        max_num_pages=16, page_size=16,
    )
    cfg.update(overrides)
    return KVManager(
        cfg=KVConfig(**cfg), name="kv", joint_comm_group=None,
        transfer_engine_info=None, device=torch.device("cpu"), dtype=dtype,
    )


def _rope(**overrides) -> RopeManager:
    """A rope manager around a config, without building its tables."""
    rope = RopeManager.__new__(RopeManager)
    rope._config = PositionConfig(kv_cache="kv", **overrides)
    return rope


def _checkpoint(tmp_path: pathlib.Path, shard: bytes = b"x" * 100) -> pathlib.Path:
    (tmp_path / "config.json").write_bytes(b'{"hidden_size": 8}')
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(shard)
    return tmp_path


# ── the kv resource ─────────────────────────────────────────────────────


def test_a_kv_fingerprint_is_the_same_twice():
    assert _kv().fingerprint() == _kv().fingerprint(), (
        "one cache configuration fingerprinted two ways, so a restart could "
        "never match what it wrote"
    )


@pytest.mark.parametrize("overrides", [
    {"page_size": 32},
    {"prefix_cache_salt": "another tenant"},
])
def test_a_kv_fingerprint_changes_with_what_a_page_holds(overrides):
    assert _kv().fingerprint() != _kv(**overrides).fingerprint(), (
        f"{sorted(overrides)} left the root unchanged"
    )


def test_a_kv_fingerprint_changes_with_the_pool_dtype():
    assert _kv().fingerprint() != _kv(dtype=torch.bfloat16).fingerprint(), (
        "two pools of different dtype share a root"
    )


def test_a_kv_fingerprint_changes_with_the_world_size():
    kv = _kv()
    before = kv.fingerprint()

    kv._world_size = 2

    assert kv.fingerprint() != before, "a head slice is not in the root"


def test_a_kv_fingerprint_ignores_how_many_pages_the_pool_has():
    assert _kv().fingerprint() == _kv(max_num_pages=64).fingerprint(), (
        "pool size cannot change what a page holds, so it must not split the root"
    )


# ── positions ───────────────────────────────────────────────────────────


def test_a_position_fingerprint_is_the_same_twice():
    assert _rope().fingerprint() == _rope().fingerprint(), (
        "one rope configuration fingerprinted two ways"
    )


@pytest.mark.parametrize("overrides", [
    {"rope_theta": 500000.0},
    {"scheme": PosScheme.BLOCK},
    {"rope_scale": 4.0},
    {"interleave": True},
])
def test_a_position_fingerprint_changes_with_any_config_field(overrides):
    assert _rope().fingerprint() != _rope(**overrides).fingerprint(), (
        f"{sorted(overrides)} left the root unchanged"
    )


# ── attention ───────────────────────────────────────────────────────────


def test_an_attention_fingerprint_names_the_backend_that_was_built():
    from mstar.engine.resources.attn.dense import DenseAttentionManager

    paged = AttentionManager.__new__(AttentionManager)
    dense = DenseAttentionManager.__new__(DenseAttentionManager)

    assert paged.fingerprint() != dense.fingerprint(), (
        "a dense build that degraded to paged would keep the root it had"
    )


# ── preprocessing ───────────────────────────────────────────────────────


def test_preprocessing_fingerprints_separate_two_models():
    # `Model` is abstract, and the default reads only the class name, so the
    # unbound method over a stand-in is the whole of what ships
    class _One:
        pass

    class _Two:
        pass

    assert Model.preprocess_fingerprint(_One()) == "_One", (
        "the default fingerprint is not the class that preprocessed the prompt"
    )
    assert (
        Model.preprocess_fingerprint(_One())
        != Model.preprocess_fingerprint(_Two())
    ), "two models that template a prompt differently share a root"


# ── the checkpoint ──────────────────────────────────────────────────────


def test_a_checkpoint_identity_is_the_same_twice(tmp_path):
    root = _checkpoint(tmp_path)

    assert checkpoint_identity(root) == checkpoint_identity(root), (
        "one checkpoint identified two ways"
    )


def test_a_checkpoint_identity_changes_when_a_shard_changes_size(tmp_path):
    root = _checkpoint(tmp_path)
    before = checkpoint_identity(root)

    (root / "model-00001-of-00001.safetensors").write_bytes(b"x" * 101)

    assert checkpoint_identity(root) != before, "a replaced shard kept its root"


def test_a_checkpoint_identity_changes_with_the_config(tmp_path):
    root = _checkpoint(tmp_path)
    before = checkpoint_identity(root)

    (root / "config.json").write_bytes(b'{"hidden_size": 16}')

    assert checkpoint_identity(root) != before, "config.json is not in the root"


def test_an_index_is_read_in_place_of_the_shards(tmp_path):
    root = _checkpoint(tmp_path)
    (root / "model.safetensors.index.json").write_bytes(b'{"weight_map": {"a": "s"}}')
    with_index = checkpoint_identity(root)

    # the index names every shard and its tensors, so the shards need no walk
    (root / "model-00001-of-00001.safetensors").write_bytes(b"y" * 4096)

    assert checkpoint_identity(root) == with_index, (
        "the shard walk ran even though the checkpoint ships an index"
    )
