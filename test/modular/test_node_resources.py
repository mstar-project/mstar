"""Unit tests for node resource declaration and construction.

Models declare per-node resources as ``NodeResourceSpec`` lists; the
engine builds them at load time and exposes them per node through
``engine.node_resources``. Submodules receive their node's dict once,
at bind time.
"""

from __future__ import annotations

import dataclasses
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.kv_store import KVCacheConfig
from mstar.engine.resources import NodeResourceSpec, ScratchKVPool, ScratchKVSpec
from mstar.model.base import Model
from mstar.model.submodule_base import NodeInputs, NodeSubmodule


def _config(nodes: list[str]) -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=64,
        max_num_pages=8,
        page_size=8,
        nodes=nodes,
    )


class TestDefaultDeclaration:
    def test_default_wraps_each_config_unchanged(self):
        configs = [_config(["A"]), _config(["B", "C"])]
        specs = Model.get_node_resources(None, configs)

        assert [s.kv_cache_config for s in specs] == configs
        assert all(s.scratch == {} for s in specs)

    def test_override_can_extend_a_spec(self):
        class _Declares(Model):
            def get_node_resources(self, kv_cache_config):
                specs = super().get_node_resources(kv_cache_config)
                specs[0].scratch["scratch_kv"] = ScratchKVSpec(shape=(2, 4))
                return specs

        _Declares.__abstractmethods__ = frozenset()
        specs = _Declares().get_node_resources([_config(["A"])])
        assert specs[0].scratch["scratch_kv"].shape == (2, 4)
        assert specs[0].scratch["scratch_kv"].dtype is None


class TestSpecTypes:
    def test_scratch_spec_is_immutable(self):
        spec = ScratchKVSpec(shape=(1, 2, 3))
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.shape = (4,)

    def test_scratch_pool_holds_its_tensor(self):
        tensor = torch.zeros(2, 3)
        pool = ScratchKVPool(tensor)
        assert pool.tensor is tensor


class _StubSubmodule(NodeSubmodule):
    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs()

    def forward(self, graph_walk, engine_inputs, **kwargs):
        return {}


class TestSubmoduleBinding:
    def test_unbound_submodule_has_no_resources(self):
        assert _StubSubmodule().node_resources == {}

    def test_bind_replaces_the_dict(self):
        submodule = _StubSubmodule()
        resources = {"scratch_kv": ScratchKVPool(torch.zeros(1))}
        submodule.bind_node_resources(resources)
        assert submodule.node_resources is resources


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
