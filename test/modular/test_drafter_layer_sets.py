"""A drafter's KV layers join its target's pool as a layer set
(`PagedKVConfig.join`, `drafter_resources`).
"""
from __future__ import annotations

import pytest

from mstar.engine.resources.draft.drafter import DrafterNeeds, drafter_resources
from mstar.engine.resources.kv.config import KVLayerSet, KVSpec, PagedKVConfig


def _pool(**kw) -> PagedKVConfig:
    return PagedKVConfig(num_layers=3, num_kv_heads=2, num_qo_heads=4, head_dim=16, max_seq_len=64, **kw)


def test_a_join_appends_the_layers_and_refuses_another_geometry_or_a_repeat():
    config = _pool()
    assert config.join(KVLayerSet("kv", "draft", 2, 2, 16)) == (3, 2)
    assert config.num_layers == 5 and [list(group.layers) for group in config.groups] == [[0, 1, 2, 3, 4]]
    with pytest.raises(ValueError, match="joined twice"):
        config.join(KVLayerSet("kv", "draft", 1, 2, 16))
    for heads, dim in ((4, 16), (2, 8)):
        with pytest.raises(ValueError, match="its own KV storage geometry is not supported"):
            _pool().join(KVLayerSet("kv", "other", 1, heads, dim))


def test_drafter_resources_needs_the_paged_pool_it_names():
    class _Decl:
        needs = DrafterNeeds(kv=KVLayerSet("kv", "draft", 1, 2, 16))

        def node_resources(self, nodes):
            return []

    from mstar.engine.resources.draft.config import DraftConfig

    config = DraftConfig.parse({"method": "dspark", "model": "m", "num_draft_tokens": 3})
    with pytest.raises(ValueError, match="doesn't declare"):
        drafter_resources([], _Decl(), {"LLM"}, config)
    paged = KVSpec(resource_key="kv", nodes={"LLM"}, config=_pool())
    (out,) = drafter_resources([paged], _Decl(), {"LLM"}, config)
    assert out.config.layer_sets["draft"] == (3, 1)
