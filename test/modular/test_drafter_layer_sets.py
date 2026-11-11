"""A drafter's KV layers join its target's pool as a layer set
(`PagedKVConfig.join`, `drafter_resources`).

The expected values below (Qwen3 + DSpark) pin the pool's layout and its
prefix identity.
"""
from __future__ import annotations

import pytest
import torch
from qwen3_cpu import draft_block, engine, make_model, patch_cpu, write_checkpoints

from mstar.engine.resources.draft.drafter import DrafterNeeds, drafter_resources
from mstar.engine.resources.kv.config import KVLayerSet, KVSpec, PagedKVConfig

# the KV tensor, its sizing, its page groups, its prefix identity and the drafter's first layer
EXPECTED = {
    "shape": [5, 512, 2, 4, 2, 16], "stride": [131072, 256, 128, 32, 16, 1],
    "num_units": 512, "request_pages": 511, "tokens_per_page": 4, "groups": [([0, 1, 2, 3, 4], 4)],
    "fingerprint": "4d9df2247faa7e8125bb0bc0599021b2a7a752085508aac3560508673cdff786",
    "root": "851edfb9d9671c32bc421d30590eb9092dec908e7786d002b1ed6385391751d0",
    "first_layer": 3,
}


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    patch_cpu(monkeypatch)


def _facts(eng, first_layer: int) -> dict:
    kv = eng._resources["kv_cache"]
    config, tensor = kv.config, kv.kv_cache.tensor
    return {
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "num_units": config.num_units, "request_pages": config.request_pages,
        "tokens_per_page": config.tokens_per_page,
        "groups": [(list(group.layers), group.page_size) for group in config.groups],
        "fingerprint": kv.fingerprint().hex(), "root": kv._prefix_root.hex(),
        "first_layer": first_layer,
    }


def test_a_dspark_drafters_pool_layout(tmp_path):
    target, dspark = write_checkpoints(tmp_path / "dspark")
    model = make_model(target, draft_block(dspark, 3))
    eng = engine(model, prefix_cache=True)
    drafter = eng._submodules["LLM"].submodule.drafter
    assert eng._resources["kv_cache"].config.layer_sets == {"target": (0, 3), "dspark": (3, 2)}
    assert _facts(eng, drafter.backbone.layers[0].self_attn.attend._layer_offset) == EXPECTED


def test_a_page_size_set_after_the_join_keeps_every_drafter_layer_in_the_first_group(tmp_path):
    target, dspark = write_checkpoints(tmp_path)
    model = make_model(target, draft_block(dspark, 3))
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    config = specs["kv_cache"].config
    # the serving config's `resources:` block applies after the model declared (and joined) its pool
    specs["kv_cache"].apply_yaml_overrides(page_size=8, max_num_pages=64)
    assert [list(group.layers) for group in config.groups] == [[0, 1, 2, 3, 4]]
    assert config.tokens_per_page == 8 and config.layer_sets["dspark"] == (3, 2)


def test_a_plain_qwen3_stack_names_no_layer_set(monkeypatch):
    """Every non-drafter user of Qwen3DenseLM: its layers name no layer set, so
    the pool places them at 0, and its forward sets no per-layer cursor."""
    import mstar.model.components.qwen3_lm as lm_mod
    from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig

    config = Qwen3LMConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=48, vocab_size=64, rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=128,
    )
    with torch.device("meta"):
        lm = Qwen3DenseLM(config, attn_key="attn", kv_key="kv_cache", pos_key="rope")
    assert all(layer.self_attn._kv_layer_set is None for layer in lm.layers)
    seen = {}

    def run_layers(layers, x, norm, set_cursor=None, taps=None):
        seen["cursor"] = set_cursor
        return x

    monkeypatch.setattr(lm_mod, "run_layers", run_layers)

    class _Attend:
        def bind_step(self, label):
            pass

    lm.layers[0].self_attn.attend = _Attend()
    lm(torch.zeros(1, 32, device="meta"), label="main")
    assert seen["cursor"] is None


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


def test_a_drafter_whose_layers_name_no_layer_set_is_refused_at_load(tmp_path, monkeypatch):
    """They would write the target's first layers."""
    from mstar.model.components.dspark.drafter import DSparkDrafter

    target, dspark = write_checkpoints(tmp_path)
    model = make_model(target, draft_block(dspark, 3))
    real = DSparkDrafter.bind_resources

    def forgetting_the_set(self, resources):
        for layer in self.backbone.layers:
            layer.self_attn._kv_layer_set = None
            layer.self_attn.bind_resources(resources)
        real(self, resources)

    monkeypatch.setattr(DSparkDrafter, "bind_resources", forgetting_the_set)
    with pytest.raises(RuntimeError, match="layer set 'dspark' .* holds 2 layers, and 0 attention layers declare it"):
        engine(model)
