"""Selective loading of ported Praxis-1 components from synthetic shards."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.praxis1 import weight_loader as loader
from mstar.model.praxis1.components.action_expert import ActionExpertBlock
from mstar.model.praxis1.components.encoder import EncoderTower, TrunkStack


def _encoder() -> EncoderTower:
    core = TrunkStack(
        depth=2,
        num_dense=1,
        dim=8,
        head_dim=4,
        num_kv_heads=1,
        dense_hidden=5,
        moe_hidden=3,
        num_experts=2,
        top_k=1,
        rope_effective_dim=4,
        rope_table_size=16,
    )
    return EncoderTower(core, dim=8, text_dim=6, video_dim=4, time_dim=8)


def _checkpoint(root, component, prefix, *, mutate=None):
    tensors = {}
    for key, value in component.state_dict().items():
        reference_key = key.replace(".ffn.linear.", ".ffn.0.linear.")
        reference_key = reference_key.replace(".ffn.down.weight", ".ffn.1.weight")
        tensors[prefix + reference_key] = value.clone().contiguous()
    if mutate is not None:
        mutate(tensors)
    save_file(tensors, root / "component.safetensors")
    save_file({"backbone.decoder.marker": torch.ones(1)}, root / "unrelated.safetensors")
    weight_map = {name: "component.safetensors" for name in tensors}
    weight_map["backbone.decoder.marker"] = "unrelated.safetensors"
    (root / loader.INDEX).write_text(json.dumps({"weight_map": weight_map}))
    return tensors


def test_encoder_and_moe_load_without_opening_unrelated_shards(tmp_path, monkeypatch):
    source = _encoder()
    expected = _checkpoint(tmp_path, source, "backbone.encoder.")
    target = _encoder()
    opened = []
    real_open = loader.safe_open

    def track(path, **kwargs):
        opened.append(path)
        return real_open(path, **kwargs)

    monkeypatch.setattr(loader, "safe_open", track)
    assert loader.required_shards(tmp_path, "backbone.encoder.") == ("component.safetensors",)
    loader.load_component(target, tmp_path, "backbone.encoder.")
    assert [path.rsplit("/", 1)[-1] for path in opened] == ["component.safetensors"]
    for key, value in target.state_dict().items():
        ref_key = key.replace(".ffn.linear.", ".ffn.0.linear.")
        ref_key = ref_key.replace(".ffn.down.weight", ".ffn.1.weight")
        assert torch.equal(value, expected["backbone.encoder." + ref_key])
    assert "core.blocks.1.ffn.expert_bias" in target.state_dict()


def test_action_expert_block_keeps_reference_names(tmp_path):
    source = ActionExpertBlock(dim=8, num_heads=2, head_dim=4, is_cross=True, kv_dim=12)
    _checkpoint(tmp_path, source, "action_expert.blocks.0.")
    target = ActionExpertBlock(dim=8, num_heads=2, head_dim=4, is_cross=True, kv_dim=12)
    loader.load_component(target, tmp_path, "action_expert.blocks.0.")
    assert all(torch.equal(target.state_dict()[key], value) for key, value in source.state_dict().items())


def test_ported_keys_match_reference_modules():
    """Check the remap against reference modules, not a synthetic M* inverse."""
    ref_backbone = pytest.importorskip("praxis.backbone")
    ref_layers = pytest.importorskip("praxis.layers")
    ref_expert = pytest.importorskip("praxis.action_expert")
    core = ref_layers.Transformer(
        depth=2,
        num_dense=1,
        dim=8,
        head_dim=4,
        num_kv_heads=1,
        dense_hidden=5,
        moe_hidden=3,
        num_experts=2,
        top_k=1,
        rope_effective_dim=4,
        rope_table_size=16,
        attention_backend="triton",
    )
    reference = ref_backbone.EncoderTower(
        core,
        dim=8,
        text_dim=6,
        video_dim=4,
        time_dim=8,
        mask_block_size=16,
    )
    port = _encoder()
    mapped = {loader._local_key(key): value for key, value in reference.state_dict().items()}
    assert set(mapped) == set(port.state_dict())
    assert {key: value.shape for key, value in mapped.items()} == {
        key: value.shape for key, value in port.state_dict().items()
    }
    ref_block = ref_expert.ActionExpertBlock(
        dim=8,
        num_heads=2,
        head_dim=4,
        is_cross=True,
        kv_dim=12,
    )
    port_block = ActionExpertBlock(dim=8, num_heads=2, head_dim=4, is_cross=True, kv_dim=12)
    assert {key: value.shape for key, value in ref_block.state_dict().items()} == {
        key: value.shape for key, value in port_block.state_dict().items()
    }


@pytest.mark.parametrize(
    "problem,pattern",
    [
        ("missing", "missing"),
        ("unexpected", "unexpected"),
        ("shape", "shape mismatch"),
        ("dtype", "dtype mismatch"),
    ],
)
def test_exact_component_contract(tmp_path, problem, pattern):
    source = ActionExpertBlock(dim=8, num_heads=2, head_dim=4, is_cross=True, kv_dim=12)

    def change(tensors):
        name = "action_expert.blocks.0.to_q.weight"
        if problem == "missing":
            del tensors[name]
        elif problem == "unexpected":
            tensors["action_expert.blocks.0.extra"] = torch.ones(1)
        elif problem == "shape":
            tensors[name] = torch.ones(1, 1)
        else:
            tensors[name] = tensors[name].to(torch.float16)

    _checkpoint(tmp_path, source, "action_expert.blocks.0.", mutate=change)
    with pytest.raises(loader.Praxis1WeightError, match=pattern):
        loader.load_component(source, tmp_path, "action_expert.blocks.0.")


def test_shard_index_mismatch_is_rejected(tmp_path):
    source = ActionExpertBlock(dim=8, num_heads=2, head_dim=4, is_cross=True, kv_dim=12)
    _checkpoint(tmp_path, source, "action_expert.blocks.0.")
    path = tmp_path / loader.INDEX
    index = json.loads(path.read_text())
    index["weight_map"]["action_expert.blocks.0.absent"] = "component.safetensors"
    path.write_text(json.dumps(index))
    with pytest.raises(loader.Praxis1WeightError, match="differs from the index"):
        loader.load_component(source, tmp_path, "action_expert.blocks.0.")
