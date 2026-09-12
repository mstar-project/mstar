"""Platform use case: load every Qwen3-VL-30B-A3B tensor exactly once."""

from __future__ import annotations

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components import ParallelSparseMoeBlock
from mstar.model.qwenvl.components import QwenVLForCausalLM
from mstar.model.qwenvl.weight_loader import (
    iter_qwen_vl_text_weights,
    iter_qwen_vl_vision_weights,
    load_qwen_vl_text_weights,
    load_qwen_vl_vision_weights,
    remap_qwen_vl_text_key,
    remap_qwen_vl_vision_key,
    require_complete_weight_load,
)

from ._helpers import tiny_config


def _hf_weights(model):
    def published_key(name):
        if name.startswith("model."):
            return f"model.language_model.{name.removeprefix('model.')}"
        return name

    weights = []
    q_size = model.config.text_config.num_attention_heads * model.config.text_config.head_dim
    kv_size = model.config.text_config.num_key_value_heads * model.config.text_config.head_dim
    for name, parameter in model.named_parameters():
        if name.endswith("qkv_proj.weight") or name.endswith("qkv_proj.bias"):
            q, k, v = parameter.detach().split([q_size, kv_size, kv_size], dim=0)
            weights.extend(
                [
                    (published_key(name.replace("qkv_proj", "q_proj")), q),
                    (published_key(name.replace("qkv_proj", "k_proj")), k),
                    (published_key(name.replace("qkv_proj", "v_proj")), v),
                ]
            )
        elif name.endswith("gate_up_proj.weight"):
            gate, up = parameter.detach().chunk(2, dim=0)
            weights.extend(
                [
                    (published_key(name.replace("gate_up_proj", "gate_proj")), gate),
                    (published_key(name.replace("gate_up_proj", "up_proj")), up),
                ]
            )
        elif name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
            # Published Qwen3 fused experts use [E, input, output], while M*
            # stores [E, output, input] for torch.linear and fused kernels.
            weights.append((published_key(name), parameter.detach().transpose(1, 2).contiguous()))
        else:
            weights.append((published_key(name), parameter.detach().clone()))
    return weights


def test_platform_maps_only_published_checkpoint_names_to_serving_graph():
    assert (
        remap_qwen_vl_text_key("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        remap_qwen_vl_text_key("model.language_model.layers.0.mlp.experts.gate_up_proj")
        == "model.layers.0.mlp.experts.gate_up_proj"
    )
    assert remap_qwen_vl_text_key("lm_head.weight") == "lm_head.weight"
    assert remap_qwen_vl_text_key("language_model.lm_head.weight") is None
    assert remap_qwen_vl_text_key("model.layers.0.self_attn.q_proj.weight") is None
    assert remap_qwen_vl_text_key("model.language_model.rotary_emb.inv_freq") is None
    assert remap_qwen_vl_vision_key("model.visual.patch_embed.proj.weight") == "patch_embed.proj.weight"
    assert remap_qwen_vl_vision_key("visual.patch_embed.proj.weight") is None
    assert remap_qwen_vl_vision_key("unrelated.weight") is None


def test_platform_rejects_partially_loaded_30b_moe_text_checkpoint():
    model = QwenVLForCausalLM(tiny_config())
    loaded = load_qwen_vl_text_weights(model, _hf_weights(model))
    require_complete_weight_load(model, loaded, "text")
    with torch.no_grad():
        model.lm_head.weight.fill_(0.25)
    assert torch.equal(model.lm_head.weight, torch.full_like(model.lm_head.weight, 0.25))
    missing_one = set(dict(model.named_parameters())) - {next(iter(dict(model.named_parameters())))}
    try:
        require_complete_weight_load(model, missing_one, "text")
    except RuntimeError as error:
        assert "missed 1 parameters" in str(error)
    else:  # pragma: no cover - required strictness assertion
        raise AssertionError("incomplete QwenVL checkpoint must fail")


def test_fused_loader_routes_qkv_gate_up_and_untied_head():
    model = QwenVLForCausalLM(tiny_config())
    weights = _hf_weights(model)
    head = torch.full_like(model.lm_head.weight, 0.25)
    weights = [(name, head if name == "lm_head.weight" else value) for name, value in weights]
    loaded = load_qwen_vl_text_weights(model, weights)
    assert any(name.endswith("qkv_proj.weight") for name in loaded)
    assert any(name.endswith("mlp.experts.gate_up_proj") for name in loaded)
    assert any(name.endswith("mlp.experts.down_proj") for name in loaded)
    assert torch.equal(model.lm_head.weight, head)


def test_published_fused_experts_are_sliced_for_tp2():
    block = ParallelSparseMoeBlock(
        hidden_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        comm_group=CommGroup(my_global_rank=1, my_group_rank=1, group_members=[0, 1]),
    )
    gate_up = torch.arange(2 * 4 * 8).reshape(2, 4, 8)
    down = torch.arange(2 * 4 * 4).reshape(2, 4, 4)

    block.experts.gate_up_proj.weight_loader(block.experts.gate_up_proj, gate_up)
    block.experts.down_proj.weight_loader(block.experts.down_proj, down)

    gate_up_execution_order = gate_up.transpose(1, 2)
    expected_gate_up = torch.cat((gate_up_execution_order[:, 2:4], gate_up_execution_order[:, 6:8]), dim=1)
    assert torch.equal(block.experts.gate_up_proj, expected_gate_up)
    assert torch.equal(block.experts.down_proj, down.transpose(1, 2)[:, :, 2:4])


def test_iterators_stream_only_the_published_checkpoint_prefixes(monkeypatch):
    import mstar.model.qwenvl.weight_loader as loader

    calls = []

    def fake_iter(directory, *, device, prefix):
        calls.append((directory, device, prefix))
        yield prefix + "weight", torch.ones(1)

    monkeypatch.setattr(loader, "iter_safetensors_shards", fake_iter)
    assert [key for key, _ in iter_qwen_vl_text_weights("snapshot", "cpu")] == [
        "model.language_model.weight",
        "lm_head.weight",
    ]
    assert [key for key, _ in iter_qwen_vl_vision_weights("snapshot", "cpu")] == ["model.visual.weight"]
    assert [prefix for _, _, prefix in calls] == ["model.language_model.", "lm_head.", "model.visual."]


def test_vision_loader_uses_the_same_prefix_contract_as_streaming():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.zeros(2, 2))
    source = torch.full((2, 2), 0.5)
    assert load_qwen_vl_vision_weights(module, [("model.visual.weight", source)]) == {"weight"}
    assert torch.equal(module.weight, source)
