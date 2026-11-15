"""CPU checks for the first LingBot-VLA 2.0 checkpoint-loading stage."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.components.lingbot_vla2_prefix import LingBotVLAPrefixConfig, load_prefix_weights
from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig
from mstar.model.components.qwen_vision import QwenVisionConfig, QwenVisionModel

VLM = "model.qwenvl_with_expert.qwenvl.model."


def _configs():
    vision = QwenVisionConfig(
        depth=1, hidden_size=12, intermediate_size=24, num_heads=2,
        in_channels=3, patch_size=2, temporal_patch_size=1,
        spatial_merge_size=2, num_position_embeddings=16,
        out_hidden_size=12, deepstack_visual_indexes=(0,),
    )
    text = Qwen3LMConfig(
        hidden_size=12, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=6, intermediate_size=24,
        vocab_size=17, tie_word_embeddings=True,
    )
    return vision, text


def _checkpoint(tmp_path, *, omit: str | None = None):
    vision_config, text_config = _configs()
    vision = QwenVisionModel(vision_config)
    text = Qwen3DenseLM(text_config)
    weights = {}
    expected = {}
    for index, (name, parameter) in enumerate(vision.named_parameters()):
        value = torch.full_like(parameter, (index + 1) / 100).contiguous()
        expected["vision." + name] = value
        weights[VLM + "visual." + name] = value
    for index, (name, parameter) in enumerate(text.named_parameters()):
        if name == "lm_head.weight":
            continue  # the released VLM ties this head to its input embedding
        value = torch.full_like(parameter, (index + 51) / 100).contiguous()
        expected["text." + name] = value
        base = VLM + "language_model." + name
        if name.endswith(".qkv_proj.weight"):
            q, k, v = value.split((12, 6, 6), dim=0)
            for suffix, part in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                weights[base.replace("qkv_proj", suffix)] = part.contiguous()
        elif name.endswith(".gate_up_proj.weight"):
            gate, up = value.split((24, 24), dim=0)
            weights[base.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
            weights[base.replace("gate_up_proj", "up_proj")] = up.contiguous()
        else:
            weights[base] = value
    weights["model.depth_align_embs"] = torch.zeros(1)  # training-only key
    if omit is not None:
        del weights[VLM + omit]
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in weights},
    }))
    return vision_config, text_config, expected


def test_config_reads_separate_qwen3_vl_architecture(tmp_path):
    ckpt = tmp_path / "checkpoint"
    qwen = tmp_path / "qwen"
    ckpt.mkdir()
    qwen.mkdir()
    vision, text = _configs()
    (ckpt / "config.json").write_text(json.dumps({"vlm_family": "qwen3_vl"}))
    (qwen / "config.json").write_text(json.dumps({
        "model_type": "qwen3_vl",
        "vision_config": asdict(vision),
        "text_config": {**asdict(text), "rope_scaling": {
            "mrope_interleaved": True, "mrope_section": [1, 1, 1],
        }},
    }))
    config = LingBotVLAPrefixConfig.from_pretrained(ckpt, qwen)
    assert config.vision.deepstack_visual_indexes == (0,)
    assert config.text.num_hidden_layers == 1
    assert config.mrope_section == (1, 1, 1)

    raw = json.loads((qwen / "config.json").read_text())
    raw["text_config"]["tie_word_embeddings"] = False
    (qwen / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="tied output head"):
        LingBotVLAPrefixConfig.from_pretrained(ckpt, qwen)

    (ckpt / "config.json").write_text(json.dumps({"vlm_family": "qwen2_vl"}))
    with pytest.raises(ValueError, match="vlm_family"):
        LingBotVLAPrefixConfig.from_pretrained(ckpt, qwen)


def test_prefix_loads_fused_vision_and_split_language_weights(tmp_path):
    vision_config, text_config, expected = _checkpoint(tmp_path)
    vision = QwenVisionModel(vision_config)
    text = Qwen3DenseLM(text_config)
    loaded_vision, loaded_text = load_prefix_weights(vision, text, tmp_path)

    assert loaded_vision == set(dict(vision.named_parameters()))
    assert loaded_text == set(dict(text.named_parameters()))
    for name, value in expected.items():
        module, key = name.split(".", 1)
        actual = dict((vision if module == "vision" else text).named_parameters())[key]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    torch.testing.assert_close(text.lm_head.weight, text.embed_tokens.weight, rtol=0, atol=0)


def test_prefix_refuses_missing_vision_weight(tmp_path):
    vision_config, text_config, _ = _checkpoint(tmp_path, omit="visual.pos_embed.weight")
    with pytest.raises(ValueError, match="vision checkpoint keys differ: 1 missing"):
        load_prefix_weights(QwenVisionModel(vision_config), Qwen3DenseLM(text_config), tmp_path)


def test_prefix_refuses_unmapped_language_weight_before_reading(tmp_path):
    vision_config, text_config, _ = _checkpoint(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    metadata = json.loads(index.read_text())
    metadata["weight_map"][VLM + "language_model.layers.0.not_a_layer.weight"] = "missing.safetensors"
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="language checkpoint keys differ: 0 missing.*1 extra"):
        load_prefix_weights(QwenVisionModel(vision_config), Qwen3DenseLM(text_config), tmp_path)


def test_prefix_requires_index_before_reading_weights(tmp_path):
    vision_config, text_config, _ = _checkpoint(tmp_path)
    (tmp_path / "model.safetensors.index.json").unlink()
    with pytest.raises(FileNotFoundError, match="LingBot-VLA prefix requires"):
        load_prefix_weights(QwenVisionModel(vision_config), Qwen3DenseLM(text_config), tmp_path)
