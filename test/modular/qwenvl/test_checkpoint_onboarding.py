"""Platform use case: safely onboard the supported Qwen3-VL MoE checkpoint."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from mstar.model.qwenvl.components import QwenVLForCausalLM, compute_mrope_cos_sin
from mstar.model.qwenvl.config import load_qwenvl_config, validate_qwenvl_config
from mstar.model.qwenvl.submodules import qwen_vl_position_ids

from ._helpers import tiny_config


def test_platform_uses_the_official_checkpoint_config(tmp_path):
    config = tiny_config()
    config.save_pretrained(tmp_path)

    loaded = load_qwenvl_config(str(tmp_path))

    assert type(loaded).__name__ == "Qwen3VLMoeConfig"
    assert type(loaded.text_config).__name__ == "Qwen3VLMoeTextConfig"
    assert loaded.text_config.hidden_size == 16
    assert loaded.vision_config.out_hidden_size == 16


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: setattr(config, "model_type", "qwen2_5_vl"), "Qwen3-VL MoE"),
        (lambda config: setattr(config.text_config, "decoder_sparse_step", 2), "every decoder layer"),
        (
            lambda config: config.text_config.rope_scaling.update(mrope_interleaved=False),
            "mrope_interleaved",
        ),
        (lambda config: setattr(config, "tie_word_embeddings", True), "untied output head"),
    ],
)
def test_platform_rejects_unsupported_architecture_variants(mutation, message):
    config = deepcopy(tiny_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_qwenvl_config(config)


def test_every_decoder_layer_is_the_target_sparse_moe_block():
    model = QwenVLForCausalLM(tiny_config())
    assert all(type(layer.mlp).__name__ == "ParallelSparseMoeBlock" for layer in model.model.layers)


def test_mrope_matches_qwen3_interleaved_layout_and_rejects_wrong_axes():
    positions = torch.tensor([[1, 2], [10, 20], [100, 200]])
    cos, sin = compute_mrope_cos_sin(
        positions, head_dim=8, rope_theta=10_000.0, mrope_section=(1, 2, 1), dtype=torch.float32
    )
    assert cos.shape == sin.shape == (2, 8)
    assert torch.allclose(cos[0], torch.cos(torch.tensor([1.0, 1.0, 1.0, 0.001] * 2)))
    with pytest.raises(ValueError, match=r"Expected \[3, tokens\]"):
        compute_mrope_cos_sin(
            torch.zeros(2, 3), head_dim=8, rope_theta=10_000.0, mrope_section=(1, 2, 1), dtype=torch.float32
        )


def test_30b_interleaved_mrope_boundaries():
    positions = torch.tensor([[3, 4, 5], [30, 40, 50], [300, 400, 500]])
    cos, _ = compute_mrope_cos_sin(
        positions, head_dim=128, rope_theta=5_000_000.0, mrope_section=(24, 20, 20), dtype=torch.float32
    )
    inv = 1.0 / (5_000_000.0 ** (torch.arange(0, 128, 2).float() / 128))
    assert torch.allclose(cos[0, 1], torch.cos(torch.tensor(30.0) * inv[1]))
    assert torch.allclose(cos[0, 2], torch.cos(torch.tensor(300.0) * inv[2]))
    assert torch.allclose(cos[0, 60], torch.cos(torch.tensor(3.0) * inv[60]))


def test_image_position_builder_accepts_grid_and_rejects_bad_processor_output():
    config = tiny_config()
    marker = config.image_token_id
    ids = torch.tensor([9, marker, marker, marker, marker, 10])
    assert qwen_vl_position_ids(ids, torch.tensor([[1, 4, 4]]), config).tolist() == [
        [0, 1, 1, 1, 1, 3],
        [0, 1, 1, 2, 2, 3],
        [0, 1, 2, 1, 2, 3],
    ]
    with pytest.raises(ValueError, match="no corresponding"):
        qwen_vl_position_ids(torch.tensor([marker]), None, config)
    with pytest.raises(ValueError, match="not divisible"):
        qwen_vl_position_ids(torch.tensor([marker]), torch.tensor([[1, 3, 4]]), config)
    with pytest.raises(ValueError, match="does not match"):
        qwen_vl_position_ids(torch.tensor([marker, 7]), torch.tensor([[1, 4, 4]]), config)
    with pytest.raises(ValueError, match="without matching"):
        qwen_vl_position_ids(torch.tensor([7]), torch.tensor([[1, 4, 4]]), config)
