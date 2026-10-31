"""The shared Qwen vision tower against transformers' Qwen3-VL tower, deepstack on.

Tiny random weights on CPU, two images of different grids packed together. The tower's
attention resource is stood in for by per-frame SDPA, which is what the ragged kernel
computes over the segments ``vision_seq_lengths`` plans. Qwen3.5 runs the same tower
without deepstack.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_vl")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel  # noqa: E402

from mstar.model.components.qwen_vision import (  # noqa: E402
    VISION_ATTN,
    FrameSDPA,
    QwenVisionConfig,
    QwenVisionModel,
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)

GRID = [(1, 4, 6), (1, 2, 4)]
SIZES = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3, patch_size=4,
             temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=24)


@pytest.mark.parametrize("deepstack", [(), (0, 1)])
def test_tower_matches_transformers(deepstack):
    torch.manual_seed(0)
    reference = Qwen3VLVisionModel(Qwen3VLVisionConfig(
        hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=list(deepstack), **SIZES,
    )).eval()
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(0, 0.2)
    ours = QwenVisionModel(QwenVisionConfig(deepstack_visual_indexes=deepstack, **SIZES)).eval()
    missing, unexpected = ours.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    for block in ours.blocks:
        block.attn.bind_resources({VISION_ATTN: FrameSDPA(vision_seq_lengths(GRID))})

    patches = sum(t * h * w for t, h, w in GRID)
    pixels = torch.randn(patches, 3 * 2 * 4 * 4)
    indices, weights = vision_interpolation(GRID, ours.config.num_grid_per_side, 2, torch.device("cpu"))
    with torch.no_grad():
        want = reference(pixels, grid_thw=torch.tensor(GRID))
        got = ours(pixels, indices, weights, vision_position_ids(GRID, 2, torch.device("cpu")))
    if deepstack:
        got, features = got
        assert len(features) == len(want.deepstack_features) == len(deepstack)
        for mine, theirs in zip(features, want.deepstack_features, strict=True):
            torch.testing.assert_close(mine, theirs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got, want.pooler_output, rtol=1e-5, atol=1e-5)


def test_bf16_tower_matches_transformers_with_fp32_rope():
    """In bf16, transformers keeps the rotary tables fp32; the tower does too by default,
    and the features then agree bit for bit."""
    torch.manual_seed(0)
    reference = Qwen3VLVisionModel(Qwen3VLVisionConfig(
        hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[1], **SIZES,
    )).eval()
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(0, 0.2)
    reference = reference.to(torch.bfloat16)
    ours = QwenVisionModel(QwenVisionConfig(deepstack_visual_indexes=(1,), **SIZES))
    ours.load_state_dict(reference.state_dict(), strict=False)
    ours = ours.to(torch.bfloat16).eval()
    for block in ours.blocks:
        block.attn.bind_resources({VISION_ATTN: FrameSDPA(vision_seq_lengths(GRID))})
    patches = sum(t * h * w for t, h, w in GRID)
    pixels = torch.randn(patches, 3 * 2 * 4 * 4).to(torch.bfloat16)
    indices, weights = vision_interpolation(GRID, ours.config.num_grid_per_side, 2, torch.device("cpu"))
    with torch.no_grad():
        want = reference(pixels, grid_thw=torch.tensor(GRID))
        got, features = ours(pixels, indices, weights, vision_position_ids(GRID, 2, torch.device("cpu")))
    assert torch.equal(got, want.pooler_output)
    assert torch.equal(features[0], want.deepstack_features[0])


def test_qwen3_5_tower_keeps_its_key_and_rounding():
    """Qwen3.5 binds the resource under its own declared key and rounds the rotary tables
    to the activations' dtype, as before the tower was shared."""
    from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
    from mstar.model.qwen3_5.config import VISION_ATTN as QWEN3_5_VISION_ATTN

    tower = Qwen3_5VisionModel(QwenVisionConfig(**SIZES))
    assert tower.rope_dtype is None
    assert all(block.attn.resource_key == QWEN3_5_VISION_ATTN for block in tower.blocks)
    frames = FrameSDPA(vision_seq_lengths(GRID))
    for block in tower.blocks:
        block.attn.bind_resources({QWEN3_5_VISION_ATTN: frames})
        block.attn.bind_resources({})  # a node without the resource keeps the binding
        assert block.attn.ragged_attn is frames
    tower = tower.to(torch.bfloat16)
    patches = sum(t * h * w for t, h, w in GRID)
    indices, weights = vision_interpolation(GRID, tower.config.num_grid_per_side, 2, torch.device("cpu"))
    _, cos, _ = tower.embed(torch.randn(patches, 3 * 2 * 4 * 4).to(torch.bfloat16), indices, weights,
                            vision_position_ids(GRID, 2, torch.device("cpu")))
    assert cos.dtype == torch.bfloat16
