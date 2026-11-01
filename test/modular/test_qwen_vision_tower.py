"""The Qwen3-VL vision tower against transformers' Qwen3-VL tower, deepstack on.

Tiny random weights on CPU, two images of different grids packed together. The tower's
attention resource is stood in for by per-frame SDPA, which is what the ragged kernel
computes over the segments ``vision_seq_lengths`` plans.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_vl")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel  # noqa: E402

from mstar.model.components.qwen3_vit import (  # noqa: E402
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)
from mstar.model.components.qwen_vision import (  # noqa: E402
    VISION_ATTN,
    FrameSDPA,
    QwenVisionConfig,
    QwenVisionModel,
)

GRID = [(1, 4, 6), (1, 2, 4)]
SIZES = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3, patch_size=4,
             temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=24)


def _towers(deepstack: tuple[int, ...], dtype: torch.dtype = torch.float32):
    """transformers' tower with random weights, and ours loaded from it with per-frame SDPA."""
    torch.manual_seed(0)
    reference = Qwen3VLVisionModel(Qwen3VLVisionConfig(
        hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=list(deepstack), **SIZES,
    )).eval()
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(0, 0.2)
    reference = reference.to(dtype)
    ours = QwenVisionModel(QwenVisionConfig(deepstack_visual_indexes=deepstack, **SIZES))
    missing, unexpected = ours.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    ours = ours.to(dtype).eval()
    for block in ours.blocks:
        block.attn.bind_resources({VISION_ATTN: FrameSDPA(vision_seq_lengths(GRID))})
    return reference, ours


def _run(reference, ours, dtype: torch.dtype = torch.float32):
    patches = sum(t * h * w for t, h, w in GRID)
    pixels = torch.randn(patches, 3 * 2 * 4 * 4).to(dtype)
    indices, weights = vision_interpolation(GRID, ours.config.num_grid_per_side, 2, torch.device("cpu"))
    with torch.no_grad():
        want = reference(pixels, grid_thw=torch.tensor(GRID))
        got = ours(pixels, indices, weights, vision_position_ids(GRID, 2, torch.device("cpu")))
    return got, want


@pytest.mark.parametrize("deepstack", [(), (0, 1)])
def test_tower_matches_transformers(deepstack):
    (got, features), want = _run(*_towers(deepstack))
    assert len(features) == len(want.deepstack_features) == len(deepstack)
    for mine, theirs in zip(features, want.deepstack_features, strict=True):
        torch.testing.assert_close(mine, theirs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got, want.pooler_output, rtol=1e-5, atol=1e-5)


def test_bf16_tower_matches_transformers_with_fp32_rope():
    """In bf16, transformers keeps the rotary tables fp32; the tower does too, and the
    features then agree bit for bit."""
    (got, features), want = _run(*_towers((1,), torch.bfloat16), torch.bfloat16)
    assert torch.equal(got, want.pooler_output)
    assert torch.equal(features[0], want.deepstack_features[0])


def test_a_node_without_the_resource_keeps_the_binding():
    """The engine binds a node's resources into every layer; a tower its owner bound to
    per-frame attention keeps it on a node that declares no vision resource."""
    tower = QwenVisionModel(QwenVisionConfig(**SIZES))
    frames = FrameSDPA(vision_seq_lengths(GRID))
    for block in tower.blocks:
        block.attn.bind_resources({VISION_ATTN: frames})
        block.attn.bind_resources({})
        assert block.attn.ragged_attn is frames
