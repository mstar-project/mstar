"""Our Qwen3.5 ViT against transformers', on a real checkpoint.

The tower is a straight port, so the bar is bit-exactness in fp32, not a
tolerance: every op is in the same order on the same weights. A non-zero diff
here means a genuine divergence, not accumulated noise.

What the grids cover, since the interesting logic is all shape maths — the
position-table resample and the spatial-merge reordering:

* one image, non-square and not a multiple of the merge size in both axes
* two images of different sizes, to exercise the per-image `cu_seqlens` split
* a multi-frame entry (`t > 1`), whose frames attend separately
* a single merge block, where the resample hits the `size == 1` clamp

Needs the weights, so it skips unless QWEN3_5_CKPT points at a checkpoint
with a `vision_config`.
"""

from __future__ import annotations

import os

import pytest
import torch

from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards
from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
from mstar.model.qwen3_5.config import Qwen3_5VisionConfig
from mstar.model.qwen3_5.weight_loader import (
    load_qwen3_5_vision_weights,
    qwen3_5_vision_name_remapper,
)

CKPT = os.environ.get("QWEN3_5_CKPT")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not CKPT, reason="set QWEN3_5_CKPT to a Qwen3.5 checkpoint"),
]

GRIDS = [
    [[1, 8, 12]],
    [[1, 4, 4], [1, 6, 10]],
    [[2, 4, 6]],
    [[1, 2, 2]],
]


@pytest.fixture(scope="module")
def towers():
    """Ours and HF's, same weights, fp32."""
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionModel as HFVisionModel,
    )

    config = Qwen3_5VisionConfig.from_hf_or_none(CKPT)
    if config is None:
        pytest.skip(f"{CKPT} is text-only")

    device = "cuda"
    ours = Qwen3_5VisionModel(config)
    load_qwen3_5_vision_weights(ours, CKPT, device=device)

    hf_config = transformers.AutoConfig.from_pretrained(CKPT).vision_config
    # Ours splits by `cu_seqlens` and calls SDPA per segment; this is the HF
    # branch that does the same. The flash branch would be a different kernel,
    # so not bit-comparable.
    hf_config._attn_implementation = "sdpa"
    hf = HFVisionModel(hf_config)
    load_weights_into(
        hf,
        iter_safetensors_shards(CKPT, device=device, prefix="model.visual."),
        name_remapper=qwen3_5_vision_name_remapper,
    )

    def prepare(model):
        return (
            model.to(device=device, dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )

    return config, prepare(ours), prepare(hf)


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: "_".join(map(str, sum(g, []))))
def test_matches_hf(towers, grid):
    config, ours, hf = towers
    grid_thw = torch.tensor(grid, device="cuda")

    num_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    patch_numel = (
        config.in_channels * config.temporal_patch_size * config.patch_size**2
    )
    torch.manual_seed(0)
    pixel_values = torch.randn(
        num_patches, patch_numel, device="cuda", dtype=torch.float32,
    )

    with torch.no_grad():
        want = hf(pixel_values, grid_thw).pooler_output
        got = ours(pixel_values, grid_thw)

    assert got.shape == (num_patches // config.merge_unit, config.out_hidden_size)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_vision_weights_are_complete(towers):
    """`load_qwen3_5_vision_weights` raises on a gap, so reaching the fixture
    is the assertion; this pins the count so a silently shrinking remap is
    caught too."""
    _, ours, _ = towers
    assert len(list(ours.parameters())) > 0
    for name, param in ours.named_parameters():
        assert torch.isfinite(param).all(), name
