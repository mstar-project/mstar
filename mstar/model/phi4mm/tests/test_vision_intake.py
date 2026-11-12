"""Phi-4-multimodal's image intake on the host: the processor's crop choice,
patch mask and token count from the grid alone, the node's pack plan over that
grid, and, where the checkpoint is cached, the pixels and masks of the
reference ``Phi4MMImageProcessor`` on synthetic images."""

from __future__ import annotations

import glob

import numpy as np
import pytest
import torch

from mstar.model.phi4mm.components.vision import Phi4MMVisionConfig
from mstar.model.phi4mm.tests._paths import snapshot
from mstar.model.phi4mm.vision import (
    GLB_GN,
    GLOBAL_PIXEL_VALUES,
    MASKED,
    PIXEL_VALUES,
    SUB_GN,
    ImageGrid,
    hd_layout,
    image_intake,
    image_plan,
    image_tokens,
    navit_position_ids,
    step_indices,
)

CONFIG = Phi4MMVisionConfig()
SNAPSHOT = snapshot() and next(iter(glob.glob(str(snapshot() / "processing_phi4mm.py"))), None)


@pytest.mark.parametrize(("size", "grid", "tokens"), [
    # one crop, no pad: 16 x 16 cells, each image row and the global view's closed by sub_GN
    ((448, 448), (1, 1, 32, 32), 16 * 17 + 1 + 16 * 17),
    # the oracle's images (width, height): bagel, food_small, food_wide
    ((464, 427), (1, 2, 32, 35), 577),
    ((384, 512), (2, 1, 43, 32), 647),
    ((1500, 900), (3, 4, 77, 128), 2808),
    # 8 x 6 crops is over the budget of 36: the closest aspect ratio, 4 x 3, unpadded
    ((3584, 2688), (3, 4, 96, 128), 48 * 65 + 1 + 272),
    # smaller than a crop: upscaled to 448 wide, the bottom padded
    ((200, 100), (1, 1, 16, 32), 8 * 17 + 1 + 272),
])
def test_crop_grid_and_tokens(size, grid, tokens):
    layout = hd_layout(*size, CONFIG)
    assert tuple(layout.grid) == grid
    assert image_tokens(layout.grid, CONFIG) == tokens == image_plan(grid, 32).tokens


def test_an_extreme_aspect_ratio_is_a_400():
    with pytest.raises(ValueError, match="aspect ratio"):
        hd_layout(20000, 10, CONFIG)


def test_intake_takes_float_or_bytes_and_ships_crops_as_bytes():
    rng = np.random.RandomState(0)
    pixels = torch.from_numpy(rng.randint(0, 256, (3, 427, 464), dtype=np.uint8))
    from_bytes = image_intake(pixels, {}, CONFIG)
    from_float = image_intake(pixels.float() / 255, {}, CONFIG)  # as `Model.load_image` gives it
    crops = from_bytes.inputs[PIXEL_VALUES]
    assert crops.dtype == torch.uint8 and crops.shape == (2, 3, 448, 448)
    assert from_bytes.inputs[GLOBAL_PIXEL_VALUES].dtype == torch.float32
    assert from_bytes.inputs[GLOBAL_PIXEL_VALUES].shape == (1, 3, 448, 448)
    for name in (PIXEL_VALUES, GLOBAL_PIXEL_VALUES):
        assert torch.equal(from_bytes.inputs[name], from_float.inputs[name])
    assert from_bytes.grids == [(1, 2, 32, 35)] and from_bytes.tokens == [577]
    # resized to 486 x 448, then padded white to two crops' width
    assert (crops[1, :, :, 486 - 448:] == 255).all() and not (crops[1, :, :, 486 - 449] == 255).all()


def test_the_plan_packs_valid_patches_and_pools_padded_ones_at_odd_edges():
    # food_wide: a 3 x 4 grid whose last crop row holds 13 of its 32 patch rows
    grid = ImageGrid(3, 4, 77, 128)
    plan = image_plan(tuple(grid), 32)
    assert plan.segments == (1024,) * 9 + (13 * 32,) * 4
    assert plan.patches == sum(plan.segments) == plan.position_ids.numel() == 1024 + plan.crop_rows.numel()
    kept_rows, kept_cols = 39, 64
    sub = plan.pool_index[:kept_rows * (kept_cols + 1)].view(kept_rows, kept_cols + 1, 4)
    assert (sub[:, -1] == SUB_GN).all()
    assert (plan.pool_index[kept_rows * (kept_cols + 1)] == GLB_GN).all()
    # the last kept row pools patch row 76 (valid) with row 77 (padding)
    cells = sub[:, :-1]
    assert (cells[-1, :, 2:] == MASKED).all() and (cells[-1, :, :2] >= 0).all()
    assert (cells[:-1] >= 0).all()
    # every packed patch is pooled exactly once except the dropped ones past row 76: none here
    pooled = plan.pool_index[plan.pool_index >= 0]
    assert pooled.unique().numel() == pooled.numel() == plan.patches


def test_navit_position_ids():
    assert torch.equal(navit_position_ids(32, 32, 32), torch.arange(1024))
    # 13 valid rows spread over the 32-row table, as SiglipVisionEmbeddings buckets them
    ids = navit_position_ids(13, 32, 32).view(13, 32)
    assert torch.equal(ids[:, 0] // 32, torch.tensor([0, 2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27, 29]))
    assert torch.equal(ids[0], torch.arange(32))


def test_step_indices_offset_each_image_of_the_step():
    plans = [image_plan((1, 2, 32, 35), 32), image_plan((2, 1, 43, 32), 32)]
    idx = step_indices(plans, 32)
    total = sum(p.patches for p in plans)
    assert idx.patch_src.numel() == idx.position_ids.numel() == total
    # globals are the conv's first two images, then image 0's two crops, then image 1's
    assert torch.equal(idx.patch_src[:1024], torch.arange(1024))
    second = plans[0].patches
    assert torch.equal(idx.patch_src[second:second + 1024], torch.arange(1024, 2048))
    assert idx.patch_src[second + 1024] == (2 + 2) * 1024
    assert idx.pool_index.shape == (sum(p.tokens for p in plans), 4)
    assert idx.pool_index.max() == total + 2 and idx.pool_index.min() == 0
    # each image's rows read only its own patches
    first_rows = plans[0].tokens
    own = idx.pool_index[:first_rows]
    assert own[own < total].max() < second
    assert idx.pool_index[first_rows:][idx.pool_index[first_rows:] < total].min() >= second


@pytest.mark.skipif(SNAPSHOT is None, reason="the Phi-4-multimodal snapshot is not cached")
@pytest.mark.parametrize("size", [
    (464, 427), (1000, 300), (300, 1000), (3200, 1900), (2500, 300), (60, 45), (896, 448),
])
def test_intake_matches_the_reference_processor(size):
    pytest.importorskip("torchvision")
    from PIL import Image
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor = get_class_from_dynamic_module(
        "processing_phi4mm.Phi4MMImageProcessor", SNAPSHOT.rsplit("/", 1)[0],
    )(dynamic_hd=CONFIG.max_crops)
    rng = np.random.RandomState(sum(size))
    pixels = rng.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    want = processor.preprocess([Image.fromarray(pixels)], return_tensors="pt")
    got = image_intake(torch.from_numpy(pixels).permute(2, 0, 1), {}, CONFIG)
    down, across, rows, cols = got.grids[0]
    ref = want["input_image_embeds"][0]
    assert want["image_sizes"][0].tolist() == [448 * down, 448 * across]
    assert got.tokens == [int(want["num_img_tokens"][0])]
    assert torch.equal(got.inputs[GLOBAL_PIXEL_VALUES][0], ref[0])
    crops = got.inputs[PIXEL_VALUES].float().div(255).sub(0.5).div(0.5)
    assert torch.equal(crops, ref[1:1 + down * across])
    # the processor's patch mask is the grid's valid rectangle
    mask = want["image_attention_mask"][0, 1:1 + down * across]
    canvas = mask.view(down, across, 32, 32).permute(0, 2, 1, 3).reshape(32 * down, 32 * across)
    expect = torch.zeros_like(canvas)
    expect[:rows, :cols] = 1
    assert torch.equal(canvas, expect)
