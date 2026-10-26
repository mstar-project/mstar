"""Image-to-video conditioning recipes (components/conditioning.py).

``aspect_crop`` is compared on 8-bit frames with NVIDIA's recipe written with
torchvision (cosmos-framework's ``_resize_and_center_crop``: ``TF.resize``,
``TF.center_crop``, round) and with diffusers' ``_preprocess_conditioning_image``
when the installed diffusers has it (0.40 and later).
"""

from __future__ import annotations

import math

import pytest
import torch

from mstar.model.cosmos3.components.conditioning import prepare_conditioning_frames
from mstar.model.cosmos3.config import Cosmos3Config

TARGET = (480, 832)  # (height, width)
SOURCES = [  # (height, width)
    (180, 320),  # 16:9, upscaled, width cropped
    (640, 360),  # portrait, upscaled, height cropped
    (1920, 1080),  # portrait, downscaled
    TARGET,  # already the target size
]


def _as_uint8(x: torch.Tensor) -> torch.Tensor:
    """Frames in [-1, 1] back to the 8-bit values they encode."""
    return ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def _source(size: tuple[int, int]) -> torch.Tensor:
    g = torch.Generator().manual_seed(size[0] * 10000 + size[1])
    return torch.randint(0, 256, (3, *size), generator=g, dtype=torch.uint8)


def _nvidia_crop(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """cosmos-framework's ``load_conditioning_image_pixels`` on a uint8 ``[3, H, W]``."""
    import torchvision.transforms.functional as TF

    x = img.float().unsqueeze(0)
    scale = max(width / x.shape[3], height / x.shape[2])
    size = [math.ceil(scale * x.shape[2]), math.ceil(scale * x.shape[3])]
    x = TF.resize(x, size, interpolation=TF.InterpolationMode.BILINEAR)
    x = TF.center_crop(x, [height, width])
    return x.squeeze(0).round().clamp(0, 255).to(torch.uint8)


@pytest.mark.parametrize("size", SOURCES)
def test_aspect_crop_matches_cosmos_framework(size) -> None:
    img = _source(size)
    # load_image hands the node [0, 1] floats; they must give what the 8-bit image gives.
    out = prepare_conditioning_frames(img.float() / 255.0, *TARGET)
    assert out.shape == (1, 3, 1, *TARGET)
    assert torch.equal(out, prepare_conditioning_frames(img, *TARGET))
    assert torch.equal(_as_uint8(out)[0, :, 0], _nvidia_crop(img, *TARGET))


@pytest.mark.parametrize("size", SOURCES)
def test_aspect_crop_matches_diffusers(size) -> None:
    pipeline = pytest.importorskip("diffusers.pipelines.cosmos.pipeline_cosmos3_omni")
    if not hasattr(pipeline, "_preprocess_conditioning_image"):
        pytest.skip("diffusers before 0.40 stretches the conditioning image")
    img = _source(size)
    out = prepare_conditioning_frames(img.float() / 255.0, *TARGET)
    assert torch.equal(out[:, :, 0], pipeline._preprocess_conditioning_image(img, *TARGET))


def test_aspect_crop_by_hand() -> None:
    # A 3x1 row into 4x2: the height sets the scale (2, not 4/3), the 2x
    # upscale blends neighbours 3/4 + 1/4, and the center crop keeps columns
    # 1-4 of 6. 135.25 and 185.75 round to 135 and 186.
    row = torch.tensor([10, 110, 211], dtype=torch.uint8).expand(3, 1, 3)
    out = _as_uint8(prepare_conditioning_frames(row, 2, 4))[0, :, 0]
    assert torch.equal(out, torch.tensor([35, 85, 135, 186], dtype=torch.uint8).expand(3, 2, 4))
    # A crop margin of 5 columns splits 2 + 3: Python's round goes half to
    # even, as in torchvision's center_crop and diffusers.
    cols = (torch.arange(7, dtype=torch.uint8) * 10).expand(3, 2, 7)
    out = _as_uint8(prepare_conditioning_frames(cols, 2, 2))[0, :, 0]
    assert torch.equal(out, torch.tensor([20, 30], dtype=torch.uint8).expand(3, 2, 2))


def test_stretch_is_antialiased_bilinear() -> None:
    # Four columns into two: each output averages three source columns with
    # weights (3, 3, 1) / 7 and (1, 3, 3) / 7. Nearest-neighbour would give
    # (0, 140), bilinear without antialiasing (35, 175).
    row = torch.tensor([0, 70, 140, 210], dtype=torch.uint8).expand(3, 1, 4)
    out = _as_uint8(prepare_conditioning_frames(row, 1, 2, "stretch"))[0, :, 0]
    assert torch.equal(out, torch.tensor([50, 160], dtype=torch.uint8).expand(3, 1, 2))


def test_frame_stacks_and_modes() -> None:
    assert Cosmos3Config().conditioning_resize == "aspect_crop"
    clip = torch.randint(0, 256, (4, 3, 45, 80), generator=torch.Generator().manual_seed(0), dtype=torch.uint8)
    for mode in ("aspect_crop", "stretch"):
        out = prepare_conditioning_frames(clip, 64, 64, mode)
        assert out.shape == (1, 3, 4, 64, 64)
        for t in range(4):
            assert torch.equal(out[:, :, t], prepare_conditioning_frames(clip[t], 64, 64, mode)[:, :, 0])
    with pytest.raises(ValueError, match="conditioning_resize"):
        prepare_conditioning_frames(clip, 64, 64, "pad")
