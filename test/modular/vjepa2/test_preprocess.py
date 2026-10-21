"""V-JEPA 2's video preprocessing against HF's VJEPA2VideoProcessor (CPU)."""

import pytest
import torch

pytest.importorskip("torchvision")
pytest.importorskip("transformers")

from mstar.model.vjepa2.vjepa2_model import (  # noqa: E402
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _preprocess_video,
    _preprocess_video_droid,
)


def _clip(t: int, h: int, w: int) -> torch.Tensor:
    """uint8 [T, 3, H, W]: smooth gradients plus noise."""
    g = torch.Generator().manual_seed(h * w)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = torch.stack([yy, xx, (yy + xx) / 2]) * 200 + 20
    return (base + torch.randn(t, 3, h, w, generator=g) * 20).clamp(0, 255).round().to(torch.uint8)


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (256, 256), (480, 640)])
def test_preprocess_matches_hf_processor(hw):
    """Same pixels as HF's processor. Being within one uint8 step is not enough:
    the float resize this replaced was, and cost ~0.1 of embedding cosine."""
    from transformers.models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor

    frames = _clip(8, *hw)
    ref = VJEPA2VideoProcessor()(videos=frames, return_tensors="pt")["pixel_values_videos"][0]
    ours = _preprocess_video(frames.float() / 255.0, crop_size=256, target_frames=8)
    assert ours.shape == ref.shape == (8, 3, 256, 256)
    assert (ours - ref).abs().max().item() < 1e-4


@pytest.mark.parametrize("hw", [(180, 320), (320, 180)])
def test_droid_transform_sees_the_whole_centred_square(hw):
    """The AC model's transform (upstream's DROID one) reads the whole short side
    and nothing outside the centred square: grey inside it, white around it, and
    a black first row and column of the square that the upscale keeps."""
    h, w = hw
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames = torch.ones(2, 3, h, w)
    frames[..., top:top + side, left:left + side] = 0.5
    frames[..., top, left:left + side] = 0.0
    frames[..., top:top + side, left] = 0.0
    out = _preprocess_video_droid(frames, crop_size=256, target_frames=2)
    assert out.shape == (2, 3, 256, 256)

    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    pixels = out * std + mean
    assert (pixels[..., 0, :] - 0.0).abs().max() < 1e-4
    assert (pixels[..., :, 0] - 0.0).abs().max() < 1e-4
    assert (pixels[..., 2:, 2:] - 0.5).abs().max() < 1e-2


def test_droid_transform_of_a_square_frame_only_normalizes():
    frames = _clip(2, 256, 256).float() / 255.0
    out = _preprocess_video_droid(frames, crop_size=256, target_frames=2)
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    assert (out - (frames - mean) / std).abs().max().item() < 1e-4
