"""V-JEPA 2 video preprocessing on synthetic frames (CPU).

The AC model takes the official repo's central square crop; the HF-checkpoint
model keeps HF's resize-and-crop, which sees less of a non-square frame.
"""

import pytest
import torch

from mstar.model.vjepa2.vjepa2_model import _IMAGENET_MEAN, _IMAGENET_STD, _preprocess_video

_MEAN = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)


def _frames(h: int, w: int, t: int = 2) -> torch.Tensor:
    """[T, 3, H, W] in [0, 1]: grey, with the central square's border rows and columns marked."""
    frames = torch.full((t, 3, h, w), 0.5)
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames[:, :, top, left : left + side] = 0.0
    frames[:, :, top + side - 1, left : left + side] = 1.0
    frames[:, :, top : top + side, left] = 0.25
    frames[:, :, top : top + side, left + side - 1] = 0.75
    return frames


def _pixels(out: torch.Tensor) -> torch.Tensor:
    return out * _STD + _MEAN


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (256, 256)])
def test_ac_crop_is_the_whole_central_square(hw):
    """Upscaling keeps the square's edges as the output's edges, so all four
    marked borders land on the output's border."""
    out = _pixels(_preprocess_video(_frames(*hw), crop_size=256, target_frames=2, central_square=True))
    assert out.shape == (2, 3, 256, 256)
    assert torch.allclose(out[..., 0, 2:-2], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(out[..., -1, 2:-2], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out[..., 2:-2, 0], torch.tensor(0.25), atol=1e-5)
    assert torch.allclose(out[..., 2:-2, -1], torch.tensor(0.75), atol=1e-5)


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (480, 640)])
def test_ac_crop_ignores_everything_outside_the_square(hw):
    h, w = hw
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames = _frames(h, w)
    noisy = torch.rand_like(frames, generator=torch.Generator().manual_seed(0))
    noisy[:, :, top : top + side, left : left + side] = frames[:, :, top : top + side, left : left + side]
    a = _preprocess_video(frames, crop_size=256, target_frames=2, central_square=True)
    b = _preprocess_video(noisy, crop_size=256, target_frames=2, central_square=True)
    assert torch.equal(a, b)


def test_ac_crop_of_a_square_frame_only_normalizes():
    frames = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    out = _preprocess_video(frames, crop_size=256, target_frames=2, central_square=True)
    assert torch.allclose(out, (frames - _MEAN) / _STD, atol=1e-6)


def test_hf_crop_is_unchanged_and_narrower():
    """The default path still crops 256 out of a 292-pixel short side, so the
    square's top and bottom rows fall outside it."""
    out = _pixels(_preprocess_video(_frames(180, 320), crop_size=256, target_frames=2))
    assert out.shape == (2, 3, 256, 256)
    assert out[..., 0, :].min() > 0.4
    assert out[..., -1, :].max() < 0.6
