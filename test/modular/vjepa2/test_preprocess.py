"""V-JEPA 2 video preprocessing on synthetic frames (CPU).

The HF-checkpoint model runs the checkpoint's own video processor on the uint8
frames. The AC model takes the official repo's central square crop, which sees
more of a non-square frame.
"""

import sys
import types
from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("torchvision")
pytest.importorskip("transformers")

from mstar.model.vjepa2.vjepa2_model import (  # noqa: E402
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    VJepa2Model,
    _central_square_crop,
)

_MEAN = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)


def _frames(h: int, w: int, t: int = 2) -> torch.Tensor:
    """uint8 [T, 3, H, W]: grey, with the central square's border rows and columns marked."""
    frames = torch.full((t, 3, h, w), 128, dtype=torch.uint8)
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames[:, :, top, left : left + side] = 0
    frames[:, :, top + side - 1, left : left + side] = 255
    frames[:, :, top : top + side, left] = 64
    frames[:, :, top : top + side, left + side - 1] = 192
    return frames


def _clip(t: int, h: int, w: int) -> torch.Tensor:
    """uint8 [T, 3, H, W]: smooth gradients plus noise."""
    g = torch.Generator().manual_seed(h * w)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = torch.stack([yy, xx, (yy + xx) / 2]) * 200 + 20
    return (base + torch.randn(t, 3, h, w, generator=g) * 20).clamp(0, 255).round().to(torch.uint8)


def _pixels(out: torch.Tensor) -> torch.Tensor:
    """Normalized output back on the [0, 1] scale."""
    return out * _STD + _MEAN


def _video_frames(model: VJepa2Model, frames: torch.Tensor) -> torch.Tensor:
    out = model.process_prompt(None, ["video"], ["video"], tensors={"video_inputs": [frames]})
    return out.new_input_tensors["video_frames"][0]


@pytest.fixture
def model(tmp_path):
    """The masked model, its processor read back through AutoVideoProcessor.
    vjepa2-vitl-fpc64-256's processor settings are the class defaults."""
    from transformers.models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor

    VJEPA2VideoProcessor().save_pretrained(tmp_path)
    return VJepa2Model(str(tmp_path), skip_weight_loading=True)


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (256, 256), (480, 640)])
def test_preprocess_matches_hf_processor(hw, model):
    """Same pixels as HF's processor, bit for bit. Being within one uint8 step
    is not enough: the float resize this replaced was, and cost ~0.1 of
    embedding cosine."""
    from transformers.models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor

    frames = _clip(8, *hw)
    ours = _video_frames(model, frames)
    ref = VJEPA2VideoProcessor()(videos=frames, return_tensors="pt")["pixel_values_videos"][0]
    assert ours.shape == ref.shape == (8, 3, 256, 256)
    assert torch.equal(ours, ref)


def test_hf_crop_is_narrower(model):
    """HF's crop takes 256 out of a 292-pixel short side, so the square's top
    and bottom rows fall outside it."""
    out = _pixels(_video_frames(model, _frames(180, 320)))
    assert out.shape == (2, 3, 256, 256)
    assert out[..., 0, :].min() > 0.4
    assert out[..., -1, :].max() < 0.6


def test_warmup_loads_the_processor(model):
    model.warmup_preprocess()
    assert model._video_processor is not None
    ac = VJepa2Model("unused", skip_weight_loading=True, predictor_kind="ac")
    ac.warmup_preprocess()
    assert ac._video_processor is None


def test_preprocess_refuses_float_frames(model):
    with pytest.raises(ValueError, match="uint8"):
        _video_frames(model, _clip(2, 64, 64).float() / 255.0)


@dataclass
class _Metadata:
    num_frames: int = 4
    height: int = 36
    width: int = 64


def test_load_video_returns_the_decoded_uint8_frames(monkeypatch):
    decoded = _clip(4, 36, 64)

    class Decoder:
        metadata = _Metadata()

        def __init__(self, *args, **kwargs):
            pass

        def get_frames_at(self, indices):
            return types.SimpleNamespace(data=decoded[indices])

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.VideoDecoder = Decoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    model = VJepa2Model("unused", skip_weight_loading=True)
    assert torch.equal(model.load_video("clip.mp4", "cpu").data, decoded)


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (256, 256)])
def test_ac_crop_is_the_whole_central_square(hw):
    """Upscaling keeps the square's edges as the output's edges, so all four
    marked borders land on the output's border."""
    out = _pixels(_central_square_crop(_frames(*hw), crop_size=256, target_frames=2))
    assert out.shape == (2, 3, 256, 256)
    assert torch.allclose(out[..., 0, 2:-2], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(out[..., -1, 2:-2], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out[..., 2:-2, 0], torch.tensor(64 / 255), atol=1e-5)
    assert torch.allclose(out[..., 2:-2, -1], torch.tensor(192 / 255), atol=1e-5)


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (480, 640)])
def test_ac_crop_ignores_everything_outside_the_square(hw):
    h, w = hw
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames = _frames(h, w)
    noisy = torch.randint(0, 256, frames.shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    noisy[:, :, top : top + side, left : left + side] = frames[:, :, top : top + side, left : left + side]
    a = _central_square_crop(frames, crop_size=256, target_frames=2)
    b = _central_square_crop(noisy, crop_size=256, target_frames=2)
    assert torch.equal(a, b)


def test_ac_crop_of_a_square_frame_only_normalizes():
    frames = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    out = _central_square_crop(frames, crop_size=256, target_frames=2)
    assert torch.allclose(out, (frames / 255.0 - _MEAN) / _STD, atol=1e-5)
