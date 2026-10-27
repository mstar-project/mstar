"""V-JEPA 2's video preprocessing against HF's VJEPA2VideoProcessor (CPU)."""

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
    _preprocess_video_droid,
)


def _clip(t: int, h: int, w: int) -> torch.Tensor:
    """uint8 [T, 3, H, W]: smooth gradients plus noise."""
    g = torch.Generator().manual_seed(h * w)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = torch.stack([yy, xx, (yy + xx) / 2]) * 200 + 20
    return (base + torch.randn(t, 3, h, w, generator=g) * 20).clamp(0, 255).round().to(torch.uint8)


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


@pytest.mark.parametrize("hw", [(180, 320), (320, 180)])
def test_droid_transform_sees_the_whole_centred_square(hw):
    """The AC model's transform (upstream's DROID one) reads the whole short side
    and nothing outside the centred square: grey inside it, white around it, and
    a black first row and column of the square that the upscale keeps."""
    h, w = hw
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames = torch.full((2, 3, h, w), 255, dtype=torch.uint8)
    frames[..., top:top + side, left:left + side] = 128
    frames[..., top, left:left + side] = 0
    frames[..., top:top + side, left] = 0
    out = _preprocess_video_droid(frames, crop_size=256, target_frames=2)
    assert out.shape == (2, 3, 256, 256)

    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    pixels = (out * std + mean) * 255.0
    assert pixels[..., 0, :].abs().max() < 1e-3
    assert pixels[..., :, 0].abs().max() < 1e-3
    assert (pixels[..., 2:, 2:] - 128).abs().max() < 1e-3


def test_droid_transform_of_a_square_frame_only_normalizes():
    frames = _clip(2, 256, 256)
    out = _preprocess_video_droid(frames, crop_size=256, target_frames=2)
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    assert (out - (frames / 255.0 - mean) / std).abs().max().item() < 1e-4
