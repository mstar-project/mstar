"""V-JEPA 2 video preprocessing on synthetic uint8 frames (CPU).

The HF checkpoints' frames go through their own video processor; the AC model
takes the official repo's central square crop, which sees more of a non-square
frame than the processor's resize and crop.
"""

import json

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import TensorPointerInfo
from mstar.model.vjepa2 import vjepa2_model
from mstar.model.vjepa2.submodules import VJepa2EncoderSubmodule
from mstar.model.vjepa2.vjepa2_model import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    VJepa2ACModel,
    VJepa2Model,
    _central_square,
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
    frames[:, :, top : top + side, left + side - 1] = 191
    return frames


def _clip(t: int, h: int, w: int) -> torch.Tensor:
    """uint8 [T, 3, H, W]: smooth gradients plus noise."""
    g = torch.Generator().manual_seed(h * w)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = torch.stack([yy, xx, (yy + xx) / 2]) * 200 + 20
    return (base + torch.randn(t, 3, h, w, generator=g) * 20).clamp(0, 255).round().to(torch.uint8)


def _pixels(out: torch.Tensor) -> torch.Tensor:
    """Normalized frames back in [0, 1]."""
    return out * _STD + _MEAN


def _checkpoint(path, crop_size: int | None = None) -> str:
    """A checkpoint directory with a tiny config.json and, given a crop size,
    the video processor's config. Constructing the model builds nothing."""
    (path / "config.json").write_text(json.dumps({"model_type": "vjepa2", "crop_size": 256}))
    if crop_size is not None:
        processing = pytest.importorskip("transformers.models.vjepa2.video_processing_vjepa2")
        processing.VJEPA2VideoProcessor(crop_size=crop_size).save_pretrained(path)
    return str(path)


@pytest.fixture
def hf_model(tmp_path):
    """The HF-checkpoint model, with its processor read through AutoVideoProcessor
    from the checkpoint directory at construction."""
    return VJepa2Model(_checkpoint(tmp_path, crop_size=256))


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (256, 256), (480, 640)])
def test_preprocess_matches_hf_processor(hf_model, hw, monkeypatch):
    """Same pixels as HF's processor on the whole clip, though process_prompt
    runs it on each decode group as it lands. Being within one uint8 step is
    not enough: the float resize this replaced was, and cost ~0.1 of embedding
    cosine."""
    from transformers.models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor

    frames = _clip(8, *hw)
    ref = VJEPA2VideoProcessor()(videos=frames, return_tensors="pt")["pixel_values_videos"][0]
    # channels last, as torchcodec decodes them, in uneven decode groups
    decoded = frames.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    monkeypatch.setattr(vjepa2_model, "decode_frames_at", lambda src, idx, transform: torch.cat(
        [transform(decoded[grp]) for grp in torch.tensor(idx).tensor_split([1, 4])]))
    out = hf_model.process_prompt(
        None, ["video"], ["embedding"], tensors={"video_inputs": [torch.zeros(1, dtype=torch.uint8)]},
        input_metadata={"video_inputs": [{"sampled_indices": list(range(8))}]},
    )
    ours = out.new_input_tensors["video_frames"][0]
    assert ours.shape == ref.shape == (8, 3, 256, 256)
    assert (ours - ref).abs().max().item() < 1e-4


def test_hf_crop_is_narrower(hf_model):
    """The processor crops 256 out of a 292-pixel short side, so the square's
    top and bottom rows fall outside it."""
    out = _pixels(hf_model._preprocess_frames(_frames(180, 320)))
    assert out.shape == (2, 3, 256, 256)
    assert out[..., 0, :].min() > 0.4
    assert out[..., -1, :].max() < 0.6


def test_a_checkpoint_without_a_video_processor_is_refused_at_startup(tmp_path):
    pytest.importorskip("transformers")
    with pytest.raises(OSError, match="video processor"):
        VJepa2Model(_checkpoint(tmp_path))


def test_a_processor_that_crops_another_size_is_refused_at_startup(tmp_path):
    with pytest.raises(ValueError, match="crops"):
        VJepa2Model(_checkpoint(tmp_path, crop_size=384))


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (256, 256)])
def test_ac_crop_is_the_whole_central_square(hw):
    """Upscaling keeps the square's edges as the output's edges, so all four
    marked borders land on the output's border."""
    out = _pixels(_central_square(_frames(*hw), crop_size=256))
    assert out.shape == (2, 3, 256, 256)
    assert torch.allclose(out[..., 0, 2:-2], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(out[..., -1, 2:-2], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out[..., 2:-2, 0], torch.tensor(64 / 255), atol=1e-5)
    assert torch.allclose(out[..., 2:-2, -1], torch.tensor(191 / 255), atol=1e-5)


@pytest.mark.parametrize("hw", [(180, 320), (320, 180), (480, 640)])
def test_ac_crop_ignores_everything_outside_the_square(hw):
    h, w = hw
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    frames = _frames(h, w)
    noisy = torch.randint(0, 256, frames.shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    noisy[:, :, top : top + side, left : left + side] = frames[:, :, top : top + side, left : left + side]
    assert torch.equal(_central_square(frames, crop_size=256), _central_square(noisy, crop_size=256))


def test_ac_crop_of_a_square_frame_only_normalizes():
    frames = _clip(2, 256, 256)
    out = _central_square(frames, crop_size=256)
    assert torch.allclose(out, (frames / 255 - _MEAN) / _STD, atol=1e-5)


def test_ac_crop_resizes_the_bytes_as_floats_like_upstream():
    """Upstream's make_transforms turns the whole clip into floats, channels
    first, and resizes those. One frame at a time from the bytes gives the same
    floats to the bit; resizing the bytes themselves would round them."""
    frames = _clip(4, 180, 320)
    clip = frames.permute(1, 0, 2, 3).to(torch.float32)[:, :, :, 70:250]
    resized = torch.nn.functional.interpolate(clip, size=(256, 256), mode="bilinear", align_corners=False)
    upstream = (resized.permute(1, 0, 2, 3) - _MEAN * 255.0) / (_STD * 255.0)
    assert torch.equal(_central_square(frames, crop_size=256), upstream)


def _pointer(tensor: torch.Tensor) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=list(tensor.shape), dtype=tensor.dtype, nbytes=tensor.nbytes, address=0,
        stride=list(tensor.stride()), uuid="u", source_session_id="", source_entity="api_server",
    )


@pytest.mark.parametrize("stream", [False, True])
def test_ac_rollout_request_replays_the_encoder_capture(stream, monkeypatch):
    """An AC rollout clip goes through the request's own trim and crop and lands
    on the encoder capture's key; a clip of any other shape runs eager without a
    word, so a change to either would quietly drop the capture."""
    # the frames load_video sampled, decoded by index as the request's decode does
    clip = torch.randint(0, 256, (16, 3, 180, 320), dtype=torch.uint8)
    monkeypatch.setattr(vjepa2_model, "decode_frames_at", lambda src, idx, transform: transform(clip[idx]))
    model = VJepa2ACModel("unused", skip_weight_loading=True)
    kwargs = {"actions": [[0.1] * 7] * 4, "states": [[0.2] * 7] * 4, "rollout_horizon": 4, "stream_rollout": stream}
    out = model.process_prompt(
        None, ["video"], ["embedding"], tensors={"video_inputs": [torch.zeros(1, dtype=torch.uint8)]},
        input_metadata={"video_inputs": [{"sampled_indices": list(range(16))}]}, **kwargs,
    ).new_input_tensors
    signals = {name: [_pointer(t) for t in tensors] for name, tensors in out.items()}
    args = model.get_initial_forward_pass_args("p0", ["video"], ["embedding"], signals, kwargs)
    walk = args.full_metadata.graph_walk
    info = CurrentForwardPassInfo(request_id="r", graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=0)
    info.step_metadata.update(args.step_metadata)

    encoder = VJepa2EncoderSubmodule(encoder=None, config=model.config)
    (capture,) = encoder.get_accelerator_graph_configs(torch.device("cpu"))
    assert walk in capture.replay_graph_walks
    assert encoder.cg_key_info(walk, {"r": info}) == capture.additional_key_info
