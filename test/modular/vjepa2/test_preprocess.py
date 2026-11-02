"""V-JEPA 2 video preprocessing on synthetic frames (CPU).

The AC model takes the official repo's central square crop; the HF-checkpoint
model keeps HF's resize-and-crop, which sees less of a non-square frame.
"""

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import TensorPointerInfo
from mstar.model.vjepa2 import vjepa2_model
from mstar.model.vjepa2.submodules import VJepa2EncoderSubmodule
from mstar.model.vjepa2.vjepa2_model import _IMAGENET_MEAN, _IMAGENET_STD, VJepa2ACModel, _preprocess_video

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
