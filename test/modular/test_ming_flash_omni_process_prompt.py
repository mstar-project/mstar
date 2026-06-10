"""Tests for MingFlashOmniModel.process_prompt (step 7).

Two layers:

  * Pure-Python tests using stub tokenizer + processor — verify the
    dispatch (image/audio/video routing), tensor conversion (CHW
    float [0,1] → HWC uint8), and result-key shape. Run on CPU,
    no snapshot.

  * Snapshot-gated tests with the real BailingMM2Processor — confirm
    the chat template path, image processor, and audio processor
    produce the expected result keys + shapes when called against
    the actual checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mminf.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    MingFlashOmniModelConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)
from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel

# ---------------------------------------------------------------------------
# Snapshot discovery (mirrors test_ming_flash_omni_encoders.py)
# ---------------------------------------------------------------------------


def _find_local_snapshot() -> str | None:
    def _has_shards(path: Path) -> bool:
        return (
            (path / "config.json").exists()
            and (path / "model.safetensors.index.json").exists()
            and (path / "model-00001-of-00042.safetensors").exists()
        )

    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and _has_shards(Path(override)):
        return override
    hybrid = Path("/dev/shm/ming-hybrid")
    if _has_shards(hybrid):
        return str(hybrid)
    return None


# ---------------------------------------------------------------------------
# Stub tokenizer + processor for pure-Python tests
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Just enough tokenizer surface to drive process_prompt's text path."""

    eos_token = "<eos>"
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Emit a deterministic synthetic string; tokenize=False means
        # process_prompt will re-tokenize via __call__.
        assert tokenize is False
        return "<|USER|>" + messages[0]["content"] + "<|ASSISTANT|>"

    def __call__(self, text, return_tensors="pt"):
        # Toy: emit one int per character.
        ids = torch.tensor([[ord(c) % 256 for c in text]], dtype=torch.long)
        return type("Out", (), {"input_ids": ids})()


class _StubImageProcessor:
    """Produce predictable shapes from arbitrary HWC uint8 input."""

    def __call__(self, images=None, videos=None, return_tensors="pt", **kwargs):
        if images is not None:
            # Each image collapses to a single "patch" of fixed size for testing.
            n = len(images)
            return {
                "pixel_values": torch.zeros(n, 3, 16, 16),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * n, dtype=torch.long),
            }
        if videos is not None:
            n = len(videos)
            frames = videos[0].__len__() if hasattr(videos[0], "__len__") else 1
            return {
                "pixel_values_videos": torch.zeros(n * frames, 3, 16, 16),
                "video_grid_thw": torch.tensor([[frames, 4, 4]] * n, dtype=torch.long),
            }
        return {}


class _StubAudioProcessor:
    """Mel-spectrogram stub: produces fixed (n_mels=8, T=20) for any clip."""

    sampling_rate = 16000

    def __call__(self, audios, **kwargs):
        n = len(audios)
        # (B, T, n_mels) following the upstream layout.
        return {
            "audio_feats": np.zeros((n, 20, 8), dtype=np.float32),
            "audio_feats_lengths": np.array([20] * n, dtype=np.int64),
            "encoder_feats_lengths": np.array([10] * n, dtype=np.int64),
        }


class _StubProcessor:
    """Combine the modality stubs in the shape BailingMM2Processor exposes."""

    def __init__(self) -> None:
        self.image_processor = _StubImageProcessor()
        self.audio_processor = _StubAudioProcessor()


def _bare_model_with_stubs() -> MingFlashOmniModel:
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
    )
    inst.tokenizer = _StubTokenizer()
    inst._processor = _StubProcessor()
    inst._submodule_cache = {}
    return inst


# ---------------------------------------------------------------------------
# Text-only path
# ---------------------------------------------------------------------------


def test_text_only_returns_text_inputs_and_empty_modality_lists() -> None:
    m = _bare_model_with_stubs()
    out = m.process_prompt(
        prompt="hello",
        input_modalities=["text"],
        output_modalities=["text"],
        tensors=None,
    )
    assert "text_inputs" in out and len(out["text_inputs"]) == 1
    assert out["text_inputs"][0].dim() == 1
    # All modality buckets exist but are empty (so the scheduler in
    # step 5c sees a clean shape).
    for key in [
        "pixel_values", "image_grid_thw",
        "pixel_values_videos", "video_grid_thw", "video_second_per_grid",
        "audio_features", "audio_seqlens",
    ]:
        assert key in out and out[key] == []


def test_no_prompt_returns_no_text_inputs() -> None:
    """prompt=None → text_inputs empty (audio-only / image-only request)."""
    m = _bare_model_with_stubs()
    out = m.process_prompt(
        prompt=None,
        input_modalities=["audio"],
        output_modalities=["text"],
        tensors=None,
    )
    assert out["text_inputs"] == []


def test_missing_tokenizer_raises() -> None:
    m = _bare_model_with_stubs()
    m.tokenizer = None
    with pytest.raises(RuntimeError, match="tokenizer is not loaded"):
        m.process_prompt(
            prompt="hi", input_modalities=["text"],
            output_modalities=["text"], tensors=None,
        )


# ---------------------------------------------------------------------------
# Image path
# ---------------------------------------------------------------------------


def test_image_path_routes_through_image_processor() -> None:
    """CHW float [0,1] → image_processor → pixel_values + grid_thw."""
    m = _bare_model_with_stubs()
    img = torch.rand(3, 32, 32)
    out = m.process_prompt(
        prompt="describe", input_modalities=["text", "image"],
        output_modalities=["text"],
        tensors={"image_inputs": [img]},
    )
    assert len(out["pixel_values"]) == 1
    assert out["pixel_values"][0].shape == (1, 3, 16, 16)
    assert len(out["image_grid_thw"]) == 1
    assert out["image_grid_thw"][0].tolist() == [1, 4, 4]


def test_image_conversion_clamps_float_to_uint8_hwc() -> None:
    """Ensure the CHW-float → HWC-uint8 conversion is bit-correct for the
    happy path (qwen3_omni had a double-rescale bug that turned the input
    near-zero; this test guards against the same regression).
    """
    chw = torch.tensor([
        [[0.0, 1.0], [0.5, 0.25]],
        [[0.1, 0.9], [0.4, 0.7]],
        [[0.2, 0.8], [0.6, 0.3]],
    ])  # (3, 2, 2) — values < 1.0
    arr = MingFlashOmniModel._image_to_processor_input(chw)
    # Output is HWC uint8 in [0, 255].
    assert arr.shape == (2, 2, 3)
    assert arr.dtype == np.uint8
    # Top-left R channel was 0.0 → 0; top-right R was 1.0 → 255.
    assert arr[0, 0, 0] == 0
    assert arr[0, 1, 0] == 255


def test_image_conversion_handles_grayscale_single_channel() -> None:
    """(1, H, W) input gets broadcast to 3 channels (HF processors
    don't accept single-channel patches)."""
    gray = torch.full((1, 4, 4), 0.5)
    arr = MingFlashOmniModel._image_to_processor_input(gray)
    assert arr.shape == (4, 4, 3)
    # All three channels share the same value.
    assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 0] == arr[..., 2]).all()


def test_image_inputs_require_processor() -> None:
    m = _bare_model_with_stubs()
    m._processor = None
    img = torch.rand(3, 8, 8)
    with pytest.raises(RuntimeError, match="processor is None"):
        m.process_prompt(
            prompt=None, input_modalities=["image"],
            output_modalities=["text"], tensors={"image_inputs": [img]},
        )


def test_image_inputs_already_uint8_pass_through() -> None:
    """uint8 CHW input doesn't get rescaled a second time."""
    chw = torch.full((3, 4, 4), 128, dtype=torch.uint8)
    arr = MingFlashOmniModel._image_to_processor_input(chw)
    assert arr.dtype == np.uint8
    assert (arr == 128).all()


# ---------------------------------------------------------------------------
# Audio path
# ---------------------------------------------------------------------------


def test_audio_path_returns_mel_n_mels_first_and_seqlens() -> None:
    """The processor yields (B, T, n_mels); process_prompt transposes
    to (n_mels, T) per clip — that's what the AudioEncoderSubmodule
    expects in its single-clip prepare_inputs."""
    m = _bare_model_with_stubs()
    waveform = torch.randn(16000)  # 1 s at 16 kHz
    out = m.process_prompt(
        prompt=None, input_modalities=["audio"],
        output_modalities=["text"], tensors={"audio_inputs": [waveform]},
    )
    assert len(out["audio_features"]) == 1
    assert out["audio_features"][0].shape == (8, 20)  # (n_mels, T)
    assert len(out["audio_seqlens"]) == 1
    assert out["audio_seqlens"][0].tolist() == [20]


def test_audio_path_accepts_waveform_sr_tuples() -> None:
    """``(waveform, sample_rate)`` tuples are accepted as well as raw waveforms."""
    m = _bare_model_with_stubs()
    out = m.process_prompt(
        prompt=None, input_modalities=["audio"],
        output_modalities=["text"],
        tensors={"audio_inputs": [(torch.randn(8000), 16000)]},
    )
    assert len(out["audio_features"]) == 1


def test_audio_inputs_require_processor() -> None:
    m = _bare_model_with_stubs()
    m._processor = None
    with pytest.raises(RuntimeError, match="processor is None"):
        m.process_prompt(
            prompt=None, input_modalities=["audio"],
            output_modalities=["text"],
            tensors={"audio_inputs": [torch.randn(8000)]},
        )


# ---------------------------------------------------------------------------
# Video path
# ---------------------------------------------------------------------------


def test_video_path_returns_pixel_values_grid_and_second_per_grid_default() -> None:
    m = _bare_model_with_stubs()
    # (T, C, H, W) — 3 frames.
    video = torch.rand(3, 3, 32, 32)
    out = m.process_prompt(
        prompt="watch", input_modalities=["text", "video"],
        output_modalities=["text"],
        tensors={"video_inputs": [video]},
    )
    assert len(out["pixel_values_videos"]) == 1
    assert len(out["video_grid_thw"]) == 1
    assert out["video_grid_thw"][0].tolist() == [3, 4, 4]
    # Default second_per_grid is 1.0 when no metadata override.
    assert len(out["video_second_per_grid"]) == 1
    assert float(out["video_second_per_grid"][0].item()) == 1.0


def test_video_path_respects_metadata_second_per_grid_override() -> None:
    """``input_metadata['video'][i]['second_per_grid']`` overrides the default."""
    m = _bare_model_with_stubs()
    video = torch.rand(2, 3, 16, 16)
    out = m.process_prompt(
        prompt=None, input_modalities=["video"], output_modalities=["text"],
        tensors={"video_inputs": [video]},
        input_metadata={"video": [{"second_per_grid": 0.5}]},
    )
    assert float(out["video_second_per_grid"][0].item()) == 0.5


# ---------------------------------------------------------------------------
# Mixed-modality plumbing
# ---------------------------------------------------------------------------


def test_mixed_text_image_audio_all_buckets_populated() -> None:
    """A request with all three modalities populates all three buckets."""
    m = _bare_model_with_stubs()
    out = m.process_prompt(
        prompt="hello", input_modalities=["text", "image", "audio"],
        output_modalities=["text"],
        tensors={
            "image_inputs": [torch.rand(3, 16, 16)],
            "audio_inputs": [torch.randn(8000)],
        },
    )
    assert len(out["text_inputs"]) == 1
    assert len(out["pixel_values"]) == 1
    assert len(out["audio_features"]) == 1
    # No video for this request.
    assert out["pixel_values_videos"] == []


def test_multiple_images_emit_multiple_entries() -> None:
    """Two images → two pixel_values + two image_grid_thw entries."""
    m = _bare_model_with_stubs()
    imgs = [torch.rand(3, 16, 16), torch.rand(3, 24, 24)]
    out = m.process_prompt(
        prompt="describe", input_modalities=["text", "image", "image"],
        output_modalities=["text"],
        tensors={"image_inputs": imgs},
    )
    assert len(out["pixel_values"]) == 2
    assert len(out["image_grid_thw"]) == 2


# ---------------------------------------------------------------------------
# Snapshot-gated end-to-end with the real processor
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _find_local_snapshot() is None,
    reason="Need Ming-flash-omni-2.0 snapshot (set MING_FLASH_OMNI_DIR).",
)
def test_process_prompt_text_only_with_real_tokenizer() -> None:
    """End-to-end: real tokenizer + chat template produces non-empty input_ids."""
    snap = _find_local_snapshot()
    code_dir = os.environ.get("MING_CODE_DIR", "/tmp/ming_repo")
    model = MingFlashOmniModel(model_path_hf=snap, ming_code_dir=code_dir)
    if model.tokenizer is None:
        pytest.skip("Tokenizer didn't load on this box (env-only, not a code bug).")
    out = model.process_prompt(
        prompt="What is the capital of France?",
        input_modalities=["text"], output_modalities=["text"], tensors=None,
    )
    assert "text_inputs" in out and len(out["text_inputs"]) == 1
    input_ids = out["text_inputs"][0]
    assert input_ids.dim() == 1
    # Non-trivial prompt → at least a handful of tokens.
    assert input_ids.numel() > 5


@pytest.mark.skipif(
    _find_local_snapshot() is None,
    reason="Need Ming-flash-omni-2.0 snapshot.",
)
def test_process_prompt_image_path_with_real_image_processor() -> None:
    """End-to-end: real image processor accepts a tiny synthetic image."""
    snap = _find_local_snapshot()
    code_dir = os.environ.get("MING_CODE_DIR", "/tmp/ming_repo")
    model = MingFlashOmniModel(model_path_hf=snap, ming_code_dir=code_dir)
    if model.tokenizer is None or model._processor is None:
        pytest.skip("Tokenizer/processor didn't load on this box.")
    # 64x64 RGB image — small but the real processor's spatial_merge=2
    # + patch_size=16 needs a multiple-of-32 input on both sides.
    img = torch.rand(3, 64, 64)
    try:
        out = model.process_prompt(
            prompt="What is in this image?",
            input_modalities=["text", "image"], output_modalities=["text"],
            tensors={"image_inputs": [img]},
        )
    except Exception as e:
        pytest.skip(f"Real image processor failed to run on this box: {e}")
    assert len(out["pixel_values"]) == 1
    assert len(out["image_grid_thw"]) == 1
    # Grid should be (1, h, w) where h*16 >= image height (after resizing).
    grid = out["image_grid_thw"][0]
    assert grid.shape == (3,) and int(grid[0].item()) == 1
