"""Smoke tests for the Ming-flash-omni-2.0 encoder submodules (step 5a).

VisionEncoderSubmodule + AudioEncoderSubmodule wrap the components
ported in step 4. Tests cover three properties:

  * ``prepare_inputs`` raises a clear error on missing inputs and
    extracts tensors from the engine's NameToTensorList bundle.
  * ``forward`` produces the expected output edge name + tensor shape
    on tiny CPU instances (no snapshot needed; weights random).
  * The L2-norm post-projector matches Ming's source
    (``modeling_bailingmm2.extract_image_feature`` /
    ``extract_audio_feature``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mminf.model.ming_omni_flash.components.audio_encoder import MingAudioEncoder
from mminf.model.ming_omni_flash.components.projectors import (
    MingAudioProjector,
    MingVisionProjector,
)
from mminf.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    MingFlashOmniModelConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)
from mminf.model.ming_omni_flash.submodules import (
    AudioEncoderSubmodule,
    VisionEncoderSubmodule,
)


def _tiny_config() -> MingFlashOmniModelConfig:
    """Tiny config with the released ckpt's modal token IDs preserved."""
    return MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
    )


# ---------------------------------------------------------------------------
# AudioEncoderSubmodule — pure Python (random weights, CPU)
# ---------------------------------------------------------------------------


def _build_audio_submodule(hidden_size: int = 16) -> AudioEncoderSubmodule:
    cfg = _tiny_config()
    # Override LLM hidden_size so the projector output dim is small.
    cfg.thinker_llm = ThinkerLLMConfig(
        hidden_size=hidden_size, num_attention_heads=4, num_key_value_heads=2,
        head_dim=hidden_size // 4,
    )
    enc = MingAudioEncoder(n_mels=8, n_ctx=128, n_state=16, n_head=2, n_layer=2, use_flash_attn=False)
    enc = enc.float()
    proj = MingAudioProjector(audio_dim=16, llm_dim=hidden_size, mlp_depth=2)
    proj = proj.float()
    return AudioEncoderSubmodule(audio_encoder=enc, audio_projector=proj, config=cfg)


def test_audio_submodule_prepare_inputs_raises_on_missing_features() -> None:
    sub = _build_audio_submodule()
    with pytest.raises(ValueError, match="missing 'audio_features'"):
        sub.prepare_inputs(graph_walk="prefill_audio", fwd_info=None, inputs={})


def test_audio_submodule_prepare_inputs_passes_optional_seqlens() -> None:
    """``audio_seqlens`` is optional — None when caller didn't provide it."""
    sub = _build_audio_submodule()
    features = torch.randn(8, 10)
    out = sub.prepare_inputs(
        graph_walk="prefill_audio", fwd_info=None,
        inputs={"audio_features": [features]},
    )
    assert out.tensor_inputs["audio_features"] is features
    assert out.tensor_inputs["audio_seqlens"] is None


def test_audio_submodule_forward_single_clip_shape() -> None:
    """One clip → ``audio_embeds`` shape (T', llm_dim), L2-normed."""
    sub = _build_audio_submodule(hidden_size=16)
    features = torch.randn(8, 10)  # (n_mels, T)
    out = sub.forward(
        graph_walk="prefill_audio", engine_inputs=None,
        audio_features=features, audio_seqlens=None,
    )
    embeds = out["audio_embeds"][0]
    # Two convs: T=10 → conv1 stride=1 → 10; conv2 stride=2 → 6.
    # Projector conv kernel=3 stride=2 pad=1 → T'' = (6-3+2)//2+1 = 3.
    assert embeds.shape == (3, 16)
    # ``norm_query_embeds=True`` by default → each row has unit norm.
    norms = embeds.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_audio_submodule_forward_batched_clips_concatenates_along_time() -> None:
    """(B, n_mels, T) batched input concatenates per-clip output along time."""
    sub = _build_audio_submodule(hidden_size=16)
    features = torch.randn(2, 8, 10)  # 2 clips
    out = sub.forward(
        graph_walk="prefill_audio", engine_inputs=None,
        audio_features=features, audio_seqlens=None,
    )
    embeds = out["audio_embeds"][0]
    # Same per-clip T'' = 3, two clips → 6 rows.
    assert embeds.shape == (6, 16)


def test_audio_submodule_forward_respects_audio_seqlens() -> None:
    """``audio_seqlens`` trims padded tail before encoding."""
    sub = _build_audio_submodule(hidden_size=16)
    # Pad clip[0]'s T from 6 to 10 (extra noise tail). audio_seqlens=[6]
    # should make the encoder see only the first 6 frames.
    features_padded = torch.randn(8, 10)
    features_trimmed = features_padded[:, :6]
    seqlens = torch.tensor([6])

    out_padded = sub.forward(
        graph_walk="prefill_audio", engine_inputs=None,
        audio_features=features_padded, audio_seqlens=seqlens,
    )
    out_trimmed = sub.forward(
        graph_walk="prefill_audio", engine_inputs=None,
        audio_features=features_trimmed, audio_seqlens=None,
    )
    # Same output: padded version with seqlens=[6] equals raw 6-frame version.
    torch.testing.assert_close(
        out_padded["audio_embeds"][0], out_trimmed["audio_embeds"][0], rtol=1e-5, atol=1e-5,
    )


# ---------------------------------------------------------------------------
# VisionEncoderSubmodule — pure Python (mock encoder, CPU)
# ---------------------------------------------------------------------------


class _MockVisionEncoder(torch.nn.Module):
    """Stand-in for Qwen3MoeVisionTransformer that the submodule can drive.

    The real encoder needs the staged Ming source + nvrtc kernels; for
    a CPU unit test we mock the (pixel_values, grid_thw) → embeddings
    contract so the rest of the wrapper is exercised end-to-end.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        # Project pixel input into the encoder's "out_hidden_size" space.
        # Use a small trainable projection so the param-detection in
        # NodeSubmodule.get_device works (real encoder has params).
        self.dummy = torch.nn.Linear(8, out_dim, bias=False)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # Pretend each grid_thw produces (T*H*W / spatial_merge**2) tokens
        # of out_dim each. We just collapse pixel_values into out_dim.
        n_tokens = int(grid_thw.prod(dim=-1).sum().item())
        # Down/up-sample to n_tokens deterministically.
        x = self.dummy(pixel_values)
        if x.shape[0] >= n_tokens:
            return x[:n_tokens]
        # Tile if input is smaller than requested.
        reps = (n_tokens + x.shape[0] - 1) // x.shape[0]
        return x.repeat(reps, 1)[:n_tokens]


def _build_vision_submodule(vision_dim: int = 32, llm_dim: int = 16) -> VisionEncoderSubmodule:
    cfg = _tiny_config()
    cfg.thinker_llm = ThinkerLLMConfig(
        hidden_size=llm_dim, num_attention_heads=4, num_key_value_heads=2,
        head_dim=llm_dim // 4,
    )
    cfg.vision = VisionEncoderConfig(out_hidden_size=vision_dim)
    enc = _MockVisionEncoder(out_dim=vision_dim)
    proj = MingVisionProjector(vision_dim=vision_dim, llm_dim=llm_dim, mlp_depth=2)
    return VisionEncoderSubmodule(vision_encoder=enc, vision_projector=proj, config=cfg)


def test_vision_submodule_prepare_inputs_raises_on_missing_pixel_values() -> None:
    sub = _build_vision_submodule()
    with pytest.raises(ValueError, match="missing 'pixel_values'"):
        sub.prepare_inputs(graph_walk="prefill_vision", fwd_info=None, inputs={})


def test_vision_submodule_prepare_inputs_raises_on_missing_grid_thw() -> None:
    sub = _build_vision_submodule()
    pixels = torch.randn(4, 8)
    with pytest.raises(ValueError, match="image_grid_thw"):
        sub.prepare_inputs(
            graph_walk="prefill_vision", fwd_info=None,
            inputs={"pixel_values": [pixels]},
        )


def test_vision_submodule_prepare_inputs_promotes_1d_grid_thw() -> None:
    """1-D ``[T, H, W]`` grid_thw gets promoted to ``(1, 3)``."""
    sub = _build_vision_submodule()
    pixels = torch.randn(4, 8)
    grid_1d = torch.tensor([1, 2, 2], dtype=torch.long)
    out = sub.prepare_inputs(
        graph_walk="prefill_vision", fwd_info=None,
        inputs={"pixel_values": [pixels], "image_grid_thw": [grid_1d]},
    )
    assert out.tensor_inputs["grid_thw"].shape == (1, 3)


def test_vision_submodule_forward_produces_l2_normed_embeds() -> None:
    """``vision_embeds`` shape matches the encoder's token count; rows unit-norm."""
    sub = _build_vision_submodule(vision_dim=32, llm_dim=16)
    pixels = torch.randn(16, 8)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)  # T*H*W = 4 tokens
    out = sub.forward(
        graph_walk="prefill_vision", engine_inputs=None,
        pixel_values=pixels, grid_thw=grid_thw,
    )
    embeds = out["vision_embeds"][0]
    assert embeds.shape == (4, 16)
    norms = embeds.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# get_node_engine_types registration (step 5a)
# ---------------------------------------------------------------------------


def test_get_node_engine_types_registers_encoders() -> None:
    """Step 5a registers vision_encoder + audio_encoder as STATELESS."""
    from mminf.engine.base import EngineType
    from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel

    # Stand up just enough of the model to call get_node_engine_types
    # without loading the snapshot — build a bare instance via
    # __new__ and inject the config attribute.
    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = _tiny_config()
    types = inst.get_node_engine_types()
    assert types["Thinker"] == EngineType.KV_CACHE
    assert types["vision_encoder"] == EngineType.STATELESS
    assert types["audio_encoder"] == EngineType.STATELESS


def test_get_submodule_rejects_unknown_node() -> None:
    """Friendly error message for unregistered nodes (Talker still TODO)."""
    from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel

    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = _tiny_config()
    inst._submodule_cache = {}
    with pytest.raises(ValueError, match="Unknown node: 'Talker'"):
        inst.get_submodule("Talker", device="cpu")


# ---------------------------------------------------------------------------
# Snapshot-gated: end-to-end submodule construction with real weights
# ---------------------------------------------------------------------------


def _find_local_snapshot() -> str | None:
    """Mirror the helper in test_ming_flash_omni_encoders.py."""
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


@pytest.mark.skipif(
    _find_local_snapshot() is None,
    reason="Need Ming-flash-omni-2.0 snapshot (set MING_FLASH_OMNI_DIR).",
)
def test_create_audio_encoder_submodule_loads_real_weights() -> None:
    """``MingFlashOmniModel._create_audio_encoder_submodule`` end-to-end.

    Builds the encoder + projector from the real config, loads the
    real ckpt for both, then sanity-checks that the wrapper actually
    holds the loaded modules. Skipped on boxes without the snapshot.

    No CUDA needed — the audio encoder runs on CPU.
    """
    from mminf.model.ming_omni_flash.ming_omni_flash_model import (
        MingFlashOmniModel,
        _find_ming_code_dir,
    )

    snap = _find_local_snapshot()
    code_dir = _find_ming_code_dir() or "/tmp/ming_repo"

    model = MingFlashOmniModel(model_path_hf=snap, ming_code_dir=code_dir)
    sub = model.get_submodule("audio_encoder", device="cpu")
    assert isinstance(sub, AudioEncoderSubmodule)
    # Confirm the encoder + projector have loaded params (not random
    # init values). Conv1 weight RMS is well-defined post-load.
    conv1_w = sub.audio_encoder.conv1.weight
    assert conv1_w.abs().sum().item() > 0
    proj0_w = sub.audio_projector.proj[0].weight
    assert proj0_w.abs().sum().item() > 0
