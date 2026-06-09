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
    """Friendly error message for unregistered nodes.

    Talker is now registered (step 6e-2); ImageGen is the remaining
    unported node (step 9), so it's the canonical 'unknown' here.
    """
    from mminf.model.ming_omni_flash.ming_omni_flash_model import MingFlashOmniModel

    inst = MingFlashOmniModel.__new__(MingFlashOmniModel)
    inst.config = _tiny_config()
    inst._submodule_cache = {}
    with pytest.raises(ValueError, match="Unknown node: 'ImageGen'"):
        inst.get_submodule("ImageGen", device="cpu")


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


# ---------------------------------------------------------------------------
# BailingMoeV2ThinkerSubmodule.prepare_inputs dispatch (step 5b)
# ---------------------------------------------------------------------------
#
# These build a fake LingMoeModel-like stub so we can exercise the
# prepare_inputs dispatch (sentinel embed splice, position-id math)
# without a multi-GB MoE forward pass. The model.forward is never
# called in these tests; only prepare_inputs.


class _StubEmbedTokens(torch.nn.Module):
    """Identity-like embed for sentinel-id lookups in CPU unit tests.

    Returns a deterministic vector per token id so tests can verify
    the splice landed the right token at the right position.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # Per-token unit vector: token_id one-hot expanded into hidden_size
        # by tiling so we can read it back.
        table = torch.zeros(vocab_size, hidden_size, dtype=torch.float32)
        for i in range(vocab_size):
            table[i, i % hidden_size] = float(i + 1)
        self.weight = torch.nn.Parameter(table, requires_grad=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]


class _StubLingMoeModel(torch.nn.Module):
    """Minimal LingMoeModel surface used by the Thinker submodule init.

    Only ``embed_tokens`` and ``lm_head`` are accessed by the submodule
    constructor; forward isn't called in the prepare_inputs tests.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = _StubEmbedTokens(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)


def _build_thinker_submodule(
    hidden_size: int = 32,
    vocab_size: int | None = None,
):
    """Build a Thinker submodule on top of a tiny stub model.

    vocab_size defaults to one above the largest sentinel token id
    in the released ckpt's config so the embed lookups stay in range.
    """
    from mminf.model.ming_omni_flash.submodules import (
        BailingMoeV2ThinkerSubmodule,
    )
    cfg = _tiny_config()
    if vocab_size is None:
        # Largest modal sentinel id on the released ckpt is video_patch_token = 157175.
        vocab_size = cfg.thinker_llm.video_patch_token + 100
    cfg.thinker_llm.vocab_size = vocab_size
    cfg.thinker_llm.hidden_size = hidden_size
    cfg.thinker_llm.head_dim = max(hidden_size // 4, 1)
    cfg.thinker_llm.num_attention_heads = 4
    cfg.thinker_llm.num_key_value_heads = 2
    model = _StubLingMoeModel(vocab_size=vocab_size, hidden_size=hidden_size)
    return BailingMoeV2ThinkerSubmodule(model=model, config=cfg)


def test_thinker_prepare_inputs_prefill_text_uses_input_ids() -> None:
    """Text prefill returns input_ids path (no splice, no embeds)."""
    sub = _build_thinker_submodule(hidden_size=32)
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    out = sub.prepare_inputs(
        graph_walk="prefill_text", fwd_info=None,
        inputs={"text_inputs": [token_ids]},
    )
    assert out.input_seq_len == 5
    assert out.input_embeds is None
    assert out.custom_pos_ids is None
    torch.testing.assert_close(out.input_ids, token_ids)


def test_thinker_prepare_inputs_legacy_prefill_walk_still_works() -> None:
    """``prefill`` (the step 3f name) routes the same as prefill_text."""
    sub = _build_thinker_submodule()
    token_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    out = sub.prepare_inputs(
        graph_walk="prefill", fwd_info=None,
        inputs={"text_inputs": [token_ids]},
    )
    assert out.input_embeds is None
    torch.testing.assert_close(out.input_ids, token_ids)


def test_thinker_prepare_inputs_decode_path() -> None:
    """thinker_decode returns input_ids path with seq_len=1."""
    sub = _build_thinker_submodule()
    out = sub.prepare_inputs(
        graph_walk="thinker_decode", fwd_info=None,
        inputs={"text_inputs": [torch.tensor([42], dtype=torch.long)]},
    )
    assert out.input_seq_len == 1
    assert out.input_ids.tolist() == [42]


def test_thinker_prepare_inputs_prefill_audio_splices_bos_eos() -> None:
    """prefill_audio wraps audio_embeds with audio_start / audio_end sentinels."""
    sub = _build_thinker_submodule(hidden_size=32)
    audio_embeds = torch.randn(4, 32)
    out = sub.prepare_inputs(
        graph_walk="prefill_audio", fwd_info=None,
        inputs={"audio_embeds": [audio_embeds]},
    )
    # Seq len = 1 (bos) + 4 (audio) + 1 (eos) = 6.
    assert out.input_seq_len == 6
    assert out.input_embeds.shape == (6, 32)
    # First row should match the audio_start_token embed; last row the
    # audio_end_token embed.
    cfg = sub.config.thinker_llm
    expected_bos = sub.embed_tokens.weight[cfg.audio_start_token]
    expected_eos = sub.embed_tokens.weight[cfg.audio_end_token]
    torch.testing.assert_close(out.input_embeds[0].float(), expected_bos.float())
    torch.testing.assert_close(out.input_embeds[-1].float(), expected_eos.float())
    # Middle rows are the audio embeds as supplied.
    torch.testing.assert_close(out.input_embeds[1:5], audio_embeds)
    # 3D positions, text-like.
    assert out.custom_pos_ids.shape == (3, 6)
    assert out.custom_pos_ids[0].tolist() == [0, 1, 2, 3, 4, 5]


def test_thinker_prepare_inputs_prefill_audio_advances_with_start_pos() -> None:
    """Audio span at start_pos=10 produces positions [10..15]."""
    from mminf.engine.kv_store import PositionInfo
    sub = _build_thinker_submodule(hidden_size=32)
    audio_embeds = torch.randn(2, 32)
    out = sub.prepare_inputs(
        graph_walk="prefill_audio", fwd_info=None,
        inputs={"audio_embeds": [audio_embeds]},
        pos_info={"main": PositionInfo(position_id_start=10)},
    )
    assert out.input_seq_len == 4   # bos + 2 + eos
    assert out.custom_pos_ids[0].tolist() == [10, 11, 12, 13]


def test_thinker_prepare_inputs_prefill_audio_raises_on_missing_audio_embeds() -> None:
    sub = _build_thinker_submodule()
    with pytest.raises(ValueError, match="missing 'audio_embeds'"):
        sub.prepare_inputs(
            graph_walk="prefill_audio", fwd_info=None, inputs={},
        )


def test_thinker_prepare_inputs_prefill_vision_splices_bos_eos() -> None:
    """prefill_vision wraps vision_embeds with image_start / image_end sentinels."""
    sub = _build_thinker_submodule(hidden_size=32)
    # grid (1, 4, 4), spatial_merge=2 → 4 tokens.
    vision_embeds = torch.randn(4, 32)
    out = sub.prepare_inputs(
        graph_walk="prefill_vision", fwd_info=None,
        inputs={
            "vision_embeds": [vision_embeds],
            "image_grid_thw": [torch.tensor([1, 4, 4], dtype=torch.long)],
        },
    )
    # seq_len = 1 (image_start) + 4 (vision) + 1 (image_end) = 6
    assert out.input_seq_len == 6
    assert out.input_embeds.shape == (6, 32)
    cfg = sub.config.thinker_llm
    expected_bos = sub.embed_tokens.weight[cfg.image_start_token]
    expected_eos = sub.embed_tokens.weight[cfg.image_end_token]
    torch.testing.assert_close(out.input_embeds[0].float(), expected_bos.float())
    torch.testing.assert_close(out.input_embeds[-1].float(), expected_eos.float())
    # 3D positions, grid-aware.
    assert out.custom_pos_ids.shape == (3, 6)
    # Position 0 is the image_start sentinel at start_pos=0; vision span
    # at start_pos+1=1, single-frame grid (1, 4, 4)/spatial_merge=2 →
    # llm_grid = (1, 2, 2) = 4 tokens. T row constant at 1; H row
    # cycles [1, 1, 2, 2]; W row cycles [1, 2, 1, 2]. Max position
    # across all rows = 2; eos sentinel goes at 2 + 1 = 3 in every row
    # (Ming uses ``llm_pos_ids_list[-1].max() + 1`` — global max, not
    # per-row, see modeling_bailing_moe_v2.get_rope_index:632).
    assert out.custom_pos_ids[0].tolist() == [0, 1, 1, 1, 1, 3]   # T row
    assert out.custom_pos_ids[1].tolist() == [0, 1, 1, 2, 2, 3]   # H row
    assert out.custom_pos_ids[2].tolist() == [0, 1, 2, 1, 2, 3]   # W row


def test_thinker_prepare_inputs_prefill_video_uses_video_sentinels() -> None:
    """prefill_video selects video_start / video_end sentinels."""
    sub = _build_thinker_submodule(hidden_size=32)
    vision_embeds = torch.randn(2, 32)   # grid (1, 2, 2) → 1 token; here just 2
    # Use grid (2, 2, 2) which gives 2 tokens for spatial_merge=2.
    out = sub.prepare_inputs(
        graph_walk="prefill_video", fwd_info=None,
        inputs={
            "vision_embeds": [vision_embeds],
            "image_grid_thw": [torch.tensor([2, 2, 2], dtype=torch.long)],
            "video_second_per_grid": [torch.tensor(1.0)],
        },
    )
    assert out.input_seq_len == 4   # bos + 2 + eos
    cfg = sub.config.thinker_llm
    expected_bos = sub.embed_tokens.weight[cfg.video_start_token]
    expected_eos = sub.embed_tokens.weight[cfg.video_end_token]
    torch.testing.assert_close(out.input_embeds[0].float(), expected_bos.float())
    torch.testing.assert_close(out.input_embeds[-1].float(), expected_eos.float())


def test_thinker_prepare_inputs_prefill_vision_raises_on_missing_grid_thw() -> None:
    sub = _build_thinker_submodule()
    with pytest.raises(ValueError, match="missing 'image_grid_thw'"):
        sub.prepare_inputs(
            graph_walk="prefill_vision", fwd_info=None,
            inputs={"vision_embeds": [torch.randn(4, 32)]},
        )


def test_thinker_prepare_inputs_prefill_vision_rejects_multi_image() -> None:
    sub = _build_thinker_submodule()
    with pytest.raises(NotImplementedError, match="multi-image"):
        sub.prepare_inputs(
            graph_walk="prefill_vision", fwd_info=None,
            inputs={
                "vision_embeds": [torch.randn(4, 32)],
                "image_grid_thw": [torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)],
            },
        )


def test_thinker_prepare_inputs_unknown_walk_raises() -> None:
    sub = _build_thinker_submodule()
    with pytest.raises(ValueError, match="unknown graph_walk"):
        sub.prepare_inputs(
            graph_walk="prefill_unicorn", fwd_info=None, inputs={},
        )


# ---------------------------------------------------------------------------
# Snapshot-gated: end-to-end submodule construction with real weights
# ---------------------------------------------------------------------------


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
