"""Tests for the AudioVAE port (step 6d).

Covers the building blocks in ``components/audio_vae.py``:

  * ISTFT round-trip (center + same padding modes).
  * StreamingLinearUpsample chunked-vs-single-shot equivalence.
  * ISTFTHead, Encoder, Decoder shape contracts.
  * AudioVAE construction from real config + encode/decode round-trip
    on a tiny synthetic config (CPU, no snapshot).
  * Snapshot-gated structural assertions against the real
    ``talker/vae/model.safetensors`` (no weight load — that's 6f).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mminf.model.ming_omni_flash.components.audio_vae import (
    AudioVAE,
    build_audio_vae,
)
from mminf.model.ming_omni_flash.components.audio_vae import (
    _Decoder,
    _Encoder,
    _ISTFT,
    _ISTFTHead,
    _StreamingLinearUpsample,
    _oobleck_sample,
)
from mminf.model.ming_omni_flash.config import AudioVAEConfig


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------


def _find_local_snapshot() -> str | None:
    def _has(p: Path) -> bool:
        return (
            (p / "talker" / "vae" / "config.json").exists()
            and (p / "talker" / "vae" / "model.safetensors").exists()
        )
    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and _has(Path(override)):
        return override
    hybrid = Path("/dev/shm/ming-hybrid")
    if _has(hybrid):
        return str(hybrid)
    return None


# ---------------------------------------------------------------------------
# Tiny Qwen2 backbone dict (keeps tests fast; matches released layout)
# ---------------------------------------------------------------------------


def _tiny_qwen2_backbone(hidden_size: int = 32, num_layers: int = 1) -> dict:
    return {
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "vocab_size": 1,
        "use_sliding_window": True,
        "sliding_window": 32,
        "max_window_layers": 0,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    }


# ---------------------------------------------------------------------------
# Oobleck sampler
# ---------------------------------------------------------------------------


def test_oobleck_sample_split_mean_scale() -> None:
    """`_oobleck_sample` chunks parameters along dim=1; scale is softplus+eps.

    Verify the chunk split is on the right axis and shape collapses
    from (B, 2*L, T) to (B, L, T).
    """
    params = torch.zeros(2, 8, 5)   # 2*latent_dim=8 → latent_dim=4
    out = _oobleck_sample(params)
    assert out.shape == (2, 4, 5)


def test_oobleck_sample_returns_mean_when_scale_is_very_negative() -> None:
    """With scale_raw = -inf-ish, softplus → 0, so sample → mean."""
    B, L, T = 1, 2, 3
    mean = torch.full((B, L, T), 7.0)
    scale_raw = torch.full((B, L, T), -1000.0)
    params = torch.cat([mean, scale_raw], dim=1)
    out = _oobleck_sample(params)
    # softplus(-1000) + 1e-4 ≈ 1e-4 → sample ≈ mean within 1e-3 tolerance.
    torch.testing.assert_close(out, mean, atol=1e-2, rtol=0)


# ---------------------------------------------------------------------------
# ISTFT
# ---------------------------------------------------------------------------


def test_istft_rejects_invalid_padding() -> None:
    with pytest.raises(ValueError, match="Padding must be"):
        _ISTFT(n_fft=8, hop_length=2, win_length=8, padding="left")


def test_istft_center_mode_uses_torch_istft() -> None:
    """`center` mode is a thin torch.istft wrapper; check it runs end-to-end."""
    n_fft, hop, win = 16, 4, 16
    istft = _ISTFT(n_fft, hop, win, padding="center")
    # 4 frames → original waveform length T = (4-1)*hop + win (center=True
    # internally trims by win/2 each side, but the wrapper passes
    # center=True so the upstream choice stands).
    spec = torch.complex(torch.randn(1, n_fft // 2 + 1, 4), torch.randn(1, n_fft // 2 + 1, 4))
    y, ab, wb = istft(spec)
    assert y.dim() == 2
    assert torch.isfinite(y).all()
    assert ab is None and wb is None


def test_istft_same_mode_runs_non_streaming() -> None:
    """`same` mode path is the streaming-able variant; non-streaming usage trims `pad` from both ends."""
    n_fft, hop, win = 8, 2, 8
    istft = _ISTFT(n_fft, hop, win, padding="same")
    # Choose enough frames that `output_size - 2*pad` is positive.
    spec = torch.complex(torch.randn(1, n_fft // 2 + 1, 8), torch.randn(1, n_fft // 2 + 1, 8))
    y, ab, wb = istft(spec, streaming=False)
    assert y.dim() == 2
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# StreamingLinearUpsample
# ---------------------------------------------------------------------------


def test_streaming_upsample_single_shot_path_returns_upscaled() -> None:
    """``is_first=True, is_last=True`` → straight upsample, no state."""
    up = _StreamingLinearUpsample(scale_factor=4)
    x = torch.randn(1, 3, 5)
    out, state = up(x, state=None, is_last=True)
    assert state is None
    assert out.shape == (1, 12, 5)  # 3 * 4 = 12


def test_streaming_upsample_first_non_last_defers() -> None:
    """First chunk with more to come → return None, populate prev_chunk."""
    up = _StreamingLinearUpsample(scale_factor=4)
    x = torch.randn(1, 2, 5)
    out, state = up(x, state=None, is_last=False)
    assert out is None
    assert state["prev_chunk"] is x
    assert state["is_first"] is False


def test_streaming_upsample_two_chunk_equivalent_to_single_shot() -> None:
    """Concatenating two chunked outputs matches a single-shot upsample.

    This is the key correctness property: chunked streaming must not
    introduce boundary artefacts (the upsampler's left/right lookahead
    + history_last bookkeeping is exactly what makes this hold).
    """
    up = _StreamingLinearUpsample(scale_factor=4)
    a = torch.randn(1, 3, 5)
    b = torch.randn(1, 4, 5)

    # Chunked path: first(a), then last(b).
    out_a, state = up(a, state=None, is_last=False)
    assert out_a is None
    out_b, state = up(b, state=state, is_last=True)
    assert state is None
    chunked = out_b

    # Single-shot path: concat(a, b) → one upsample.
    full = torch.cat([a, b], dim=1)
    single, _ = up(full, state=None, is_last=True)

    torch.testing.assert_close(chunked, single, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# ISTFTHead
# ---------------------------------------------------------------------------


def test_istft_head_output_shape() -> None:
    """ISTFTHead returns (audio: (B, 1, T), x_pred: (B, n_fft+2, T_frames), bufs...)."""
    head = _ISTFTHead(dim=16, n_fft=16, hop_length=4)
    x = torch.randn(1, 8, 16)
    audio, x_pred, ab, wb = head(x)
    assert audio.dim() == 3
    assert audio.shape[0] == 1 and audio.shape[1] == 1
    assert x_pred.shape == (1, 16 + 2, 8)


# ---------------------------------------------------------------------------
# Encoder / Decoder shape contracts
# ---------------------------------------------------------------------------


def test_encoder_get_frames_pads_right_edge() -> None:
    """get_frames windows the waveform with input_dim/hop_size stride."""
    enc = _Encoder(
        encoder_args=_tiny_qwen2_backbone(hidden_size=32, num_layers=1),
        input_dim=16, hop_size=16, latent_dim=4, patch_size=-1,
        attn_implementation="sdpa",
    )
    enc = enc.float()
    # 50-sample waveform with input_dim=16, hop=16 → ceil(50/16) = 4 frames.
    # Formula: (50 + 16 - 1) // 16 = 65 // 16 = 4.  After padding to
    # (4-1)*16 + 16 = 64 samples, unfold(size=16, step=16) yields 4 windows.
    waveform = torch.randn(1, 50)
    frames = enc.get_frames(waveform)
    assert frames.shape[0] == 1
    assert frames.shape[1] == 4      # frames
    assert frames.shape[2] == 16     # input_dim


def test_encoder_forward_emits_latent_params_no_patching() -> None:
    """patch_size=-1 → skip aggregator path; output `(B, T, 2*latent_dim)`."""
    enc = _Encoder(
        encoder_args=_tiny_qwen2_backbone(hidden_size=32),
        input_dim=16, hop_size=16, latent_dim=4, patch_size=-1,
        attn_implementation="sdpa",
    )
    enc = enc.float().eval()
    waveform = torch.randn(1, 64)
    with torch.no_grad():
        params, y = enc(waveform)
    # T_frames = ceil((64 + 15) / 16) = 4. 2*latent_dim = 8.
    assert params.shape == (1, 4, 8)
    assert y.shape == (1, 1, 64)


def test_encoder_forward_with_patching_emits_per_patch_latents() -> None:
    """patch_size > 0 → aggregator output keeps the [CLS] row per patch."""
    enc = _Encoder(
        encoder_args=_tiny_qwen2_backbone(hidden_size=32),
        input_dim=16, hop_size=16, latent_dim=4, patch_size=2,
        attn_implementation="sdpa",
    )
    enc = enc.float().eval()
    waveform = torch.randn(1, 64)
    with torch.no_grad():
        params, _ = enc(waveform)
    # T_frames=4, patch_size=2 → 2 patches → 2 latent rows.
    assert params.shape == (1, 2, 8)


def test_decoder_low_level_reconstruct_non_streaming_shape() -> None:
    """Non-streaming decode produces a waveform tensor of the right rank."""
    dec = _Decoder(
        decoder_args=_tiny_qwen2_backbone(hidden_size=32),
        output_dim=16, latent_dim=4, patch_size=-1,
        attn_implementation="sdpa",
    )
    dec = dec.float().eval()
    # latent_dim=4, T_frames=3 → after upsampler... patch_size=-1 so no
    # upsampler. fc1 maps to hidden_size=32, then Qwen2 backbone, then
    # ISTFTHead emits 1 audio channel.
    latent = torch.randn(1, 3, 4)
    with torch.no_grad():
        out, state, pkv = dec.low_level_reconstruct(latent, use_cache=False)
    assert out.dim() == 3
    assert out.shape[0] == 1 and out.shape[1] == 1
    assert torch.isfinite(out).all()
    assert state == (None, None, None)


def test_decoder_with_patching_upsamples_before_backbone() -> None:
    """patch_size != -1 enables the streaming upsampler before the Qwen2 backbone."""
    dec = _Decoder(
        decoder_args=_tiny_qwen2_backbone(hidden_size=32),
        output_dim=16, latent_dim=4, patch_size=4,
        attn_implementation="sdpa",
    )
    dec = dec.float().eval()
    latent = torch.randn(1, 2, 4)
    with torch.no_grad():
        out, _, _ = dec.low_level_reconstruct(latent, use_cache=False)
    # Output waveform exists and is finite.
    assert out.dim() == 3
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# AudioVAE wrapper construction + end-to-end on tiny config
# ---------------------------------------------------------------------------


def _tiny_audio_vae_config() -> AudioVAEConfig:
    backbone = _tiny_qwen2_backbone(hidden_size=32, num_layers=1)
    return AudioVAEConfig(
        sample_rate=8000,
        patch_size=-1,                # disable patching to keep tests fast
        latent_dim=4,
        encoder_input_dim=16,
        encoder_hop_size=16,
        decoder_output_dim=16,
        enc_backbone=dict(backbone),
        dec_backbone=dict(backbone),
    )


def test_build_audio_vae_constructs_encoder_and_decoder() -> None:
    cfg = _tiny_audio_vae_config()
    vae = build_audio_vae(cfg, dtype=torch.float32, device="cpu")
    assert isinstance(vae, AudioVAE)
    assert vae.sample_rate == 8000
    assert vae.encoder.input_dim == 16
    assert vae.decoder.hop_length == 16
    # patch_size=-1 → no aggregator / upsampler.
    assert not hasattr(vae.encoder, "aggregator") or vae.encoder.patch_size != -1
    assert not hasattr(vae.decoder, "upsampling")


def test_audio_vae_encode_latent_returns_correct_shape() -> None:
    cfg = _tiny_audio_vae_config()
    vae = build_audio_vae(cfg, dtype=torch.float32, device="cpu")
    waveform = torch.randn(2, 64)
    waveform_length = torch.tensor([64, 48])
    with torch.no_grad():
        latent, frame_num = vae.encode_latent(waveform, waveform_length)
    # input_dim=16 → frame_num[0] = ceil(64/16) = 4, frame_num[1] = ceil(48/16) = 3.
    assert frame_num.tolist() == [4, 3]
    # Latent dimensions: (B, T_latents, latent_dim) after transpose.
    # T_latents = encoder T_frames = ceil((64 + 15) / 16) = 4 (same for both since waveform was padded to max len before encoder)
    assert latent.shape[0] == 2
    assert latent.shape[2] == 4   # latent_dim
    assert torch.isfinite(latent).all()


def test_audio_vae_decode_runs_end_to_end() -> None:
    cfg = _tiny_audio_vae_config()
    vae = build_audio_vae(cfg, dtype=torch.float32, device="cpu")
    latent = torch.randn(1, 5, 4)
    with torch.no_grad():
        waveform, state, pkv = vae.decode(latent, use_cache=False)
    assert waveform.dim() == 3
    assert waveform.shape[0] == 1 and waveform.shape[1] == 1
    assert torch.isfinite(waveform).all()


# ---------------------------------------------------------------------------
# Snapshot-gated structure asserts (key parity only — no weight load)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _find_local_snapshot() is None,
    reason="Need Ming-flash-omni-2.0 snapshot with talker/vae/.",
)
def test_audio_vae_module_keys_match_snapshot_state_dict() -> None:
    """Built AudioVAE.state_dict() contains the keys present in the ckpt.

    Smoke test for the eventual loader (step 6f): construct an AudioVAE
    from the real config, list its state_dict, and verify the major
    keys present in `talker/vae/model.safetensors` line up. We only
    check structural buckets (encoder.encoder.layers.0.*,
    decoder.decoder.*, fc1/fc2/fc3, head.out, head.istft.window,
    aggregator.layers.0.*) — full parameter coverage is the loader's job.
    """
    from safetensors import safe_open
    from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig

    snap = _find_local_snapshot()
    config = MingFlashOmniModelConfig.from_pretrained(snap)
    assert config.talker is not None
    vae = build_audio_vae(config.talker.vae, dtype=torch.float32, device="cpu")
    module_keys = set(vae.state_dict().keys())

    with safe_open(
        f"{snap}/talker/vae/model.safetensors", framework="pt"
    ) as f:
        ckpt_keys = set(f.keys())

    representative = {
        "encoder.fc1.weight",
        "encoder.fc2.weight",
        "encoder.fc3.weight",
        "encoder.norm.weight",
        "encoder.cls_embed",
        "encoder.encoder.embed_tokens.weight",
        "encoder.aggregator.embed_tokens.weight",
        "decoder.fc1.weight",
        "decoder.head.out.weight",
        "decoder.head.istft.window",
        "decoder.decoder.embed_tokens.weight",
    }
    missing_in_module = representative - module_keys
    assert not missing_in_module, f"Built VAE missing keys present in ckpt: {missing_in_module}"
    missing_in_ckpt = representative - ckpt_keys
    assert not missing_in_ckpt, f"Ckpt missing keys expected by VAE: {missing_in_ckpt}"
