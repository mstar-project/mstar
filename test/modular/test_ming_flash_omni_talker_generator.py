"""Tests for the TalkerGenerator orchestration helper (step 6e-1).

Covers the standalone helper class that composes Qwen2 + CFM +
Aggregator + stop_head + AudioVAE into the .generate_latents() /
.decode_to_waveform() API. Pure-Python tests with tiny stub-like
configs on CPU; integration with mminf's graph system lands in 6e-2.
"""

from __future__ import annotations

import pytest
import torch

from mminf.model.ming_omni_flash.components.audio_vae import build_audio_vae
from mminf.model.ming_omni_flash.components.talker_dit import (
    build_aggregator,
    build_talker_cfm,
    build_talker_heads,
    build_talker_llm,
)
from mminf.model.ming_omni_flash.components.talker_generator import (
    TalkerGenerator,
    silence_holder,
    trim_trailing_silence,
)
from mminf.model.ming_omni_flash.config import (
    AudioVAEConfig,
    DiTBlockConfig,
    TalkerConfig,
    TalkerLLMConfig,
)

# ---------------------------------------------------------------------------
# Tiny TalkerConfig + Qwen2 backbone for fast CPU tests
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


def _tiny_talker_config() -> TalkerConfig:
    """Tiny end-to-end TalkerConfig: 32-hidden, 1-layer talker LLM,
    8-channel CFM, 4-channel VAE — sized so all CPU forwards finish
    in <1s."""
    return TalkerConfig(
        steps=2,                # CFM substeps per generation step (use_predefined N/A)
        patch_size=2,           # CFM patch length
        history_patch_size=2,   # match patch_size for the simple update path
        cfg_strength=2.0,
        llm=TalkerLLMConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            sliding_window=64,
            max_window_layers=0,
            use_sliding_window=False,
        ),
        flowmodel=DiTBlockConfig(
            depth=1, hidden_size=32, num_heads=2,
            mlp_ratio=2, in_channels=4, dropout=0.0,
            attn_mask_enabled=False,
        ),
        aggregator=DiTBlockConfig(
            depth=1, hidden_size=32, num_heads=2,
            mlp_ratio=2, in_channels=4, dropout=0.0,
            attn_mask_enabled=False,
        ),
        vae=AudioVAEConfig(
            sample_rate=8000,
            patch_size=-1,             # no patching inside the VAE
            latent_dim=4,
            encoder_input_dim=16,
            encoder_hop_size=16,
            decoder_output_dim=16,
            enc_backbone=_tiny_qwen2_backbone(),
            dec_backbone=_tiny_qwen2_backbone(),
        ),
    )


def _build_generator(with_vae: bool = True) -> TalkerGenerator:
    cfg = _tiny_talker_config()
    llm = build_talker_llm(cfg.llm, dtype=torch.float32, device="cpu")
    cfm = build_talker_cfm(cfg, dtype=torch.float32, device="cpu")
    agg = build_aggregator(cfg, dtype=torch.float32, device="cpu")
    heads = build_talker_heads(cfg, dtype=torch.float32, device="cpu")
    vae = (
        build_audio_vae(cfg.vae, dtype=torch.float32, device="cpu", attn_implementation="sdpa")
        if with_vae else None
    )
    return TalkerGenerator(
        talker_config=cfg, llm=llm, cfm=cfm, aggregator=agg,
        stop_head=heads["stop_head"], audio_vae=vae,
    )


# ---------------------------------------------------------------------------
# trim_trailing_silence
# ---------------------------------------------------------------------------


def test_trim_trailing_silence_empty_waveform_passthrough() -> None:
    assert trim_trailing_silence(torch.zeros(0), sample_rate=8000).numel() == 0


def test_trim_trailing_silence_keeps_short_clip_intact() -> None:
    """Clips shorter than one frame get truncated to tail_silence budget."""
    sr = 8000
    short = torch.randn(1, 1, sr // 20)  # ~50 ms — shorter than 100 ms frame
    out = trim_trailing_silence(short, sample_rate=sr, tail_silence_s=0.3)
    assert out.shape[-1] == short.shape[-1]


def test_trim_trailing_silence_trims_silent_tail() -> None:
    """Long silent tail past a brief noisy region gets trimmed."""
    sr = 8000
    noisy = torch.randn(1, 1, sr) * 0.5      # 1.0s of noise
    silent = torch.zeros(1, 1, 2 * sr)        # 2.0s of silence
    waveform = torch.cat([noisy, silent], dim=-1)
    out = trim_trailing_silence(waveform, sample_rate=sr, tail_silence_s=0.3)
    # Should keep ~1.0s of noise + 0.3s of trailing silence.
    assert out.shape[-1] < waveform.shape[-1]
    assert out.shape[-1] >= int(0.9 * sr)  # at least ~0.9s


def test_trim_trailing_silence_passes_through_weird_shape() -> None:
    """4-D tensors aren't supported — return unchanged."""
    weird = torch.zeros(1, 2, 3, 4)
    out = trim_trailing_silence(weird, sample_rate=8000)
    assert out.shape == weird.shape


# ---------------------------------------------------------------------------
# silence_holder
# ---------------------------------------------------------------------------


def test_silence_holder_initial_cache_shape() -> None:
    """Empty input + no cache → returns empty tensor + empty cache."""
    out, cache = silence_holder(
        torch.zeros(1, 0), sample_rate=8000, sil_cache=None, last_chunk=True,
    )
    assert out.numel() == 0
    assert cache == {"holder": [], "buffer": []}


def test_silence_holder_short_chunk_buffers_until_last() -> None:
    """Sub-frame chunks accumulate in buffer until last_chunk=True."""
    sr = 8000
    cache = None
    out1, cache = silence_holder(
        torch.zeros(1, sr // 20), sample_rate=sr, sil_cache=cache, last_chunk=False,
    )
    assert out1.shape[-1] == 0   # buffered, nothing emitted
    out2, cache = silence_holder(
        torch.zeros(1, sr // 20), sample_rate=sr, sil_cache=cache, last_chunk=True,
    )
    # On last_chunk=True, the buffered + holder regions are flushed,
    # truncated to last_sil=0.3s.
    assert out2.shape[-1] <= int(0.3 * sr)


# ---------------------------------------------------------------------------
# TalkerGenerator — construction + state-machine sanity
# ---------------------------------------------------------------------------


def test_generator_constructs_with_all_components_bound() -> None:
    gen = _build_generator(with_vae=True)
    assert gen.patch_size == 2
    assert gen.his_patch_size == 2
    assert gen.latent_dim == 4
    assert gen.cfg_strength == 2.0
    assert gen.audio_vae is not None


def test_generator_constructs_without_audio_vae() -> None:
    gen = _build_generator(with_vae=False)
    assert gen.audio_vae is None


def test_init_his_lat_zeros_when_no_prompt() -> None:
    gen = _build_generator(with_vae=False)
    his = gen._init_his_lat(None, torch.device("cpu"), torch.float32)
    assert his.shape == (1, gen.his_patch_size, gen.latent_dim)
    assert (his == 0).all()


def test_init_his_lat_right_aligns_prompt() -> None:
    """Voice-prompt latents land at the right edge of the his window."""
    gen = _build_generator(with_vae=False)
    prompt = torch.randn(1, 1, gen.latent_dim)
    his = gen._init_his_lat(prompt, torch.device("cpu"), torch.float32)
    assert his.shape == (1, gen.his_patch_size, gen.latent_dim)
    # Right-most row should equal the prompt's single frame.
    torch.testing.assert_close(his[0, -1, :], prompt[0, 0, :])


def test_update_his_lat_equal_sizes_returns_gen() -> None:
    """When his_patch_size == patch_size, the new lat replaces the buffer."""
    gen = _build_generator(with_vae=False)
    his = torch.zeros(1, 2, 4)
    new = torch.ones(1, 2, 4)
    out = gen._update_his_lat(his, new)
    torch.testing.assert_close(out, new)


def test_update_his_lat_rejects_unsupported_shape() -> None:
    """his_patch_size < patch_size is not yet implemented."""
    gen = _build_generator(with_vae=False)
    gen.his_patch_size = 1
    gen.patch_size = 2
    with pytest.raises(NotImplementedError, match="his_patch_size"):
        gen._update_his_lat(torch.zeros(1, 1, 4), torch.zeros(1, 2, 4))


# ---------------------------------------------------------------------------
# Single-step plumbing (CFM step + LLM step)
# ---------------------------------------------------------------------------


def test_cfm_sample_step_returns_three_tensors_with_right_shapes() -> None:
    """gen_lat (B, patch, latent_dim); next_emb (B, 1, llm_hidden); stop (B, 2)."""
    gen = _build_generator(with_vae=False)
    last_hs = torch.randn(1, 1, gen.config.llm.hidden_size)
    his_lat = torch.zeros(1, gen.his_patch_size, gen.latent_dim)
    with torch.no_grad():
        gen_lat, next_emb, stop_out = gen.cfm_sample_step(
            last_hs, his_lat, cfg=2.0, sigma=0.0, temperature=0.0,
        )
    assert gen_lat.shape == (1, gen.patch_size, gen.latent_dim)
    assert next_emb.shape == (1, 1, gen.config.llm.hidden_size)
    assert stop_out.shape == (1, 2)
    # Softmax across dim=-1 sums to 1.
    torch.testing.assert_close(stop_out.sum(-1), torch.ones(1), atol=1e-5, rtol=0)


def test_llm_step_step0_no_cache_position() -> None:
    """On step 0 the LLM is called without cache_position; just verify it returns."""
    gen = _build_generator(with_vae=False)
    inputs_embeds = torch.randn(1, 3, gen.config.llm.hidden_size)
    with torch.no_grad():
        out = gen.llm_step(
            inputs_embeds, step=0, past_key_values=None, use_static_cache=False,
        )
    # Returns last hidden state row only.
    assert out.shape == (1, 1, gen.config.llm.hidden_size)


# ---------------------------------------------------------------------------
# generate_latents — full AR loop on tiny config
# ---------------------------------------------------------------------------


def test_generate_latents_collects_per_step_patches() -> None:
    """Loop emits one latent per step; min_new_token=0, max_steps=3."""
    gen = _build_generator(with_vae=False)
    inputs_embeds = torch.randn(1, 4, gen.config.llm.hidden_size)
    lats = gen.generate_latents(
        inputs_embeds,
        min_new_token=0,
        max_steps=3,
        use_static_cache=False,   # avoid StaticCache complexity in tests
    )
    # min_new_token=0 means we may stop early on any step with stop_prob > 0.5.
    # On random init, stop_prob is roughly 0.5; just verify we got *some* output.
    assert 1 <= len(lats) <= 3
    for lat in lats:
        assert lat.shape == (1, gen.patch_size, gen.latent_dim)


def test_generate_latents_respects_max_steps_cap() -> None:
    """When stop signal never fires (force it via min_new_token), max_steps caps the loop."""
    gen = _build_generator(with_vae=False)
    inputs_embeds = torch.randn(1, 2, gen.config.llm.hidden_size)
    lats = gen.generate_latents(
        inputs_embeds,
        min_new_token=1000,        # never satisfies stop check
        max_steps=4,
        use_static_cache=False,
    )
    assert len(lats) == 4


# ---------------------------------------------------------------------------
# duration_capped_steps
# ---------------------------------------------------------------------------


def test_duration_capped_steps_no_audio_vae_pass_through() -> None:
    gen = _build_generator(with_vae=False)
    assert gen.duration_capped_steps(text_len=100, requested_max_steps=1000) == 1000


def test_duration_capped_steps_uses_text_len_heuristic() -> None:
    """Long text → high cap; short text → low cap (capped at 2.0s minimum)."""
    gen = _build_generator(with_vae=True)
    short = gen.duration_capped_steps(text_len=1, requested_max_steps=10_000)
    long_ = gen.duration_capped_steps(text_len=100, requested_max_steps=10_000)
    assert short <= long_


# ---------------------------------------------------------------------------
# decode_to_waveform
# ---------------------------------------------------------------------------


def test_decode_to_waveform_empty_returns_zero_length() -> None:
    """Empty latent list → (1, 1, 0) zero-length waveform."""
    gen = _build_generator(with_vae=True)
    wf = gen.decode_to_waveform([], stream_decode=False)
    assert wf.shape == (1, 1, 0)


def test_decode_to_waveform_oneshot_runs_end_to_end() -> None:
    """Non-streaming path concatenates latents and runs one VAE decode."""
    gen = _build_generator(with_vae=True)
    latents = [torch.randn(1, gen.patch_size, gen.latent_dim) for _ in range(3)]
    with torch.no_grad():
        wf = gen.decode_to_waveform(latents, stream_decode=False)
    assert wf.dim() == 3
    assert wf.shape[0] == 1 and wf.shape[1] == 1
    assert wf.shape[-1] > 0
    assert torch.isfinite(wf).all()


def test_decode_to_waveform_streaming_runs_end_to_end() -> None:
    """Streaming path threads silence_holder + decode_pad through chunks."""
    gen = _build_generator(with_vae=True)
    latents = [torch.randn(1, gen.patch_size, gen.latent_dim) for _ in range(3)]
    with torch.no_grad():
        wf = gen.decode_to_waveform(latents, stream_decode=True)
    assert wf.dim() == 3
    assert torch.isfinite(wf).all()


def test_decode_to_waveform_raises_without_audio_vae() -> None:
    gen = _build_generator(with_vae=False)
    with pytest.raises(RuntimeError, match="audio_vae is None"):
        gen.decode_to_waveform([torch.zeros(1, 2, 4)])


# ---------------------------------------------------------------------------
# trim_trailing_silence (instance method)
# ---------------------------------------------------------------------------


def test_generator_trim_trailing_silence_uses_vae_sample_rate() -> None:
    gen = _build_generator(with_vae=True)
    sr = gen.audio_vae.config.sample_rate
    # Pure silence → trimmed to last_silence (0.3s default).
    silent = torch.zeros(1, 1, 4 * sr)
    out = gen.trim_trailing_silence(silent)
    assert out.shape[-1] <= int(0.3 * sr) + 1


def test_generator_trim_trailing_silence_without_vae_is_passthrough() -> None:
    gen = _build_generator(with_vae=False)
    x = torch.randn(1, 1, 1000)
    out = gen.trim_trailing_silence(x)
    torch.testing.assert_close(out, x)
