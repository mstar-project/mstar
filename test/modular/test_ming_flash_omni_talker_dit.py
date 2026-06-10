"""Tests for the Ming-flash-omni-2.0 Talker DiT + CFM (step 6b).

Covers the building blocks ported in ``components/talker_dit.py``:

  * RotaryEmbedding's interleaved-pair layout (rotate_half) matches
    x_transformers' convention so the released ckpt's weights load
    against the same RoPE shape they were trained with.
  * DiTTimestepEmbedding outputs the right shape and is dtype-stable.
  * RMSNorm / FeedForward / Attention / DiTBlock / FinalLayer /
    CondEmbedder shapes round-trip correctly.
  * DiT.forward / forward_with_cfg returns the expected dims given a
    flowmodel-shaped config.
  * CFM.sample integrates over the EPSS schedule and returns the
    initial-noise shape unchanged.
  * build_talker_cfm constructs a DiT + CFM from a real TalkerConfig
    without needing the checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from mminf.model.ming_omni_flash.components.talker_dit import (
    CFM,
    DiT,
    DiTTimestepEmbedding,
    RotaryEmbedding,
    _apply_rotary_pos_emb,
    _Attention,
    _CondEmbedder,
    _DiTBlock,
    _FeedForward,
    _FinalLayer,
    _RMSNorm,
    _rotate_half_interleaved,
    _SinusPositionEmbedding,
    build_talker_cfm,
    get_epss_timesteps,
)
from mminf.model.ming_omni_flash.config import (
    AudioVAEConfig,
    DiTBlockConfig,
    TalkerConfig,
    TalkerLLMConfig,
)

# ---------------------------------------------------------------------------
# RotaryEmbedding — match x_transformers' interleaved-pair layout
# ---------------------------------------------------------------------------


def test_rotate_half_interleaved_matches_pair_negation() -> None:
    """``(x1, x2, x3, x4) -> (-x2, x1, -x4, x3)`` per the upstream rotate_half."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = _rotate_half_interleaved(x)
    torch.testing.assert_close(out, torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))


def test_rotary_embedding_forward_from_seq_len_shape_and_pair_repeat() -> None:
    """``(1, T, dim)`` with each adjacent pair sharing the same frequency."""
    rope = RotaryEmbedding(dim=4)
    freqs, xpos = rope.forward_from_seq_len(seq_len=3)
    assert freqs.shape == (1, 3, 4)
    assert xpos is None
    # Adjacent pairs share the same value: freqs[:, t, 0] == freqs[:, t, 1].
    torch.testing.assert_close(freqs[..., 0::2], freqs[..., 1::2])


def test_apply_rotary_pos_emb_partial_rotation_preserves_passed_through() -> None:
    """Trailing channels beyond ``rot_dim`` are untouched."""
    rope = RotaryEmbedding(dim=4)
    freqs, _ = rope.forward_from_seq_len(seq_len=2)
    # 6-channel tensor; only first 4 should rotate.
    t = torch.randn(1, 2, 2, 6)  # (B, H, T, head_dim)
    out = _apply_rotary_pos_emb(t, freqs)
    assert out.shape == t.shape
    # Last 2 channels unchanged.
    torch.testing.assert_close(out[..., 4:], t[..., 4:])


# ---------------------------------------------------------------------------
# DiTTimestepEmbedding
# ---------------------------------------------------------------------------


def test_sinus_position_embedding_concat_sin_cos_shape() -> None:
    emb = _SinusPositionEmbedding(dim=8)
    out = emb(torch.tensor([0.0, 1.0, 2.0]))
    assert out.shape == (3, 8)
    # Halves are sin / cos so sum-of-squares per row should be 4 (= half_dim).
    sq = (out ** 2).sum(dim=-1)
    torch.testing.assert_close(sq, torch.full((3,), 4.0))


def test_sinus_position_embedding_rejects_odd_dim() -> None:
    with pytest.raises(ValueError, match="must be even"):
        _SinusPositionEmbedding(dim=7)


def test_dit_timestep_embedding_shape_and_dtype() -> None:
    """MLP output is (N, hidden_size); dtype follows the input."""
    embed = DiTTimestepEmbedding(dim=16, freq_embed_dim=8)
    t = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    out = embed(t)
    assert out.shape == (3, 16)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# DiT building blocks
# ---------------------------------------------------------------------------


def test_rmsnorm_normalises_per_row_to_unit_var() -> None:
    norm = _RMSNorm(dim=4, eps=1e-12)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = norm(x)
    # rms = sqrt((1+4+9+16)/4) = sqrt(7.5).
    expected = x / (7.5 ** 0.5)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_feed_forward_layer_indices_match_checkpoint_keys() -> None:
    """FF inner-Sequential indices must align with the released talker
    ckpt's ``blocks.N.mlp.ff.0.0`` / ``ff.0.1`` / ``ff.2`` keys.

    ff.0 → Sequential(Linear, GELU); ff.1 → Dropout; ff.2 → Linear.
    """
    ff = _FeedForward(dim=8, mult=2, dropout=0.0)
    # ff is the outer Sequential; named children
    seq = ff.ff
    assert isinstance(seq[0], torch.nn.Sequential)
    assert isinstance(seq[0][0], torch.nn.Linear)
    assert isinstance(seq[0][1], torch.nn.GELU)
    assert isinstance(seq[1], torch.nn.Dropout)
    assert isinstance(seq[2], torch.nn.Linear)
    # Round-trip shape.
    x = torch.randn(2, 4, 8)
    assert ff(x).shape == (2, 4, 8)


def test_attention_to_q_to_k_to_v_to_out_param_names() -> None:
    """Upstream weight keys: blocks.N.attn.{to_q,to_k,to_v}.weight + to_out.0.weight."""
    attn = _Attention(dim=8, heads=2, dim_head=4)
    keys = set(dict(attn.named_parameters()).keys())
    for must_have in [
        "to_q.weight", "to_q.bias",
        "to_k.weight", "to_k.bias",
        "to_v.weight", "to_v.bias",
        "to_out.0.weight", "to_out.0.bias",
    ]:
        assert must_have in keys, f"missing param {must_have!r}"
    # qk_norm=None → no q_norm / k_norm params.
    assert not any(k.startswith("q_norm") or k.startswith("k_norm") for k in keys)


def test_attention_forward_shape_no_rope() -> None:
    attn = _Attention(dim=16, heads=2, dim_head=8)
    x = torch.randn(2, 5, 16)
    out = attn(x, rope=None)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_attention_forward_shape_with_rope() -> None:
    attn = _Attention(dim=16, heads=2, dim_head=8)
    rope = RotaryEmbedding(dim=8)
    rope_freqs = rope.forward_from_seq_len(5)
    x = torch.randn(2, 5, 16)
    out = attn(x, rope=rope_freqs)
    assert out.shape == (2, 5, 16)


def test_attention_qk_norm_rms_adds_q_norm_k_norm_params() -> None:
    attn = _Attention(dim=16, heads=2, dim_head=8, qk_norm="rms_norm")
    keys = set(dict(attn.named_parameters()).keys())
    assert "q_norm.weight" in keys
    assert "k_norm.weight" in keys


def test_attention_rejects_unknown_qk_norm() -> None:
    with pytest.raises(ValueError, match="Unimplemented qk_norm"):
        _Attention(dim=16, heads=2, dim_head=8, qk_norm="layer_norm")


def test_dit_block_forward_runs_with_rope() -> None:
    blk = _DiTBlock(hidden_size=16, num_heads=2, mlp_ratio=2)
    rope = RotaryEmbedding(dim=8).forward_from_seq_len(5)
    x = torch.randn(2, 5, 16)
    # 6c added a mask argument to DiTBlock.forward (Aggregator needs it);
    # the CFM/DiT call path passes mask=None.
    out = blk(x, None, rope)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_final_layer_projects_to_out_channels() -> None:
    f = _FinalLayer(hidden_size=16, out_channels=64)
    x = torch.randn(2, 5, 16)
    out = f(x)
    assert out.shape == (2, 5, 64)


def test_cond_embedder_projects_llm_to_dit_hidden() -> None:
    c = _CondEmbedder(input_feature_size=896, hidden_size=1024)
    x = torch.randn(2, 1, 896)
    out = c(x)
    assert out.shape == (2, 1, 1024)


# ---------------------------------------------------------------------------
# DiT — full assembly
# ---------------------------------------------------------------------------


def _tiny_dit(spk_dim: int | None = None) -> DiT:
    return DiT(
        in_channels=8,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        llm_cond_dim=8,
        dropout=0.0,
        spk_dim=spk_dim,
    )


def test_dit_forward_output_shape_includes_prefix_tokens() -> None:
    """DiT outputs ``(B, 1 + his + patch, out_channels)`` (no spk_embedder)."""
    dit = _tiny_dit()
    B, his, patch = 2, 4, 3
    x = torch.randn(B, patch, 8)
    t = torch.tensor([0.5, 0.5])
    c = torch.randn(B, 1, 8)
    lh = torch.randn(B, his, 8)
    out = dit(x, t, c, lh)
    # Sequence: y (1 token) + latent_history (his) + x (patch).
    assert out.shape == (B, 1 + his + patch, 8)
    assert torch.isfinite(out).all()


def test_dit_forward_with_cfg_returns_only_x_rows() -> None:
    """CFG forward doubles batch internally; returns the trailing x rows."""
    dit = _tiny_dit()
    x = torch.randn(2, 3, 8)
    t = torch.tensor(0.3)
    c = torch.randn(2, 1, 8)
    lh = torch.randn(2, 4, 8)
    out = dit.forward_with_cfg(x, t, c, lh)
    # Doubled batch (B*2) keeps the original batch dim before chunk.
    # forward_with_cfg slices the last x.shape[1] rows → (B*2, patch, out).
    assert out.shape == (4, 3, 8)


def test_dit_spk_embedder_absent_raises_when_emb_supplied() -> None:
    """Explicit shape contract: providing spk_emb when spk_embedder=None is a bug."""
    dit = _tiny_dit(spk_dim=None)
    with pytest.raises(AssertionError, match="spk_embedder"):
        dit(
            x=torch.randn(2, 3, 8),
            t=torch.tensor([0.5, 0.5]),
            c=torch.randn(2, 1, 8),
            latent_history=torch.randn(2, 4, 8),
            spk_emb=torch.randn(2, 16),
        )


def test_dit_with_spk_embedder_concats_spk_token() -> None:
    dit = _tiny_dit(spk_dim=16)
    x = torch.randn(2, 3, 8)
    out = dit(
        x=x,
        t=torch.tensor([0.5, 0.5]),
        c=torch.randn(2, 1, 8),
        latent_history=torch.randn(2, 4, 8),
        # spk_emb is (B, 1, spk_dim) — same 3D shape as c. The
        # spk_embedder projects (B, 1, spk_dim) → (B, 1, hidden_size)
        # and gets concatenated alongside y on dim=1.
        spk_emb=torch.randn(2, 1, 16),
    )
    # spk (1) + y (1) + his (4) + patch (3) = 9.
    assert out.shape == (2, 9, 8)


# ---------------------------------------------------------------------------
# CFM + EPSS schedule
# ---------------------------------------------------------------------------


def test_get_epss_timesteps_predefined_n_10_matches_upstream_schedule() -> None:
    """Released ckpt uses steps=10 — schedule must match upstream exactly."""
    t = get_epss_timesteps(10, device="cpu", dtype=torch.float32)
    expected = (1 / 32) * torch.tensor(
        [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], dtype=torch.float32,
    )
    torch.testing.assert_close(t, expected)
    assert t.numel() == 11   # steps + 1


def test_get_epss_timesteps_falls_back_to_linspace_for_unknown_n() -> None:
    t = get_epss_timesteps(9, device="cpu", dtype=torch.float32)
    expected = torch.linspace(0, 1, 10, dtype=torch.float32)
    torch.testing.assert_close(t, expected)


def test_cfm_sample_returns_same_shape_as_y0() -> None:
    """Smoke: CFM.sample preserves the noise tensor shape after integration."""
    dit = _tiny_dit()
    cfm = CFM(model=dit, steps=4)
    B, patch, latent = 2, 3, 8
    llm_cond = torch.randn(B, 1, 8)
    lat_cond = torch.randn(B, 5, latent)
    y0 = torch.randn(B, patch, latent)
    t = get_epss_timesteps(4, device="cpu", dtype=torch.float32)
    sde_args = torch.tensor([2.0, 0.0, 0.0])   # cfg=2, no sde noise
    sde_rnd = torch.zeros(4, B, patch, latent)
    out = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)
    assert out.shape == y0.shape
    assert torch.isfinite(out).all()


def test_cfm_sample_rejects_mismatched_t_length() -> None:
    dit = _tiny_dit()
    cfm = CFM(model=dit, steps=4)
    bad_t = torch.zeros(3)
    with pytest.raises(ValueError, match="length steps\\+1 = 5"):
        cfm.sample(
            llm_cond=torch.randn(1, 1, 8),
            lat_cond=torch.randn(1, 5, 8),
            y0=torch.randn(1, 3, 8),
            t=bad_t,
            sde_args=torch.tensor([2.0, 0.0, 0.0]),
            sde_rnd=torch.zeros(4, 1, 3, 8),
        )


def test_cfm_sample_rejects_mismatched_sde_rnd_first_dim() -> None:
    dit = _tiny_dit()
    cfm = CFM(model=dit, steps=4)
    t = get_epss_timesteps(4, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="sde_rnd\\[0\\] = 4"):
        cfm.sample(
            llm_cond=torch.randn(1, 1, 8),
            lat_cond=torch.randn(1, 5, 8),
            y0=torch.randn(1, 3, 8),
            t=t,
            sde_args=torch.tensor([2.0, 0.0, 0.0]),
            sde_rnd=torch.zeros(3, 1, 3, 8),
        )


def test_cfm_no_sway_skips_remap() -> None:
    """sway_sampling_coef=None must skip the t remap (sanity-check the branch)."""
    dit = _tiny_dit()
    cfm = CFM(model=dit, steps=4, sway_sampling_coef=None)
    t = get_epss_timesteps(4, device="cpu", dtype=torch.float32)
    out = cfm.sample(
        llm_cond=torch.randn(1, 1, 8),
        lat_cond=torch.randn(1, 5, 8),
        y0=torch.randn(1, 3, 8),
        t=t,
        sde_args=torch.tensor([0.0, 0.0, 0.0]),
        sde_rnd=torch.zeros(4, 1, 3, 8),
    )
    assert out.shape == (1, 3, 8)


# ---------------------------------------------------------------------------
# build_talker_cfm factory
# ---------------------------------------------------------------------------


def test_build_talker_cfm_from_real_config() -> None:
    """Released ckpt's TalkerConfig (defaults) yields the expected DiT shape."""
    cfg = TalkerConfig(
        llm=TalkerLLMConfig(),
        flowmodel=DiTBlockConfig(),
        aggregator=DiTBlockConfig(),
        vae=AudioVAEConfig(),
    )
    cfm = build_talker_cfm(cfg, dtype=torch.float32, device="cpu")
    assert isinstance(cfm, CFM)
    assert cfm.steps == cfg.steps   # 10
    dit = cfm.model
    assert isinstance(dit, DiT)
    assert dit.hidden_size == cfg.flowmodel.hidden_size   # 1024
    assert dit.in_channels == cfg.flowmodel.in_channels   # 64
    assert dit.num_heads == cfg.flowmodel.num_heads       # 16
    assert len(dit.blocks) == cfg.flowmodel.depth         # 8
    # llm_cond_dim defaults to talker LLM hidden_size (896 on released ckpt).
    assert dit.c_embedder.cond_embedder.in_features == cfg.llm.hidden_size


def test_build_talker_cfm_accepts_llm_cond_dim_override() -> None:
    cfg = TalkerConfig()
    cfm = build_talker_cfm(cfg, llm_cond_dim=4096, dtype=torch.float32, device="cpu")
    assert cfm.model.c_embedder.cond_embedder.in_features == 4096


# ---------------------------------------------------------------------------
# Step 6c — Attention mask handling
# ---------------------------------------------------------------------------


def test_attention_mask_zeros_output_at_padded_positions() -> None:
    """``mask=False`` rows in input get zeroed in the output regardless of
    `attn_mask_enabled` (mirrors upstream's unconditional masked_fill).
    """
    attn = _Attention(dim=8, heads=2, dim_head=4, attn_mask_enabled=False)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    out = attn(x, mask=mask)
    # First 2 rows should be the live attention output; last 2 zero.
    assert (out[:, 2:].abs().sum() == 0).item()
    assert (out[:, :2].abs().sum() > 0).item()


def test_attention_mask_enabled_uses_sdpa_attn_mask() -> None:
    """With attn_mask_enabled=True, padded keys shouldn't contribute to softmax.

    Smoke check: forward runs without error and the live output rows
    are still finite (we don't assert numerical equivalence to the
    unmasked case since SDPA's mask changes attention weights).
    """
    attn = _Attention(dim=8, heads=2, dim_head=4, attn_mask_enabled=True)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    out = attn(x, mask=mask)
    assert torch.isfinite(out[:, :2]).all()


def test_attention_no_mask_no_zeroing() -> None:
    """mask=None must NOT zero anything (regression guard against the
    upstream branch that only runs masked_fill when mask is not None).
    """
    attn = _Attention(dim=8, heads=2, dim_head=4, attn_mask_enabled=False)
    x = torch.randn(1, 4, 8)
    out = attn(x, mask=None)
    assert (out.abs().sum() > 0).item()


# ---------------------------------------------------------------------------
# Step 6c — Aggregator
# ---------------------------------------------------------------------------


def _tiny_aggregator(llm_input_dim: int = 16):
    from mminf.model.ming_omni_flash.components.talker_dit import Aggregator
    return Aggregator(
        in_channels=8,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        llm_input_dim=llm_input_dim,
        dropout=0.0,
    )


def test_aggregator_outputs_cls_row_only() -> None:
    """Aggregator returns ``(B, 1, llm_input_dim)`` — the [CLS] row."""
    agg = _tiny_aggregator(llm_input_dim=16)
    x = torch.randn(2, 5, 8)
    out = agg(x)
    assert out.shape == (2, 1, 16)
    assert torch.isfinite(out).all()


def test_aggregator_word_embedder_has_single_row() -> None:
    """``nn.Embedding(1, hidden_size)`` — exactly one [CLS] token."""
    agg = _tiny_aggregator()
    assert agg.word_embedder.num_embeddings == 1
    assert agg.word_embedder.embedding_dim == 16


def test_aggregator_respects_mask_in_dit_blocks() -> None:
    """With a key-padding mask, the masked rows still don't contaminate the
    [CLS] output (since the DiT blocks zero them out before the final
    layer).  Verify the forward at least runs through with a mask.
    """
    agg = _tiny_aggregator(llm_input_dim=16)
    x = torch.randn(1, 5, 8)
    # mark last 2 positions invalid
    mask = torch.tensor([[True, True, True, False, False]])
    out = agg(x, mask=mask)
    assert out.shape == (1, 1, 16)
    assert torch.isfinite(out).all()


def test_aggregator_forward_matches_shape_for_various_T() -> None:
    agg = _tiny_aggregator(llm_input_dim=16)
    for T in (1, 4, 8):
        out = agg(torch.randn(2, T, 8))
        assert out.shape == (2, 1, 16)


def test_build_aggregator_from_real_config() -> None:
    """build_aggregator picks dims off TalkerConfig.aggregator."""
    from mminf.model.ming_omni_flash.components.talker_dit import (
        Aggregator,
        build_aggregator,
    )
    cfg = TalkerConfig(
        llm=TalkerLLMConfig(),
        flowmodel=DiTBlockConfig(),
        aggregator=DiTBlockConfig(),
        vae=AudioVAEConfig(),
    )
    agg = build_aggregator(cfg, dtype=torch.float32, device="cpu")
    assert isinstance(agg, Aggregator)
    assert agg.hidden_size == cfg.aggregator.hidden_size   # 1024
    assert len(agg.blocks) == cfg.aggregator.depth         # 8
    # final_layer projects to llm_input_dim = talker.llm.hidden_size = 896.
    assert agg.final_layer.linear.out_features == cfg.llm.hidden_size


def test_build_aggregator_llm_input_dim_override() -> None:
    from mminf.model.ming_omni_flash.components.talker_dit import build_aggregator
    cfg = TalkerConfig()
    agg = build_aggregator(cfg, llm_input_dim=2048, dtype=torch.float32, device="cpu")
    assert agg.final_layer.linear.out_features == 2048


# ---------------------------------------------------------------------------
# Step 6c — Qwen2 talker LLM backbone
# ---------------------------------------------------------------------------


def test_build_talker_llm_returns_qwen2_model_with_correct_dims() -> None:
    """Stock transformers.Qwen2Model with our TalkerLLMConfig dims."""
    from mminf.model.ming_omni_flash.components.talker_dit import build_talker_llm
    llm_cfg = TalkerLLMConfig(
        vocab_size=128,            # tiny vocab for speed
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=32,
        max_window_layers=0,
    )
    model = build_talker_llm(llm_cfg, dtype=torch.float32, device="cpu")
    # The HF class is Qwen2Model.
    from transformers import Qwen2Model
    assert isinstance(model, Qwen2Model)
    # Dims match what we passed in.
    assert model.config.hidden_size == 64
    assert model.config.num_hidden_layers == 2
    assert model.config.num_attention_heads == 4
    assert model.config.num_key_value_heads == 2
    assert model.config.vocab_size == 128


def test_build_talker_llm_forward_runs_on_tiny_input() -> None:
    """Forward pass through the tiny Qwen2 backbone returns hidden states."""
    from mminf.model.ming_omni_flash.components.talker_dit import build_talker_llm
    llm_cfg = TalkerLLMConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, sliding_window=32, max_window_layers=0,
    )
    model = build_talker_llm(llm_cfg, dtype=torch.float32, device="cpu")
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    # Qwen2Model.forward returns a BaseModelOutputWithPast.
    assert out.last_hidden_state.shape == (1, 4, 32)
    assert torch.isfinite(out.last_hidden_state).all()


# ---------------------------------------------------------------------------
# Step 6c — Talker heads
# ---------------------------------------------------------------------------


def test_build_talker_heads_emits_stop_head_and_spk_head() -> None:
    """stop_head: hidden → 2 (binary), spk_head: 192 → hidden."""
    from mminf.model.ming_omni_flash.components.talker_dit import build_talker_heads
    cfg = TalkerConfig(llm=TalkerLLMConfig())  # hidden_size=896
    heads = build_talker_heads(cfg, dtype=torch.float32, device="cpu")
    assert "stop_head" in heads and "spk_head" in heads
    sh = heads["stop_head"]
    assert sh.in_features == cfg.llm.hidden_size   # 896
    assert sh.out_features == 2
    assert sh.bias is not None
    spk = heads["spk_head"]
    assert spk.in_features == 192
    assert spk.out_features == cfg.llm.hidden_size  # 896
    assert spk.bias is not None


def test_build_talker_heads_spk_dim_override() -> None:
    from mminf.model.ming_omni_flash.components.talker_dit import build_talker_heads
    cfg = TalkerConfig(llm=TalkerLLMConfig())
    heads = build_talker_heads(cfg, spk_embed_dim=512, dtype=torch.float32, device="cpu")
    assert heads["spk_head"].in_features == 512
