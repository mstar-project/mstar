"""Tests for the ZImage DiT transformer (step 9b).

Pure-Python on CPU with a tiny config (the released model is dim=3840/30L —
far too large to instantiate in CI). Covers:

  * the interleaved (is_neox_style=False) RoPE matches vllm-omni's reference
    ``apply_rotary_emb_torch`` exactly (the one numeric correctness anchor that
    must agree with the validated serving path);
  * RMSNorm / timestep_embedding match the upstream formulas;
  * patchify→unpatchify roundtrips shape;
  * a full tiny forward runs and returns one latent per batch item with the
    input image shape;
  * the unfused checkpoint layout (to_q/to_k/to_v, w1/w3) loads via a direct
    copy with no stacked-param remap;
  * Ming's reference-latent subclass concatenates the ref frame and drops it
    from the output.
"""

from __future__ import annotations

import torch

from mminf.model.ming_omni_flash.components.zimage_transformer import (
    SEQ_MULTI_OF,
    MingZImageTransformer2DModel,
    RMSNorm,
    RopeEmbedder,
    ZImageTransformer2DModel,
    apply_rotary_emb_interleaved,
    timestep_embedding,
)


def _tiny_model(cls=ZImageTransformer2DModel) -> ZImageTransformer2DModel:
    # dim=16, head_dim=8, axes_dims sum to 8 (=head_dim) so RoPE covers the head.
    return cls(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=16,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        norm_eps=1e-5,
        cap_feat_dim=12,
        rope_theta=256.0,
        axes_dims=(2, 4, 2),
        axes_lens=(256, 256, 256),
    ).eval()


# ---------------------------------------------------------------------------
# RoPE numeric parity with vllm-omni's reference
# ---------------------------------------------------------------------------


def test_interleaved_rope_matches_vllm_reference() -> None:
    from einops import rearrange, repeat

    def rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    def ref(x, cos, sin):
        ro_dim = cos.shape[-1] * 2
        cos_r = repeat(cos, "... d -> ... 1 (d 2)")
        sin_r = repeat(sin, "... d -> ... 1 (d 2)")
        return torch.cat(
            [x[..., :ro_dim] * cos_r + rotate_half(x[..., :ro_dim]) * sin_r, x[..., ro_dim:]],
            dim=-1,
        )

    torch.manual_seed(0)
    x = torch.randn(2, 5, 3, 8)
    cos = torch.randn(2, 5, 4)
    sin = torch.randn(2, 5, 4)
    assert torch.allclose(ref(x, cos, sin), apply_rotary_emb_interleaved(x, cos, sin), atol=1e-6)


def test_rope_partial_dim_leaves_tail_untouched() -> None:
    x = torch.randn(1, 3, 2, 8)
    cos = torch.randn(1, 3, 2)  # ro_dim = 4 < head_dim 8
    sin = torch.randn(1, 3, 2)
    out = apply_rotary_emb_interleaved(x, cos, sin)
    assert torch.allclose(out[..., 4:], x[..., 4:])


def test_rope_embedder_axes_concatenate_to_half_head() -> None:
    emb = RopeEmbedder(theta=256.0, axes_dims=(2, 4, 2), axes_lens=(8, 8, 8))
    ids = torch.tensor([[0, 0, 0], [1, 2, 3]])
    cos, sin = emb(ids)
    # half-frequencies: sum(d/2) = 1 + 2 + 1 = 4
    assert cos.shape == (2, 4) and sin.shape == (2, 4)
    # position 0 across all axes -> cos=1, sin=0
    assert torch.allclose(cos[0], torch.ones(4))
    assert torch.allclose(sin[0], torch.zeros(4))


# ---------------------------------------------------------------------------
# Primitive parity
# ---------------------------------------------------------------------------


def test_rmsnorm_matches_manual_fp32() -> None:
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(2, 3, 8)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (norm.weight.float() * ref).to(x.dtype)
    assert torch.allclose(norm(x), ref, atol=1e-6)


def test_timestep_embedding_shape_and_formula() -> None:
    t = torch.tensor([0.0, 1.0, 5.0])
    emb = timestep_embedding(t, dim=16)
    assert emb.shape == (3, 16)
    # t=0 -> cos block all ones, sin block all zeros.
    assert torch.allclose(emb[0, :8], torch.ones(8))
    assert torch.allclose(emb[0, 8:], torch.zeros(8))


def test_timestep_embedding_odd_dim_zero_pad() -> None:
    emb = timestep_embedding(torch.tensor([1.0]), dim=7)
    assert emb.shape == (1, 7)
    assert emb[0, -1].item() == 0.0


# ---------------------------------------------------------------------------
# Patchify / unpatchify roundtrip
# ---------------------------------------------------------------------------


def test_patchify_unpatchify_roundtrip_shape() -> None:
    model = _tiny_model()
    # C=4, F=1, H=W=4 -> patch_size=2 -> 4 image tokens.
    img = torch.randn(4, 1, 4, 4)
    cap = torch.randn(SEQ_MULTI_OF, 12)
    (image_out, _, sizes, *_rest) = model.patchify_and_embed([img], [cap], patch_size=2, f_patch_size=1)
    assert sizes == [(1, 4, 4)]
    # image tokens padded up to a multiple of SEQ_MULTI_OF
    assert image_out[0].shape[0] % SEQ_MULTI_OF == 0
    assert image_out[0].shape[1] == 2 * 2 * 1 * 4  # pf*ph*pw*C


# ---------------------------------------------------------------------------
# Full tiny forward
# ---------------------------------------------------------------------------


def test_forward_returns_latent_per_item() -> None:
    model = _tiny_model()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.02)
    img = torch.randn(4, 1, 4, 4)
    cap = torch.randn(20, 12)
    t = torch.tensor([0.5])
    with torch.no_grad():
        out, _aux = model([img], t, [cap], patch_size=2, f_patch_size=1)
    assert isinstance(out, list) and len(out) == 1
    # out latent has the model's out_channels and the input image's F,H,W
    assert out[0].shape == (model.out_channels, 1, 4, 4)


def test_forward_batch_of_two() -> None:
    model = _tiny_model()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.02)
    imgs = [torch.randn(4, 1, 4, 4), torch.randn(4, 1, 4, 4)]
    caps = [torch.randn(18, 12), torch.randn(40, 12)]
    t = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        out, _ = model(imgs, t, caps, patch_size=2, f_patch_size=1)
    assert len(out) == 2
    assert out[0].shape == (model.out_channels, 1, 4, 4)


# ---------------------------------------------------------------------------
# Weight load: unfused layout copies directly
# ---------------------------------------------------------------------------


def test_load_weights_unfused_roundtrip() -> None:
    src = _tiny_model()
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    dst = _tiny_model()
    loaded = dst.load_weights(src.state_dict().items())
    assert loaded == set(dict(dst.named_parameters()).keys())
    for name, p in dst.named_parameters():
        assert torch.allclose(p, dict(src.named_parameters())[name])


def test_param_names_are_unfused() -> None:
    model = _tiny_model()
    names = set(dict(model.named_parameters()).keys())
    assert "layers.0.attention.to_q.weight" in names
    assert "layers.0.attention.to_k.weight" in names
    assert "layers.0.attention.to_v.weight" in names
    assert "layers.0.attention.to_out.0.weight" in names
    assert "layers.0.feed_forward.w1.weight" in names
    assert "layers.0.feed_forward.w3.weight" in names
    assert "layers.0.feed_forward.w2.weight" in names
    # no fused names leaked in
    assert not any("to_qkv" in n or "w13" in n for n in names)


def test_load_weights_shape_mismatch_raises() -> None:
    model = _tiny_model()
    import pytest

    with pytest.raises(ValueError, match="Shape mismatch"):
        model.load_weights({"x_pad_token": torch.zeros(1, 999)}.items())


# ---------------------------------------------------------------------------
# Ming reference-latent subclass
# ---------------------------------------------------------------------------


def test_ming_ref_latent_concats_and_drops_frame() -> None:
    model = _tiny_model(cls=MingZImageTransformer2DModel)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.02)
    img = torch.randn(4, 1, 4, 4)
    ref = torch.randn(4, 4, 4)  # [C, H, W] -> becomes one extra frame
    cap = torch.randn(20, 12)
    t = torch.tensor([0.5])
    with torch.no_grad():
        out, _ = model([img], t, [cap], patch_size=2, f_patch_size=1, ref_latent=[ref])
    # Output keeps only the first (non-reference) frame.
    assert out[0].shape == (model.out_channels, 1, 4, 4)


def test_ming_without_ref_latent_is_plain_t2i() -> None:
    model = _tiny_model(cls=MingZImageTransformer2DModel)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.02)
    img = torch.randn(4, 1, 4, 4)
    cap = torch.randn(20, 12)
    with torch.no_grad():
        out, _ = model([img], torch.tensor([0.5]), [cap], patch_size=2, f_patch_size=1)
    assert out[0].shape == (model.out_channels, 1, 4, 4)
