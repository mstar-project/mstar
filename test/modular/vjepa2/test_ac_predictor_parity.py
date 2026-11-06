"""Parity tests for the V-JEPA 2 action-conditioned predictor port.

Port source is ``vjepa2/src/models/ac_predictor.py`` +
``vjepa2/src/models/utils/modules.py`` (ACBlock, ACRoPEAttention).
Since the upstream repo requires ``timm`` and this env doesn't have it,
these tests exercise the port directly and validate:

  - Output shapes with/without extrinsics
  - Causal attention mask structure (frame t only attends to frames 0..t)
  - Causality at the output level: perturbing frame t's inputs must NOT
    change outputs for frames 0..t-1
  - State-dict key layout matches upstream (fused qkv Linear, etc.)
"""

from __future__ import annotations

import pytest
import torch

from mstar.model.vjepa2.components.ac_predictor import (
    ACBlock,
    VisionTransformerPredictorAC,
    build_action_block_causal_attention_mask,
)
from mstar.model.vjepa2.config import VJepa2ACPredictorConfig


def _tiny_ac_config(use_extrinsics: bool = False) -> VJepa2ACPredictorConfig:
    """Tiny config for fast CPU parity.

    Chosen so head_dim = predictor_embed_dim / num_heads = 24/4 = 6 and
    third = 2*(6//3//2) = 2, i.e. d_dim + h_dim + w_dim = 6 = head_dim (no
    residual rope slot).
    """
    return VJepa2ACPredictorConfig(
        img_size=(16, 16),
        patch_size=4,
        num_frames=4,
        tubelet_size=2,
        embed_dim=24,
        predictor_embed_dim=24,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=use_extrinsics,
    )


def test_causal_mask_shape_and_block_structure():
    """Each frame contributes (add_tokens + H*W) tokens. Mask must be
    block-lower-triangular at frame granularity."""
    mask = build_action_block_causal_attention_mask(grid_depth=3, grid_height=2, grid_width=2, add_tokens=2)
    tokens_per_frame = 2 + 2 * 2  # 6
    assert mask.shape == (3 * tokens_per_frame, 3 * tokens_per_frame)

    # Frame 0 can only see frame 0
    assert mask[:tokens_per_frame, :tokens_per_frame].all()
    assert not mask[:tokens_per_frame, tokens_per_frame:].any()

    # Frame 1 can see frames 0 and 1 (but not 2)
    assert mask[tokens_per_frame : 2 * tokens_per_frame, : 2 * tokens_per_frame].all()
    assert not mask[tokens_per_frame : 2 * tokens_per_frame, 2 * tokens_per_frame :].any()

    # Frame 2 can see everything
    assert mask[2 * tokens_per_frame :, :].all()


def test_ac_predictor_output_shape_no_extrinsics():
    torch.manual_seed(0)
    cfg = _tiny_ac_config(use_extrinsics=False)
    predictor = VisionTransformerPredictorAC(cfg).eval()

    b = 2
    t = cfg.num_frames // cfg.tubelet_size  # 2
    n_ctxt = t * (cfg.img_size[0] // cfg.patch_size) ** 2  # 2 * 16 = 32
    x = torch.randn(b, n_ctxt, cfg.embed_dim)
    actions = torch.randn(b, t, cfg.action_embed_dim)
    states = torch.randn(b, t, cfg.action_embed_dim)
    with torch.no_grad():
        out = predictor(x, actions, states)
    assert out.shape == (b, n_ctxt, cfg.embed_dim)


def test_ac_predictor_output_shape_with_extrinsics():
    torch.manual_seed(0)
    cfg = _tiny_ac_config(use_extrinsics=True)
    predictor = VisionTransformerPredictorAC(cfg).eval()

    b = 1
    t = cfg.num_frames // cfg.tubelet_size
    n_ctxt = t * (cfg.img_size[0] // cfg.patch_size) ** 2
    x = torch.randn(b, n_ctxt, cfg.embed_dim)
    actions = torch.randn(b, t, cfg.action_embed_dim)
    states = torch.randn(b, t, cfg.action_embed_dim)
    extrinsics = torch.randn(b, t, cfg.action_embed_dim - 1)
    with torch.no_grad():
        out = predictor(x, actions, states, extrinsics=extrinsics)
    assert out.shape == (b, n_ctxt, cfg.embed_dim)


def test_ac_predictor_extrinsics_required_when_flag_set():
    """Should error clearly if use_extrinsics=True but none passed."""
    cfg = _tiny_ac_config(use_extrinsics=True)
    predictor = VisionTransformerPredictorAC(cfg).eval()
    b = 1
    t = cfg.num_frames // cfg.tubelet_size
    n_ctxt = t * (cfg.img_size[0] // cfg.patch_size) ** 2
    x = torch.randn(b, n_ctxt, cfg.embed_dim)
    a = torch.randn(b, t, cfg.action_embed_dim)
    s = torch.randn(b, t, cfg.action_embed_dim)
    with pytest.raises(ValueError):
        predictor(x, a, s)


def test_ac_predictor_causality_at_output():
    """Perturbing inputs at frame T must NOT change outputs for frames 0..T-1.

    This is the end-to-end causality check — confirms the causal mask plus
    action/state token placement actually implement temporal isolation.
    """
    torch.manual_seed(0)
    cfg = _tiny_ac_config(use_extrinsics=False)
    predictor = VisionTransformerPredictorAC(cfg).eval()

    b = 1
    t = cfg.num_frames // cfg.tubelet_size
    h = cfg.img_size[0] // cfg.patch_size
    n_spatial = h * h
    n_ctxt = t * n_spatial

    x = torch.randn(b, n_ctxt, cfg.embed_dim)
    actions = torch.randn(b, t, cfg.action_embed_dim)
    states = torch.randn(b, t, cfg.action_embed_dim)

    with torch.no_grad():
        baseline = predictor(x, actions, states)

        # Perturb only frame t-1 (the last frame)
        x_p = x.clone()
        x_p[:, -n_spatial:, :] += 5.0
        a_p = actions.clone()
        a_p[:, -1, :] += 5.0
        s_p = states.clone()
        s_p[:, -1, :] += 5.0
        perturbed = predictor(x_p, a_p, s_p)

    # Earlier frames' outputs should be untouched
    n_early = (t - 1) * n_spatial
    assert torch.allclose(baseline[:, :n_early], perturbed[:, :n_early], atol=1e-5)

    # Last frame's output should have changed
    assert not torch.allclose(baseline[:, n_early:], perturbed[:, n_early:], atol=1e-3)


def test_ac_predictor_state_dict_keys():
    """State dict must use upstream naming: fused ``qkv.weight`` per layer,
    ``predictor_embed`` (not ``predictor_embeddings``), and the three
    action/state/extrinsics encoders."""
    cfg = _tiny_ac_config(use_extrinsics=True)
    predictor = VisionTransformerPredictorAC(cfg)
    keys = set(predictor.state_dict().keys())

    assert "predictor_embed.weight" in keys
    assert "action_encoder.weight" in keys
    assert "state_encoder.weight" in keys
    assert "extrinsics_encoder.weight" in keys
    # Fused qkv per layer
    assert "predictor_blocks.0.attn.qkv.weight" in keys
    assert "predictor_blocks.0.attn.proj.weight" in keys
    assert "predictor_blocks.0.norm1.weight" in keys
    assert "predictor_blocks.0.norm2.weight" in keys
    assert "predictor_blocks.0.mlp.fc1.weight" in keys
    assert "predictor_blocks.0.mlp.fc2.weight" in keys
    assert "predictor_norm.weight" in keys
    assert "predictor_proj.weight" in keys
    # attn_mask must NOT appear in state_dict (registered non-persistent)
    assert "attn_mask" not in keys


def test_ac_predictor_attn_mask_lazy_and_excluded_from_state_dict():
    """attn_mask is lazily computed (not a registered buffer) and never
    saved in checkpoints.  This matters because the model class uses
    ``meta → to_empty(device)`` which would zero a registered buffer.
    """
    cfg = _tiny_ac_config()
    predictor = VisionTransformerPredictorAC(cfg)
    # Lazy property builds on first access
    assert predictor.attn_mask is not None
    assert not predictor.attn_mask.requires_grad
    # Never in state_dict (and not registered as a buffer either)
    assert "attn_mask" not in predictor.state_dict()
    assert "attn_mask" not in dict(predictor.named_buffers())


def test_ac_predictor_survives_to_empty_materialization():
    """Reproduces the production path: build on meta, ``to_empty(device)``,
    then forward.  Before the fix, ``attn_mask`` was a non-persistent
    buffer and ``to_empty`` left it as uninitialized garbage, crashing
    forward.  Now that it's a lazy cache, forward rebuilds it on the
    first call.
    """
    torch.manual_seed(0)
    cfg = _tiny_ac_config()
    with torch.device("meta"):
        predictor = VisionTransformerPredictorAC(cfg)
    predictor = predictor.to_empty(device="cpu")
    # Fill the parameters with something deterministic since to_empty
    # leaves them uninitialized (real loading would come from a checkpoint).
    with torch.no_grad():
        for p in predictor.parameters():
            p.zero_()

    b = 1
    t = cfg.num_frames // cfg.tubelet_size
    n_ctxt = t * (cfg.img_size[0] // cfg.patch_size) ** 2
    x = torch.randn(b, n_ctxt, cfg.embed_dim)
    a = torch.randn(b, t, cfg.action_embed_dim)
    s = torch.randn(b, t, cfg.action_embed_dim)
    predictor.eval()
    with torch.no_grad():
        out = predictor(x, a, s)
    assert out.shape == (b, n_ctxt, cfg.embed_dim)
    assert torch.isfinite(out).all()


def test_ac_predictor_deterministic():
    """Same input → same output."""
    torch.manual_seed(0)
    cfg = _tiny_ac_config()
    predictor = VisionTransformerPredictorAC(cfg).eval()
    b = 1
    t = cfg.num_frames // cfg.tubelet_size
    n_ctxt = t * (cfg.img_size[0] // cfg.patch_size) ** 2
    x = torch.randn(b, n_ctxt, cfg.embed_dim)
    a = torch.randn(b, t, cfg.action_embed_dim)
    s = torch.randn(b, t, cfg.action_embed_dim)
    with torch.no_grad():
        out1 = predictor(x, a, s)
        out2 = predictor(x, a, s)
    assert torch.equal(out1, out2)


def test_ac_block_forward_runs():
    """ACBlock smoke test: runs without error, returns same shape."""
    torch.manual_seed(0)
    dim = 24
    num_heads = 4
    t, h, w = 2, 4, 4
    action_tokens = 2
    n = t * (action_tokens + h * w)

    blk = ACBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=2.0,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        grid_size=h,
        layer=0,
    ).eval()
    x = torch.randn(1, n, dim)
    mask = build_action_block_causal_attention_mask(t, h, w, add_tokens=action_tokens)
    with torch.no_grad():
        out = blk(x, attn_mask=mask, t=t, h=h, w=w, action_tokens=action_tokens)
    assert out.shape == x.shape
