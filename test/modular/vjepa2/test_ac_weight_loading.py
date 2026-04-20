"""Unit tests for the upstream → mminf key-rename helpers in
``mminf.model.vjepa2.weight_loader``.

These tests are pure CPU / pure Python: they build synthetic state_dicts
that exercise the upstream V-JEPA 2-AC key schema (as seen in
``vjepa2/src/hub/backbones.py::_make_vjepa2_ac_model`` and the layouts
defined by ``vjepa2/src/models/vision_transformer.py`` +
``vjepa2/src/models/ac_predictor.py``) and assert the renamer produces
keys + shapes that match our in-tree component modules.

No GPU, no HF network, no 11.7 GB checkpoint download.  Integration
tests against real weights live in ``test/integration/test_vjepa2_ac.py``.
"""

from __future__ import annotations

import torch

from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config
from mminf.model.vjepa2.weight_loader import (
    _rename_upstream_ac_predictor_keys,
    _rename_upstream_encoder_keys,
)

# ----------------------------------------------------------------------
# Encoder rename tests
# ----------------------------------------------------------------------


def test_qkv_split_shapes_and_bias_match_hf_convert():
    """Fused qkv.{weight,bias} must split into query/key/value.{weight,bias}
    with the same dim-0 slicing convention used by HF's ``convert_encoder_keys``.
    """
    hidden_size = 64
    fused_weight = torch.randn(3 * hidden_size, hidden_size)
    fused_bias = torch.randn(3 * hidden_size)
    raw = {
        "module.backbone.blocks.0.attn.qkv.weight": fused_weight,
        "module.backbone.blocks.0.attn.qkv.bias": fused_bias,
    }

    out = _rename_upstream_encoder_keys(raw, hidden_size=hidden_size)

    for name in ("query", "key", "value"):
        assert f"layer.0.attention.{name}.weight" in out
        assert f"layer.0.attention.{name}.bias" in out
    assert "layer.0.attention.qkv.weight" not in out
    assert "layer.0.attention.qkv.bias" not in out

    # Slicing matches convert_vjepa2_to_hf.py's [0:d], [d:2d], [2d:3d]
    torch.testing.assert_close(out["layer.0.attention.query.weight"], fused_weight[:hidden_size])
    torch.testing.assert_close(
        out["layer.0.attention.key.weight"],
        fused_weight[hidden_size : 2 * hidden_size],
    )
    torch.testing.assert_close(
        out["layer.0.attention.value.weight"],
        fused_weight[2 * hidden_size : 3 * hidden_size],
    )
    torch.testing.assert_close(out["layer.0.attention.query.bias"], fused_bias[:hidden_size])
    torch.testing.assert_close(
        out["layer.0.attention.key.bias"],
        fused_bias[hidden_size : 2 * hidden_size],
    )
    torch.testing.assert_close(
        out["layer.0.attention.value.bias"],
        fused_bias[2 * hidden_size : 3 * hidden_size],
    )


def test_module_backbone_prefix_strip():
    raw = {
        "module.backbone.blocks.5.norm1.weight": torch.zeros(8),
        "module.backbone.blocks.5.attn.proj.bias": torch.zeros(8),
    }
    out = _rename_upstream_encoder_keys(raw, hidden_size=8)
    assert "layer.5.norm1.weight" in out
    assert "layer.5.attention.proj.bias" in out
    # No residual upstream prefixes.
    for k in out:
        assert not k.startswith("module.")
        assert not k.startswith("backbone.")
        assert "blocks." not in k
        assert ".attn." not in k


def test_top_level_norm_becomes_layernorm_but_block_norms_unchanged():
    """Only the top-level ``norm.`` maps to ``layernorm.``.  Block-internal
    ``norm1`` / ``norm2`` must NOT get caught by the substitution.
    """
    raw = {
        "module.backbone.norm.weight": torch.randn(4),
        "module.backbone.norm.bias": torch.randn(4),
        "module.backbone.blocks.0.norm1.weight": torch.randn(4),
        "module.backbone.blocks.0.norm2.weight": torch.randn(4),
    }
    out = _rename_upstream_encoder_keys(raw, hidden_size=4)
    assert "layernorm.weight" in out
    assert "layernorm.bias" in out
    assert "layer.0.norm1.weight" in out
    assert "layer.0.norm2.weight" in out
    assert "layer.0.layernorm.weight" not in out
    assert "norm.weight" not in out  # original stripped


def test_patch_embed_rename():
    raw = {
        "module.backbone.patch_embed.proj.weight": torch.randn(8, 3, 2, 4, 4),
        "module.backbone.patch_embed.proj.bias": torch.randn(8),
    }
    out = _rename_upstream_encoder_keys(raw, hidden_size=8)
    assert "embeddings.patch_embeddings.proj.weight" in out
    assert "embeddings.patch_embeddings.proj.bias" in out
    assert "patch_embed.proj.weight" not in out


def test_pos_embed_dropped():
    """Upstream ships a pos_embed param (zero-inited when use_rope=True).
    Our VJEPA2Encoder has no position-embedding parameter, so the loader
    must drop it rather than trying to rename.
    """
    raw = {
        "module.backbone.pos_embed": torch.zeros(1, 1024, 16),
        "module.backbone.norm.weight": torch.randn(16),
    }
    out = _rename_upstream_encoder_keys(raw, hidden_size=16)
    assert "pos_embed" not in out
    assert "embeddings.position_embeddings" not in out
    assert "layernorm.weight" in out


def test_rename_loads_into_real_encoder():
    """End-to-end: rename an upstream-shaped state_dict and load it into a
    tiny real :class:`VJEPA2Encoder` with strict key coverage.

    This is the strongest possible unit check without real weights — if this
    passes, we know every key our encoder expects has a source in the
    upstream layout (modulo buffers, which don't appear in state_dict).
    """
    # Tiny config picked so all shapes are fast to allocate on CPU but the
    # 3D-RoPE head-dim-split still covers D/H/W slots.
    cfg = VJepa2Config(
        patch_size=4,
        crop_size=16,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=24,
        in_chans=3,
        num_attention_heads=4,
        num_hidden_layers=2,
        mlp_ratio=2.0,
    )
    encoder = VJEPA2Encoder(cfg)
    target_sd = encoder.state_dict()

    # Build a synthetic upstream state_dict that matches every slot our
    # encoder needs.  We fill it by inverting the rename: for each target key
    # we can deduce the upstream key.
    upstream: dict[str, torch.Tensor] = {}
    for key, tensor in target_sd.items():
        # Reverse: layer.X. -> blocks.X.; attention. -> attn.;
        # layernorm. -> norm.; embeddings.patch_embeddings. -> patch_embed.
        if key.startswith("layer."):
            up = key.replace("layer.", "blocks.", 1)
        elif key.startswith("layernorm."):
            up = key.replace("layernorm.", "norm.", 1)
        elif key.startswith("embeddings.patch_embeddings."):
            up = key.replace("embeddings.patch_embeddings.", "patch_embed.", 1)
        else:
            up = key
        up = up.replace(".attention.", ".attn.")
        upstream[f"module.backbone.{up}"] = tensor.clone()

    # Fold separate q/k/v back into a fused qkv (inverse of our splitter).
    hs = cfg.hidden_size
    fused: dict[str, torch.Tensor] = {}
    qkv_groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in list(upstream.items()):
        for name in ("query", "key", "value"):
            for suffix in ("weight", "bias"):
                tail = f".attn.{name}.{suffix}"
                if k.endswith(tail):
                    prefix = k[: -len(tail)] + f".attn.qkv.{suffix}"
                    qkv_groups.setdefault(prefix, {})[name] = v
                    break
    for prefix, parts in qkv_groups.items():
        if prefix.endswith("weight"):
            fused[prefix] = torch.cat([parts["query"], parts["key"], parts["value"]], dim=0)
        else:
            fused[prefix] = torch.cat([parts["query"], parts["key"], parts["value"]], dim=0)
    for k in list(upstream.keys()):
        for tail in (".query.", ".key.", ".value."):
            if tail in k and ".attn." in k:
                upstream.pop(k, None)
                break
    upstream.update(fused)

    renamed = _rename_upstream_encoder_keys(upstream, hidden_size=hs)
    missing, unexpected = encoder.load_state_dict(renamed, strict=False)
    # Strict coverage — every encoder param must have matched.
    assert missing == [], f"missing keys: {missing}"
    assert unexpected == [], f"unexpected keys: {unexpected}"


# ----------------------------------------------------------------------
# AC predictor rename tests
# ----------------------------------------------------------------------


def test_ac_predictor_prefix_strip():
    raw = {
        "module.predictor_embed.weight": torch.randn(32, 16),
        "module.predictor_embed.bias": torch.randn(32),
        "module.predictor_blocks.0.attn.qkv.weight": torch.randn(96, 32),
        "module.backbone.predictor_norm.weight": torch.randn(32),
    }
    out = _rename_upstream_ac_predictor_keys(raw)
    assert "predictor_embed.weight" in out
    assert "predictor_embed.bias" in out
    assert "predictor_blocks.0.attn.qkv.weight" in out
    assert "predictor_norm.weight" in out
    for k in out:
        assert not k.startswith("module.")
        assert not k.startswith("backbone.")


def test_ac_predictor_rename_loads_into_real_module():
    """Load a tiny synthetic AC predictor state_dict via the renamer.

    If our AC predictor class is key-compatible with upstream, adding an
    outer ``module.`` wrapper and stripping it back must leave a dict we can
    load with ``strict=False`` and zero missing keys.
    """
    cfg = VJepa2ACPredictorConfig(
        img_size=(16, 16),
        patch_size=4,
        num_frames=4,
        tubelet_size=2,
        embed_dim=24,
        predictor_embed_dim=24,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    predictor = VisionTransformerPredictorAC(cfg)
    target_sd = predictor.state_dict()

    # Wrap every key in the upstream DDP prefix, then strip.
    upstream = {f"module.{k}": v.clone() for k, v in target_sd.items()}

    renamed = _rename_upstream_ac_predictor_keys(upstream)
    missing, unexpected = predictor.load_state_dict(renamed, strict=False)
    assert missing == [], f"missing keys: {missing}"
    assert unexpected == [], f"unexpected keys: {unexpected}"
