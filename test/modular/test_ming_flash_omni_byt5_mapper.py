"""Tests for the ByT5 glyph mapper (step 9b).

Two layers:

  * Pure-Python structure/shape tests for ``T5EncoderBlockByT5Mapper`` using a
    tiny HF ``T5Config`` — verify block stacking, position-bias reuse across
    layers, the d_model→sdxl_channels projection, pad-mask handling, and that
    Ming's unfused ``byt5_mapper.pt`` name layout loads with a plain
    ``load_weights`` (no fused remap needed). Run on CPU, no snapshot.

  * Snapshot-gated build of the full ``MingByT5Encoder`` from the real
    checkpoint's ``byt5`` dir (skipped when absent).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import T5Config

from mminf.model.ming_omni_flash.components.byte5_encoder import MingByT5Encoder
from mminf.model.ming_omni_flash.components.t5_block_mapper import (
    T5EncoderBlockByT5Mapper,
)


def _tiny_t5_config() -> T5Config:
    return T5Config(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        vocab_size=384,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        layer_norm_epsilon=1e-6,
        is_encoder_decoder=False,
        is_decoder=False,
        dropout_rate=0.0,
        # Ming's byt5 is gated (wi_0/wi_1), matching the released checkpoint —
        # the default "relu" would build a single fused wi and break the
        # unfused-name load path we exercise below.
        feed_forward_proj="gated-gelu",
    )


# ---------------------------------------------------------------------------
# Mapper structure / forward
# ---------------------------------------------------------------------------


def test_mapper_projects_to_sdxl_channels() -> None:
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=2, sdxl_channels=48).eval()
    x = torch.randn(3, 7, cfg.d_model)
    mask = torch.ones(3, 7)
    out = mapper(x, mask)
    assert out.shape == (3, 7, 48)


def test_mapper_no_projection_keeps_d_model() -> None:
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=1, sdxl_channels=None).eval()
    assert mapper.channel_mapper is None and mapper.final_layer_norm is None
    out = mapper(torch.randn(2, 5, cfg.d_model), torch.ones(2, 5))
    assert out.shape == (2, 5, cfg.d_model)


def test_mapper_zero_layers_is_norm_plus_project() -> None:
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=0, sdxl_channels=16).eval()
    assert mapper.blocks is None
    out = mapper(torch.randn(1, 4, cfg.d_model), torch.ones(1, 4))
    assert out.shape == (1, 4, 16)


def test_mapper_only_first_block_has_relative_bias() -> None:
    """T5 weight-sharing convention: relative_attention_bias lives on block 0."""
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=3, sdxl_channels=None)
    has_bias = [
        any("relative_attention_bias" in n for n, _ in blk.named_parameters())
        for blk in mapper.blocks
    ]
    assert has_bias == [True, False, False]


def test_mapper_pad_mask_changes_output() -> None:
    """Masking out the tail should change the kept positions' representation."""
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=2, sdxl_channels=None).eval()
    x = torch.randn(1, 6, cfg.d_model)
    full = torch.ones(1, 6)
    half = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        out_full = mapper(x, full)
        out_half = mapper(x, half)
    # The first (kept) token attends to fewer keys under the half mask.
    assert not torch.allclose(out_full[:, 0], out_half[:, 0], atol=1e-5)


def test_extended_attention_mask_additive_form() -> None:
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=1, sdxl_channels=None)
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    ext = mapper.get_extended_attention_mask(mask, dtype=torch.float32)
    assert ext.shape == (1, 1, 1, 3)
    assert ext[0, 0, 0, 0].item() == 0.0
    assert ext[0, 0, 0, 1].item() == torch.finfo(torch.float32).min
    assert ext[0, 0, 0, 2].item() == 0.0


def test_extended_attention_mask_rejects_bad_rank() -> None:
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=0, sdxl_channels=None)
    with pytest.raises(ValueError, match="Unexpected attention_mask shape"):
        mapper.get_extended_attention_mask(torch.ones(2, 3, 4, 5), dtype=torch.float32)


# ---------------------------------------------------------------------------
# load_weights: Ming's unfused byt5_mapper.pt name layout loads directly
# ---------------------------------------------------------------------------


def test_load_weights_roundtrips_unfused_layout() -> None:
    cfg = _tiny_t5_config()
    src = T5EncoderBlockByT5Mapper(cfg, num_layers=2, sdxl_channels=24)
    # Randomize so a successful load is observable (not the init values).
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    dst = T5EncoderBlockByT5Mapper(cfg, num_layers=2, sdxl_channels=24)
    loaded = dst.load_weights(src.state_dict().items())
    # Every dst parameter should have been covered by the source state dict.
    assert loaded == set(dict(dst.named_parameters()).keys())
    for name, p in dst.named_parameters():
        assert torch.allclose(p, dict(src.named_parameters())[name])


def test_load_weights_source_names_match_ming_unfused_format() -> None:
    """Sanity: the param names we expose are exactly Ming's checkpoint keys
    (unfused q/k/v/o + wi_0/wi_1/wo), so no stacked-param remap is needed."""
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=1, sdxl_channels=8)
    names = set(dict(mapper.named_parameters()).keys())
    assert "blocks.0.layer.0.SelfAttention.q.weight" in names
    assert "blocks.0.layer.0.SelfAttention.k.weight" in names
    assert "blocks.0.layer.0.SelfAttention.v.weight" in names
    assert "blocks.0.layer.0.SelfAttention.o.weight" in names
    assert "blocks.0.layer.1.DenseReluDense.wi_0.weight" in names
    assert "blocks.0.layer.1.DenseReluDense.wi_1.weight" in names
    assert "blocks.0.layer.1.DenseReluDense.wo.weight" in names
    assert "channel_mapper.weight" in names and "channel_mapper.bias" in names


def test_load_weights_shape_mismatch_raises() -> None:
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=0, sdxl_channels=8)
    bad = {"channel_mapper.weight": torch.zeros(8, 999)}
    with pytest.raises(ValueError, match="Shape mismatch"):
        mapper.load_weights(bad.items())


def test_load_weights_ignores_unknown_keys() -> None:
    cfg = _tiny_t5_config()
    mapper = T5EncoderBlockByT5Mapper(cfg, num_layers=0, sdxl_channels=8)
    loaded = mapper.load_weights({"not.a.real.param": torch.zeros(3)}.items())
    assert loaded == set()


# ---------------------------------------------------------------------------
# Snapshot-gated full encoder build
# ---------------------------------------------------------------------------


def _find_byt5_dir() -> str | None:
    override = os.environ.get("MING_FLASH_OMNI_DIR")
    candidates = []
    if override:
        candidates.append(Path(override) / "byt5")
    candidates.append(Path("/dev/shm/ming-hybrid") / "byt5")
    for c in candidates:
        # Require the actual weight dirs, not just the config jsons — some
        # snapshots ship byt5.json + tokenizer stubs without the trained
        # backbone, which would fail mid-load rather than skip cleanly.
        if (c / "byt5.json").exists() and (c / "byt5_model" / "byt5_model.pt").exists():
            return str(c)
    return None


@pytest.mark.skipif(_find_byt5_dir() is None, reason="Need Ming byt5 checkpoint dir.")
def test_byt5_encoder_builds_and_runs_from_checkpoint() -> None:
    byt5_dir = _find_byt5_dir()
    enc = MingByT5Encoder.from_checkpoint(
        Path(byt5_dir), device=torch.device("cpu"), dtype=torch.float32
    )
    feats = enc.forward(["hello world", "draw a red mug"])
    assert feats.dim() == 3 and feats.shape[0] == 2
    assert feats.shape[1] == enc.max_length
