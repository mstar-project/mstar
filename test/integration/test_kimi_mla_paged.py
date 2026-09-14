"""Naive (materialized) MLA over the real paged cache.

The reduced-capability fallback: the latent is up-projected to full per-head
K/V and served by the ordinary paged backend, so q/k/v are padded to
``padded_head_dim`` and the DeepSeek scale is folded into q. This checks that
the padding and the scale compensation cancel out exactly against an unpadded
reference.
"""

import pytest
import torch
from kimi_harness import (
    DEVICE,
    build_resources,
    cleanup,
    ingest,
    paged_specs,
    rope_spec,
    step,
)
from kimi_reference import ref_deepseek_mla

from mstar.model.kimi_k2_7.components.attention import KimiMLAAttention
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real FlashInfer paged MLA needs a GPU",
)


def _build_attention(cfg, dtype):
    attn = KimiMLAAttention(cfg).to(device=DEVICE, dtype=dtype)
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    return attn


def test_paged_mla_matches_deepseek_sdpa():
    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    cfg.mla_absorb = False
    assert cfg.qk_head_dim == 24 and cfg.padded_head_dim == 64  # the mitigation
    dtype = torch.bfloat16
    layer = _build_attention(cfg, dtype)

    seq_len = 6
    h = torch.randn(seq_len, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.1

    specs = paged_specs(
        num_layers=1,
        num_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.padded_head_dim,
    )
    resources = build_resources(
        specs + [rope_spec(cfg)], entity_id="kimi_mla_paged_test",
    )
    try:
        ingest(resources, "r0")
        layer.bind_resources(resources)
        with step(resources, {"r0": seq_len}):
            layer.attend.bind_step("main")
            layer.attend.set_layer_idx(0)
            # the layer takes its position ids off the rope resource now
            pos = layer.position_ids
            assert pos.tolist() == list(range(seq_len))
            with torch.no_grad():
                got = layer(h)
        torch.cuda.synchronize()
    finally:
        cleanup(resources)

    expected = ref_deepseek_mla(layer, cfg, h, pos)
    assert got.shape == (seq_len, cfg.hidden_size)
    # Any residual after exact scale compensation is bf16 FlashInfer rounding.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
