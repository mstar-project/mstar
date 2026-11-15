"""The compiled DiT's fused q/k norm + rope kernel against the eager ops it replaces."""

import pytest
import torch


def test_a_head_count_the_block_does_not_divide_is_refused():
    """A program norms ``HB`` heads with one weight, so they must be all q or all k."""
    pytest.importorskip("triton")
    from mstar.model.minimax_h3.components.qk_norm_rope import qk_norm_rope

    rows, heads, d = 4, 28, 128
    qkv = torch.zeros(rows, 3 * heads * d, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 8 heads"):
        qk_norm_rope(qkv, torch.ones(2, d), torch.ones(rows, d), torch.zeros(rows, d), 1e-5, heads, 96)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a Triton kernel")
def test_the_kernel_matches_the_eager_norms_and_rope():
    from mstar.model.minimax_h3.components.layout import H3Layout, position_ids
    from mstar.model.minimax_h3.components.qk_norm_rope import qk_norm_rope
    from mstar.model.minimax_h3.components.transformer import apply_rope, rope_inv_freq, rope_table
    from mstar.model.minimax_h3.config import H3TransformerConfig

    cfg, heads = H3TransformerConfig(), 8
    d = cfg.attention_head_dim
    pos = position_ids(H3Layout(text_len=7, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4))
    rope = rope_table(pos.cuda(), rope_inv_freq(cfg), torch.bfloat16, d)
    g = torch.Generator(device="cuda").manual_seed(0)
    qkv = (3 * torch.randn(pos.shape[0], 3 * heads * d, device="cuda", generator=g)).to(torch.bfloat16)
    norms = [torch.nn.RMSNorm(d, eps=cfg.qk_norm_eps, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    for norm in norms:
        torch.nn.init.normal_(norm.weight, 1.0, 0.1, generator=g)
    q, k = (norms[i](qkv[:, i * heads * d:(i + 1) * heads * d].unflatten(-1, (heads, d)))[None] for i in range(2))
    want = torch.cat([apply_rope(q, rope), apply_rope(k, rope)], dim=2)[0]
    weights = torch.stack([norm.weight for norm in norms]).detach()
    got = qk_norm_rope(qkv, weights, rope.cos, rope.sin, cfg.qk_norm_eps, heads, rope.rotary_dim)
    # the norm's fp32 sum runs in another order than torch's kernel: at most a few values one bf16 step apart
    diff = (got.float() - want.float()).abs()
    assert (diff > 0).float().mean() < 0.01 and torch.allclose(got.float(), want.float(), rtol=1e-2, atol=1e-2)
