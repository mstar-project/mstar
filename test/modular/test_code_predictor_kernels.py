"""The code predictor's kernels (Qwen3-Omni's and Qwen3-TTS's depth loops)
against the op sequences they replace (GPU only).

* ``cp_qk_norm_rope_kv_write``: q/k RMSNorm + RoPE and the dense-KV write,
  against the same math in fp32 rounded to the cache dtype.
* ``cp_embed_accumulate`` and ``cp_attention``: the index_select + add, and
  causal attention over the depth cache.
* ``ResidualCodePredictor``'s depth loop, which calls them with its own config:
  against its layers run unfused.
"""
from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _norm_rope(x: torch.Tensor, w: torch.Tensor, pos: torch.Tensor, eps: float, theta: float) -> torch.Tensor:
    """RMSNorm over the head dim, then rotate-half RoPE at ``pos``, in fp32."""
    d = x.shape[-1]
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    inv_freq = torch.exp(-math.log(theta) * torch.arange(d // 2, device=x.device).float() * 2.0 / d)
    angle = pos.float()[:, None, None] * inv_freq
    first, second = xf[..., : d // 2], xf[..., d // 2:]
    return torch.cat([first * angle.cos() - second * angle.sin(), second * angle.cos() + first * angle.sin()], -1)


@pytest.mark.parametrize("bs,seq_len,cache_pos", [(1, 1, 0), (5, 1, 7), (32, 1, 15), (3, 2, 4)])
def test_cp_prologue_matches_the_unfused_ops(bs, seq_len, cache_pos):
    from mstar.model.components.code_predictor_kernels import cp_qk_norm_rope_kv_write

    torch.manual_seed(0)
    hq, hkv, d, layers, max_bs, max_s = 16, 8, 128, 5, 32, 16
    dev, dt = "cuda", torch.bfloat16
    qkv = torch.randn(bs, seq_len, (hq + 2 * hkv) * d, device=dev, dtype=dt)
    wq = torch.randn(d, device=dev, dtype=dt)
    wk = torch.randn(d, device=dev, dtype=dt)
    cache = torch.randn(layers, max_bs, 2, max_s, hkv, d, device=dev, dtype=dt)
    ref_cache, ref_qkv = cache.clone(), qkv.clone()
    # a token's RoPE position is its cache slot
    flat_pos = (cache_pos + torch.arange(seq_len, device=dev)).repeat(bs)

    q, k, v = ref_qkv.split((hq * d, hkv * d, hkv * d), dim=-1)
    q = _norm_rope(q.reshape(bs * seq_len, hq, d), wq, flat_pos, 1e-6, 1e6)
    k = _norm_rope(k.reshape(bs * seq_len, hkv, d), wk, flat_pos, 1e-5, 1e6)
    view = ref_cache[:, :bs]
    view[2, :, 0, cache_pos:cache_pos + seq_len] = k.view(bs, seq_len, hkv, d).to(dt)
    view[2, :, 1, cache_pos:cache_pos + seq_len] = v.view(bs, seq_len, hkv, d)

    cp_qk_norm_rope_kv_write(
        qkv.view(bs * seq_len, -1), wq, wk, cache[:, :bs][2], cache_pos, seq_len,
        1e-6, 1e-5, 1e6, hq, hkv, d,
    )
    # the kernel rounds once to bf16 from the same fp32 arithmetic
    torch.testing.assert_close(qkv[..., :hq * d].float(), q.view(bs, seq_len, -1), rtol=1.6e-2, atol=1e-2)
    torch.testing.assert_close(cache.float(), ref_cache.float(), rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("bs", [1, 8, 32])
def test_cp_embed_accumulate_matches_the_unfused_ops(bs):
    from mstar.model.components.code_predictor_kernels import cp_embed_accumulate

    torch.manual_seed(2)
    weight = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    tokens = torch.randint(0, 2048, (bs,), device="cuda")
    emb_sum = torch.randn(bs, 1024, device="cuda")
    codes = torch.randint(0, 2048, (bs, 16), device="cuda")
    ref_sum, ref_codes = emb_sum.clone(), codes.clone()

    ref_codes[:, 5] = tokens
    ref_emb = torch.nn.functional.embedding(tokens, weight)
    ref_sum.add_(ref_emb)

    emb = cp_embed_accumulate(tokens, weight, emb_sum, codes, 5)
    assert torch.equal(emb, ref_emb)
    assert torch.equal(emb_sum, ref_sum)
    assert torch.equal(codes, ref_codes)



@pytest.mark.parametrize("bs,seq,cache_pos", [(1, 1, 0), (4, 2, 0), (32, 1, 7), (3, 1, 15), (5, 2, 3)])
def test_cp_attention_is_causal_attention_over_the_depth_cache(bs, seq, cache_pos):
    from mstar.model.components.code_predictor_kernels import cp_attention

    torch.manual_seed(4)
    hq, hkv, d, depth = 16, 8, 128, 16
    cache = torch.randn(bs, 2, depth, hkv, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(bs, seq, hq, d, device="cuda", dtype=torch.bfloat16)
    out = cp_attention(q, cache[:, 0], cache[:, 1], cache_pos)
    for s in range(seq):
        n = cache_pos + s + 1
        k = cache[:, 0, :n].float().repeat_interleave(hq // hkv, dim=2)  # (bs, n, hq, d)
        v = cache[:, 1, :n].float().repeat_interleave(hq // hkv, dim=2)
        w = torch.softmax(torch.einsum("bhd,bnhd->bhn", q[:, s].float(), k) / d ** 0.5, -1)
        ref = torch.einsum("bhn,bnhd->bhd", w, v)
        torch.testing.assert_close(out[:, s].float(), ref, rtol=2e-2, atol=2e-2)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def test_code_predictor_depth_loop_matches_its_layers_unfused():
    """The depth loop drives the kernels with the predictor's own head counts,
    eps and RoPE theta, through an input projection (Qwen3-TTS 1.7B's widths):
    a frame's groups, teacher-forced, against the same layers over the whole
    depth sequence at once, norms, RoPE and causal attention in fp32."""
    from mstar.model.components.code_predictor_kernels import ResidualCodePredictor
    from mstar.model.qwen3_tts.config import Qwen3TTSCodePredictorConfig

    torch.manual_seed(6)
    cp = Qwen3TTSCodePredictorConfig(num_hidden_layers=2)
    width, bs, groups = 2048, 3, cp.num_code_groups
    predictor = ResidualCodePredictor(cp, width, max_batch_size=4).to("cuda", torch.bfloat16).eval()
    for name, param in predictor.named_parameters():
        if name.endswith("norm.weight"):
            param.data.uniform_(0.5, 1.5)  # a skipped or swapped norm weight shows
        else:
            param.data.normal_(std=0.02)
    predictor.consolidate_stacked_weights()
    hq, hkv, d = cp.num_attention_heads, cp.num_key_value_heads, cp.head_dim
    last_hidden = torch.randn(bs, width, device="cuda", dtype=torch.bfloat16)
    layer0_embed = torch.randn(bs, width, device="cuda", dtype=torch.bfloat16)
    codes = torch.randint(0, cp.vocab_size, (bs, groups), device="cuda")
    got_logits = []

    def teacher(logits):
        got_logits.append(logits.float())
        return codes[:, len(got_logits)].contiguous()

    with torch.no_grad():
        got_codes, embed_sum = predictor.depth_loop(last_hidden, codes[:, 0], layer0_embed, teacher)

        # position 0 holds the Talker's hidden state, position g >= 1 the embedding
        # of group g - 1's code, and predicts group g
        embeds = [predictor.model.codec_embedding[g - 1](codes[:, g]) for g in range(1, groups)]
        x = torch.stack([last_hidden, layer0_embed, *embeds[:-1]], dim=1)
        h = predictor.small_to_mtp_projection(x).float()
        flat_pos = torch.arange(groups, device="cuda").repeat(bs)
        for layer in predictor.model.layers:
            attn = layer.self_attn
            qkv = attn.qkv_proj(_rms(h, layer.input_layernorm.weight, cp.rms_norm_eps).bfloat16()).float()
            q, k, v = qkv.split((hq * d, hkv * d, hkv * d), dim=-1)
            q = _norm_rope(q.reshape(-1, hq, d), attn.q_norm.weight, flat_pos, cp.rms_norm_eps, cp.rope_theta)
            k = _norm_rope(k.reshape(-1, hkv, d), attn.k_norm.weight, flat_pos, cp.rms_norm_eps, cp.rope_theta)
            q = q.view(bs, groups, hq, d).transpose(1, 2)
            k = k.view(bs, groups, hkv, d).repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            v = v.view(bs, groups, hkv, d).repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + attn.o_proj(out.transpose(1, 2).reshape(bs, groups, -1).bfloat16()).float()
            h = h + layer.mlp(_rms(h, layer.post_attention_layernorm.weight, cp.rms_norm_eps).bfloat16()).float()
        h = _rms(h, predictor.model.norm.weight, cp.rms_norm_eps)
        want_logits = [h[:, g] @ predictor.lm_head_weight[g - 1].float().t() for g in range(1, groups)]
        # summed in fp32 in group order, rounded once to the embeddings' dtype
        want_sum = layer0_embed.float()
        for embed in embeds:
            want_sum = want_sum + embed.float()

    assert torch.equal(got_codes, codes)
    torch.testing.assert_close(torch.stack(got_logits), torch.stack(want_logits), rtol=5e-2, atol=1e-1)
    assert torch.equal(embed_sum, want_sum.bfloat16())
