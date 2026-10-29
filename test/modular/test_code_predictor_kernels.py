"""The code predictor's kernels (Qwen3-Omni's and Qwen3-TTS's depth loops)
against the op sequences they replace (GPU only).

* ``cp_qk_norm_rope_kv_write``: q/k RMSNorm + RoPE and the dense-KV write,
  against the same math in fp32 rounded to the cache dtype.
* ``cp_embed_accumulate`` and ``cp_attention``: the index_select + add, and
  causal attention over the depth cache.
* Qwen3-TTS's depth loop, which calls them with its own config: against its
  layers run unfused.
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
    pos = torch.randint(0, 16, (bs, seq_len), device=dev)
    cache = torch.randn(layers, max_bs, 2, max_s, hkv, d, device=dev, dtype=dt)
    ref_cache, ref_qkv = cache.clone(), qkv.clone()
    flat_pos = pos.reshape(-1)

    q, k, v = ref_qkv.split((hq * d, hkv * d, hkv * d), dim=-1)
    q = _norm_rope(q.reshape(bs * seq_len, hq, d), wq, flat_pos, 1e-6, 1e6)
    k = _norm_rope(k.reshape(bs * seq_len, hkv, d), wk, flat_pos, 1e-5, 1e6)
    view = ref_cache[:, :bs]
    view[2, :, 0, cache_pos:cache_pos + seq_len] = k.view(bs, seq_len, hkv, d).to(dt)
    view[2, :, 1, cache_pos:cache_pos + seq_len] = v.view(bs, seq_len, hkv, d)

    cp_qk_norm_rope_kv_write(
        qkv.view(bs * seq_len, -1), wq, wk, flat_pos, cache[:, :bs][2], cache_pos, seq_len,
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


def test_qwen3_tts_depth_loop_matches_its_layers_unfused():
    """Qwen3-TTS's depth loop drives the kernels with its own head counts, eps
    and RoPE theta: a frame's 16 depth steps against the same layers over the
    whole frame at once, norms, RoPE and causal attention in fp32."""
    from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor
    from mstar.model.qwen3_tts.config import (
        Qwen3TTSCodePredictorConfig,
        Qwen3TTSModelConfig,
        Qwen3TTSTalkerConfig,
    )

    torch.manual_seed(6)
    cp = Qwen3TTSCodePredictorConfig(num_hidden_layers=2)
    config = Qwen3TTSModelConfig(talker=Qwen3TTSTalkerConfig(code_predictor=cp))
    predictor = Qwen3TTSCodePredictor(config).to("cuda", torch.bfloat16).eval()
    for name, param in predictor.named_parameters():
        if name.endswith("norm.weight"):
            param.data.uniform_(0.5, 1.5)  # a skipped or swapped norm weight shows
        else:
            param.data.normal_(std=0.02)
    hq, hkv, d = cp.num_attention_heads, cp.num_key_value_heads, cp.head_dim
    bs, depth = 3, config.num_code_groups
    x = torch.randn(bs, depth, config.talker.hidden_size, device="cuda", dtype=torch.bfloat16)

    cache = torch.zeros(cp.num_hidden_layers, bs, 2, depth, hkv, d, device="cuda", dtype=torch.bfloat16)
    pos = torch.zeros(bs, 1, dtype=torch.int32, device="cuda")
    got = []
    with torch.no_grad():
        for s in range(depth):
            pos.fill_(s)
            predictor.depth.bind_step(cache, pos, cache_pos=s)
            got.append(predictor.forward_depth_unrolled(x[:, s:s + 1]))

        # the projection is the identity at these widths; linears run in bf16
        h = x.float()
        flat_pos = torch.arange(depth, device="cuda").repeat(bs)
        for layer in predictor.model.layers:
            attn = layer.self_attn
            qkv = attn.qkv_proj(_rms(h, layer.input_layernorm.weight, cp.rms_norm_eps).bfloat16()).float()
            q, k, v = qkv.split((hq * d, hkv * d, hkv * d), dim=-1)
            q = _norm_rope(q.reshape(-1, hq, d), attn.q_norm.weight, flat_pos, cp.rms_norm_eps, cp.rope_theta)
            k = _norm_rope(k.reshape(-1, hkv, d), attn.k_norm.weight, flat_pos, cp.rms_norm_eps, cp.rope_theta)
            q = q.view(bs, depth, hq, d).transpose(1, 2)
            k = k.view(bs, depth, hkv, d).repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            v = v.view(bs, depth, hkv, d).repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + attn.o_proj(out.transpose(1, 2).reshape(bs, depth, -1).bfloat16()).float()
            h = h + layer.mlp(_rms(h, layer.post_attention_layernorm.weight, cp.rms_norm_eps).bfloat16()).float()
        want = _rms(h, predictor.model.norm.weight, cp.rms_norm_eps)

    torch.testing.assert_close(torch.cat(got, dim=1).float(), want, rtol=5e-2, atol=1e-1)
