"""Qwen3-Omni's fused kernels against the op sequences they replace (GPU only).

* ``cp_qk_norm_rope_kv_write``: the code predictor's q/k norm + RoPE and its
  dense-KV write, bit-for-bit what two ``fused_qk_norm_rope`` calls and two
  strided copies produce.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize("bs,seq_len,cache_pos", [(1, 1, 0), (5, 1, 7), (32, 1, 15), (3, 2, 4)])
def test_cp_prologue_matches_the_unfused_ops(bs, seq_len, cache_pos):
    from mstar.model.qwen3_omni.components.cp_kernels import cp_qk_norm_rope_kv_write
    from mstar.utils.attention import fused_qk_norm_rope

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
    q = fused_qk_norm_rope(q.view(bs * seq_len, hq, d), wq, pos=flat_pos, eps=1e-6, rope_theta=1e6)
    k = fused_qk_norm_rope(k.view(bs * seq_len, hkv, d), wk, pos=flat_pos, eps=1e-5, rope_theta=1e6)
    view = ref_cache[:, :bs]
    view[2, :, 0, cache_pos:cache_pos + seq_len] = k.view(bs, seq_len, hkv, d)
    view[2, :, 1, cache_pos:cache_pos + seq_len] = v.view(bs, seq_len, hkv, d)

    cp_qk_norm_rope_kv_write(
        qkv.view(bs * seq_len, -1), wq, wk, flat_pos, cache[:, :bs][2], cache_pos, seq_len,
        1e-6, 1e-5, 1e6, hq, hkv, d,
    )
    torch.testing.assert_close(qkv[..., :hq * d], ref_qkv[..., :hq * d], rtol=0, atol=0)
    torch.testing.assert_close(cache, ref_cache, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [1, 8, 32])
def test_cp_embed_accumulate_matches_the_unfused_ops(bs):
    from mstar.model.qwen3_omni.components.cp_kernels import cp_embed_accumulate

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
    from mstar.model.qwen3_omni.components.cp_kernels import cp_attention

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
