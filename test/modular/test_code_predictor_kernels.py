"""The code predictor's kernels (Qwen3-Omni's and Qwen3-TTS's depth loops)
against the op sequences they replace (GPU only), and ``cp_linear``'s tiles.

* ``cp_attend``: q/k RMSNorm + RoPE, the dense-KV write and causal attention
  over the depth cache, against the same math in fp32.
* ``cp_embed_accumulate``: the index_select + add, and the next input (projected
  or not) with its norm partials.
* ``ResidualCodePredictor``'s depth loop, which calls them with its own config:
  against its layers run unfused.
"""
from __future__ import annotations

import math

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _norm_rope(x: torch.Tensor, w: torch.Tensor, pos: torch.Tensor, eps: float, theta: float) -> torch.Tensor:
    """RMSNorm over the head dim, then rotate-half RoPE at ``pos``, in fp32."""
    d = x.shape[-1]
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    inv_freq = torch.exp(-math.log(theta) * torch.arange(d // 2, device=x.device).float() * 2.0 / d)
    angle = pos.float()[:, None, None] * inv_freq
    first, second = xf[..., : d // 2], xf[..., d // 2:]
    return torch.cat([first * angle.cos() - second * angle.sin(), second * angle.cos() + first * angle.sin()], -1)


@needs_cuda
@pytest.mark.parametrize("bs,seq_len,cache_pos", [(1, 1, 0), (5, 1, 7), (32, 1, 15), (64, 1, 3), (3, 2, 0), (5, 2, 4)])
def test_cp_attend_matches_the_unfused_ops(bs, seq_len, cache_pos):
    """q/k RMSNorm + RoPE, the step's k/v written into the dense cache, and
    causal attention over the depth positions so far, against the same math
    in fp32 (q and k rounded to the cache dtype, as the unfused path stores
    them)."""
    from mstar.model.components.code_predictor_kernels import cp_attend

    torch.manual_seed(0)
    hq, hkv, d, layers, max_bs, max_s = 16, 8, 128, 5, 64, 16
    dev, dt = "cuda", torch.bfloat16
    qkv = torch.randn(bs, seq_len, (hq + 2 * hkv) * d, device=dev, dtype=dt)
    wq = torch.randn(d, device=dev, dtype=dt)
    wk = torch.randn(d, device=dev, dtype=dt)
    cache = torch.randn(layers, max_bs, 2, max_s, hkv, d, device=dev, dtype=dt)
    ref_cache = cache.clone()
    # a token's RoPE position is its cache slot
    flat_pos = (cache_pos + torch.arange(seq_len, device=dev)).repeat(bs)

    q, k, v = qkv.split((hq * d, hkv * d, hkv * d), dim=-1)
    q = _norm_rope(q.reshape(bs * seq_len, hq, d), wq, flat_pos, 1e-6, 1e6).to(dt).view(bs, seq_len, hq, d)
    k = _norm_rope(k.reshape(bs * seq_len, hkv, d), wk, flat_pos, 1e-5, 1e6)
    view = ref_cache[:, :bs]
    view[2, :, 0, cache_pos:cache_pos + seq_len] = k.view(bs, seq_len, hkv, d).to(dt)
    view[2, :, 1, cache_pos:cache_pos + seq_len] = v.view(bs, seq_len, hkv, d)

    out = cp_attend(
        qkv.view(bs * seq_len, -1), wq, wk, cache[:, :bs][2], cache_pos, seq_len,
        1e-6, 1e-5, 1e6, hq, hkv, d,
    ).view(bs, seq_len, hq, d)
    # the kernel rounds k once to bf16 from the same fp32 arithmetic
    torch.testing.assert_close(cache.float(), ref_cache.float(), rtol=1.6e-2, atol=1e-2)
    kv = ref_cache[2, :bs].float()
    for s in range(seq_len):
        n = cache_pos + s + 1
        kk = kv[:, 0, :n].repeat_interleave(hq // hkv, dim=2)  # (bs, n, hq, d)
        vv = kv[:, 1, :n].repeat_interleave(hq // hkv, dim=2)
        w = torch.softmax(torch.einsum("bhd,bnhd->bhn", q[:, s].float(), kk) / d ** 0.5, -1)
        ref = torch.einsum("bhn,bnhd->bhd", w, vv)
        torch.testing.assert_close(out[:, s].float(), ref, rtol=2e-2, atol=2e-2)


@needs_cuda
@pytest.mark.parametrize("bs,projected,init", [(1, False, False), (8, False, True), (32, True, False), (5, True, True)])
def test_cp_embed_accumulate_matches_the_unfused_ops(bs, projected, init):
    """The index_select + add (or, with ``init``, the sum's first term), the
    code recorded, and the next input: the embedding, or its row of the
    projected table, with each 256-wide chunk's row sum of squares for the
    first layer's norm."""
    from mstar.model.components.code_predictor_kernels import EMBED_BLOCK, cp_embed_accumulate, depth_partials

    torch.manual_seed(2)
    weight = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    table = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16) if projected else None
    tokens = torch.randint(0, 2048, (bs,), device="cuda")
    emb_sum = torch.randn(bs, 1024, device="cuda")
    codes = torch.randint(0, 2048, (bs, 16), device="cuda")
    ref_sum, ref_codes = emb_sum.clone(), codes.clone()

    ref_codes[:, 5] = tokens
    ref_emb = torch.nn.functional.embedding(tokens, weight)
    ref_sum = ref_emb.float() if init else ref_sum.add_(ref_emb)

    partials = depth_partials(torch.device("cuda"))
    emb, n_part = cp_embed_accumulate(tokens, weight, emb_sum, codes, 5, inputs=table, init=init, partials=partials)
    assert torch.equal(emb, ref_emb if table is None else torch.nn.functional.embedding(tokens, table))
    assert torch.equal(emb_sum, ref_sum)
    assert torch.equal(codes, ref_codes)
    # one partial per chunk of the returned input, [chunk, row]
    assert n_part == emb.shape[1] // EMBED_BLOCK
    want_partials = emb.float().view(bs, n_part, EMBED_BLOCK).pow(2).sum(-1).T
    torch.testing.assert_close(partials[:n_part, :bs], want_partials, rtol=1e-5, atol=1e-5)  # summation order


@needs_cuda
@pytest.mark.parametrize("rows", [1, 2, 16, 37, 64])
def test_cp_linear_matches_the_unfused_ops(rows):
    """Each ``cp_linear`` mode against its op sequence in fp32: the folded
    RMSNorm (row sums of squares from x, and from a producer's partials), the
    residual add, silu-gate and bias epilogues."""
    from mstar.model.components.code_predictor_kernels import STREAM_TILE, cp_linear, depth_partials

    torch.manual_seed(3)
    dev, dt, h = "cuda", torch.bfloat16, 1024
    x = torch.randn(rows, h, device=dev, dtype=dt)
    norm_w = torch.rand(h, device=dev, dtype=dt) + 0.5
    w = torch.randn(512, h, device=dev, dtype=dt) * 0.05
    up = torch.randn(512, h, device=dev, dtype=dt) * 0.05
    w_res = torch.randn(h, 2048, device=dev, dtype=dt) * 0.05
    bias = torch.randn(h, device=dev, dtype=dt)
    a = torch.randn(rows, 2048, device=dev, dtype=dt)
    partials = depth_partials(torch.device(dev))
    folded = (w.float() * norm_w.float()).to(dt)
    xn = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)

    def close(got, want):
        torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2 * want.abs().max().item())

    close(cp_linear(x, folded, norm_eps=1e-6), xn @ folded.float().t())
    up_f = (up.float() * norm_w.float()).to(dt)
    close(cp_linear(x, folded, up=up_f, norm_eps=1e-6),
          torch.nn.functional.silu(xn @ folded.float().t()) * (xn @ up_f.float().t()))
    # a residual add records the stream's partial sums; the next norm reads them
    stream = x.clone()
    cp_linear(a, w_res, out=stream, residual=True, partials=partials)
    want = x.float() + a.float() @ w_res.float().t()
    close(stream, want)
    sn = stream.float() * torch.rsqrt(stream.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    close(cp_linear(stream, folded, norm_eps=1e-6, partials=partials, n_part=h // STREAM_TILE), sn @ folded.float().t())
    out = cp_linear(a, w_res, bias=bias, partials=partials)
    close(out, a.float() @ w_res.float().t() + bias.float())
    on = out.float() * torch.rsqrt(out.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    close(cp_linear(out, folded, norm_eps=1e-6, partials=partials, n_part=h // STREAM_TILE), on @ folded.float().t())


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


@needs_cuda
@pytest.mark.parametrize("width", [2048, 1024])
def test_code_predictor_depth_loop_matches_its_layers_unfused(width):
    """The depth loop drives the kernels with the predictor's own head counts,
    eps and RoPE theta, from a Talker of ``width``: 2048 through an input
    projection (Qwen3-TTS 1.7B), 1024 without one (Qwen3-Omni, Qwen3-TTS 0.6B),
    where the inputs are the raw embeddings and their norm partials. A frame's
    groups, teacher-forced, through the fused projections and ``cp_attend``
    against the same layers over the whole depth sequence at once, norms, RoPE
    and causal attention in fp32, compared as each group's logits."""
    from mstar.model.components.code_predictor_kernels import ResidualCodePredictor
    from mstar.model.qwen3_tts.config import Qwen3TTSCodePredictorConfig

    torch.manual_seed(6)
    cp = Qwen3TTSCodePredictorConfig(num_hidden_layers=2)
    bs, groups = 3, cp.num_code_groups
    predictor = ResidualCodePredictor(cp, width, max_batch_size=4).to("cuda", torch.bfloat16).eval()
    for name, param in predictor.named_parameters():
        if name.endswith("norm.weight"):
            param.data.uniform_(0.5, 1.5)  # a skipped or swapped norm weight shows
        else:
            param.data.normal_(std=0.02)
    talker_table = torch.randn(cp.vocab_size, width, device="cuda", dtype=torch.bfloat16)
    predictor.consolidate_stacked_weights(talker_table)
    hq, hkv, d = cp.num_attention_heads, cp.num_key_value_heads, cp.head_dim
    last_hidden = torch.randn(bs, width, device="cuda", dtype=torch.bfloat16)
    codes = torch.randint(0, cp.vocab_size, (bs, groups), device="cuda")
    got_logits = []

    def teacher(logits):
        got_logits.append(logits.float())
        return codes[:, len(got_logits)].contiguous()

    with torch.no_grad():
        got_codes, embed_sum = predictor.depth_loop(last_hidden, codes[:, 0].contiguous(), talker_table, teacher)

        # position 0 holds the Talker's hidden state, position g >= 1 the embedding
        # of group g - 1's code, and predicts group g
        layer0_embed = talker_table[codes[:, 0]]
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
        heads = torch.stack([head.weight for head in predictor.lm_head]).float()
        want_logits = torch.stack([h[:, g] @ heads[g - 1].t() for g in range(1, groups)])
        # summed in fp32 in group order, rounded once to the embeddings' dtype
        want_sum = layer0_embed.float()
        for embed in embeds:
            want_sum = want_sum + embed.float()

    assert torch.equal(got_codes, codes)
    torch.testing.assert_close(
        torch.stack(got_logits), want_logits, rtol=5e-2, atol=5e-2 * want_logits.abs().max().item(),
    )
    assert torch.equal(embed_sum, want_sum.bfloat16())
