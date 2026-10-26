"""The code predictor's captured depth loop, with each residual add carried into the next norm, computes what the
plain pre-norm loop computes. On the CPU the fused norm falls back to an add and a norm, so the CPU cases pin the
residual chaining, with the Triton attention helpers swapped for torch equivalents. The GPU case runs the fused
kernel in bf16 at the real sizes."""
import copy
import functools

import pytest
import torch
import torch.nn.functional as F

from mstar.model.qwen3_omni.components import talker as talker_module
from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor
from mstar.model.qwen3_omni.config import CodePredictorConfig, Qwen3OmniModelConfig, TalkerConfig

GROUPS = 4


def _qk_norm_rope(x, w_norm, pos, eps, rope_theta):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w_norm.float()
    half = x.shape[-1] // 2
    inv_freq = rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angle = pos.float()[:, None, None] * inv_freq
    cos, sin = angle.cos(), angle.sin()
    first, second = x32[..., :half], x32[..., half:]
    x.copy_(torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1))
    return x


def _decode_attn(q, k_cache, v_cache, cache_len, sm_scale=None):
    k, v = k_cache[:, :cache_len], v_cache[:, :cache_len]
    rep = q.shape[2] // k.shape[2]
    k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=sm_scale)
    return out.transpose(1, 2)


def _pre_norm_depth(cp, hidden, position_ids, kv_cache, cache_pos, qk_norm_rope=_qk_norm_rope, attend=_decode_attn):
    """The loop with explicit residual adds, as it was written before the adds moved into the norms."""
    bs, seq_len, _ = hidden.shape
    flat_pos = position_ids.reshape(-1)
    for layer_idx, layer in enumerate(cp.model.layers):
        attn = layer.self_attn
        residual = hidden
        qkv = F.linear(layer.input_layernorm(hidden), attn.qkv_proj.weight)
        q_size, kv_size = attn.num_heads * attn.head_dim, attn.num_kv_heads * attn.head_dim
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q = qk_norm_rope(q.reshape(bs * seq_len, attn.num_heads, attn.head_dim).clone(), attn.q_norm.weight,
                         flat_pos, attn.q_norm.variance_epsilon, float(attn.rope_theta))
        k = qk_norm_rope(k.reshape(bs * seq_len, attn.num_kv_heads, attn.head_dim).clone(), attn.k_norm.weight,
                         flat_pos, attn.k_norm.variance_epsilon, float(attn.rope_theta))
        kv_cache[layer_idx, :, 0, cache_pos:cache_pos + seq_len] = k.view(bs, seq_len, attn.num_kv_heads, -1)
        kv_cache[layer_idx, :, 1, cache_pos:cache_pos + seq_len] = v.reshape(bs, seq_len, attn.num_kv_heads, -1)
        out = attend(q.view(bs, seq_len, attn.num_heads, -1), kv_cache[layer_idx, :, 0],
                     kv_cache[layer_idx, :, 1], cache_pos + seq_len)
        hidden = residual + attn.o_proj(out.reshape(bs, seq_len, -1))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return cp.model.norm(hidden)


@pytest.fixture
def code_predictor(monkeypatch):
    monkeypatch.setattr(talker_module, "fused_qk_norm_rope", _qk_norm_rope)
    monkeypatch.setattr(talker_module, "decode_attn_nhd", _decode_attn)
    cfg = Qwen3OmniModelConfig(
        talker=TalkerConfig(num_code_groups=GROUPS),
        code_predictor=CodePredictorConfig(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                                           num_attention_heads=4, num_key_value_heads=2, head_dim=8),
    )
    return _randomized(Qwen3OmniCodePredictor(cfg))


def _randomized(cp):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in cp.named_parameters():
            noise = torch.randn(p.shape, generator=g)
            p.copy_(1.0 + 0.1 * noise if "norm" in name else noise / p.shape[-1] ** 0.5)
    return cp.eval()


def _fused_depth(cp, hidden, position_ids, kv_cache, cache_pos):
    return cp.forward_depth_unrolled(hidden, position_ids, kv_cache, cache_pos=cache_pos)


def _run(depth, cp, first, embeds):
    """The Talker's sequence: the Talker's hidden at depth 0, then one codec embedding per later group."""
    cpc = cp.model.layers[0].self_attn
    bs = first.shape[0]
    kv = torch.zeros(len(cp.model.layers), bs, 2, len(embeds) + 1, cpc.num_kv_heads, cpc.head_dim,
                     dtype=first.dtype, device=first.device)
    pos = torch.zeros(bs, 1, dtype=torch.long, device=first.device)
    outs = [depth(cp, first.clone().unsqueeze(1), pos, kv, 0)]
    for i, embed in enumerate(embeds, start=1):
        pos += 1
        outs.append(depth(cp, embed.clone().unsqueeze(1), pos, kv, i))
    return torch.stack(outs), kv


@pytest.mark.parametrize("bs", [1, 3])
def test_fused_residual_depth_loop_matches_the_pre_norm_loop(code_predictor, bs):
    g = torch.Generator().manual_seed(1)
    first = torch.randn(bs, 32, generator=g)
    embeds = torch.randn(GROUPS - 1, bs, 32, generator=g)

    with torch.no_grad():
        got, got_kv = _run(_fused_depth, code_predictor, first, embeds)
        want, want_kv = _run(_pre_norm_depth, code_predictor, first, embeds)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_kv, want_kv, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused add + norm kernel runs on a GPU")
@pytest.mark.parametrize("bs", [1, 8, 32])
def test_on_a_gpu_the_fused_bf16_loop_is_as_close_to_fp32_as_the_unfused_one(bs):
    """Real sizes, bf16, the GPU kernels. Both loops round each residual to bf16 once, and the fused kernel also
    norms from the fp32 sum, so its outputs should be no further from the fp32 loop on the same bf16 weights and
    inputs than those of the loop with separate adds."""
    cfg = Qwen3OmniModelConfig()
    hidden = cfg.code_predictor.hidden_size
    cp = _randomized(Qwen3OmniCodePredictor(cfg)).to(torch.bfloat16)
    g = torch.Generator().manual_seed(1)
    first = torch.randn(bs, hidden, generator=g).to(torch.bfloat16)
    embeds = torch.randn(cfg.num_code_groups - 1, bs, hidden, generator=g).to(torch.bfloat16)
    unfused_depth = functools.partial(
        _pre_norm_depth, qk_norm_rope=talker_module.fused_qk_norm_rope, attend=talker_module.decode_attn_nhd)

    with torch.no_grad():
        exact, _ = _run(_pre_norm_depth, copy.deepcopy(cp).float(), first.float(), embeds.float())
        cp = cp.cuda()
        fused, _ = _run(_fused_depth, cp, first.cuda(), embeds.cuda())
        unfused, _ = _run(unfused_depth, cp, first.cuda(), embeds.cuda())

    err_fused = (fused.float().cpu() - exact).abs()
    err_unfused = (unfused.float().cpu() - exact).abs()
    print(f"bs {bs}: |error| vs fp32, fused mean {err_fused.mean():.3e} max {err_fused.max():.3e}, "
          f"separate adds mean {err_unfused.mean():.3e} max {err_unfused.max():.3e}, "
          f"mean |fp32 output| {exact.abs().mean():.3e}")
    assert err_fused.mean() <= 1.1 * err_unfused.mean()
    assert err_fused.max() <= 2 * err_unfused.max()
