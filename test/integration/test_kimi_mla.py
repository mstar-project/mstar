"""M3 golden tests for Kimi-K2.7 MLA attention (naive/materialized path).

Three goldens against independent references cited to vLLM:
  - YARN RoPE (KimiYarnRotaryEmbedding) vs a DeepseekScalingRotaryEmbedding-style
    forward_static,
  - the q/k/v assembly (projections + rope-on-slice + k_pe broadcast + v-pad +
    mscale^2 q-prescale) captured at ``run_attention`` via a mock cache handle, and
  - the full attention forward (+ causal attention + output slice + o_proj).

A ``_MockMLACache`` stands in for the paged cache: its ``run_attention`` does a
causal SDPA at the fixed ``1/sqrt(qk_head_dim)`` scale (what FlashInfer uses),
which is what lets us golden the MLA math without the paged engine. The real
FlashInfer path over a 192-dim cache is exercised at M4/M6.

Refs: vLLM ``models/deepseek_v2.py::DeepseekV2Attention`` (naive path) and
``rotary_embedding/deepseek_scaling_rope.py``.

Run:  pytest test/integration/test_kimi_mla.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.kimi_k2_7.components.attention import KimiMLAAttention
from mstar.model.kimi_k2_7.components.rope import (
    KimiYarnRotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    rotate_gptj,
    yarn_get_mscale,
)
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="M3 golden tests need a GPU (MLA RMSNorm uses a FlashInfer kernel)",
)

DEVICE = "cuda"


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

def _ref_rmsnorm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def _ref_yarn_rope(pos, q_pe, k_pe, rotary_dim, base, factor, max_pos,
                   beta_fast, beta_slow, mscale, mscale_all_dim):
    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, device=q_pe.device).float() / rotary_dim)
    ext, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, rotary_dim, base, max_pos)
    mask = (1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, torch.float).to(q_pe.device))
    inv_freq = interp * (1 - mask) + ext * mask
    amp = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
    freqs = torch.outer(pos.float(), inv_freq)
    cos = (freqs.cos() * amp).repeat_interleave(2, -1).unsqueeze(-2)
    sin = (freqs.sin() * amp).repeat_interleave(2, -1).unsqueeze(-2)
    qr = q_pe.float() * cos + rotate_gptj(q_pe.float()) * sin
    kr = k_pe.float() * cos + rotate_gptj(k_pe.float()) * sin
    return qr.to(q_pe.dtype), kr.to(k_pe.dtype)


class _MockMLACache:
    """Paged-cache stand-in: causal SDPA at 1/sqrt(head_dim)."""

    def __init__(self, head_dim: int):
        self.scale = head_dim ** -0.5
        self.captured: dict = {}

    def set_layer_idx(self, _i):  # noqa: D401
        pass

    def set_active_label(self, _l):
        pass

    def advance_seq_lens(self, *_a, **_k):
        pass

    def run_attention(self, q, k, v):
        self.captured = {"q": q.clone(), "k": k.clone(), "v": v.clone()}
        qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))  # (H,T,D)
        scores = torch.einsum("hqd,hkd->hqk", qt, kt) * self.scale
        num_tokens = q.shape[0]
        causal = torch.triu(
            torch.full((num_tokens, num_tokens), float("-inf"), device=q.device), diagonal=1)
        attn = (scores + causal).softmax(-1)
        return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


def _rope_kwargs(cfg):
    r = cfg.rope_scaling
    return dict(
        rotary_dim=cfg.qk_rope_head_dim, base=cfg.rope_theta, factor=r["factor"],
        max_pos=r["original_max_position_embeddings"],
        beta_fast=r.get("beta_fast", 32), beta_slow=r.get("beta_slow", 1),
        mscale=r.get("mscale", 1.0), mscale_all_dim=r.get("mscale_all_dim", 0.0),
    )


def _ref_mla(attn: KimiMLAAttention, cfg, h, pos, scale, boost):
    """Independent MLA forward using weights extracted from ``attn``."""
    T, H = h.shape[0], attn.num_heads
    Dnope, Drope, Dv, L = (
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank)
    eps = cfg.rms_norm_eps
    q = F.linear(h, attn.q_a_proj.weight)
    q = _ref_rmsnorm(q, attn.q_a_layernorm.weight, eps)
    q = F.linear(q, attn.q_b_proj.weight).view(T, H, cfg.qk_head_dim)
    q_nope, q_pe = q.split([Dnope, Drope], dim=-1)
    latent = F.linear(h, attn.kv_a_proj_with_mqa.weight)
    kv_a, k_pe = latent.split([L, Drope], dim=-1)
    kv_a = _ref_rmsnorm(kv_a, attn.kv_a_layernorm.weight, eps)
    kv = F.linear(kv_a, attn.kv_b_proj.weight).view(T, H, Dnope + Dv)
    k_nope, v = kv.split([Dnope, Dv], dim=-1)
    k_pe = k_pe.view(T, 1, Drope)
    rk = _rope_kwargs(cfg)
    q_pe, k_pe = _ref_yarn_rope(pos, q_pe, k_pe, **rk)
    # M6 mitigation: q/k assembled at Dqk then zero-padded to padded_head_dim, v
    # padded from Dv to padded_head_dim (see KimiMLAAttention.forward).
    pad = cfg.padded_head_dim
    q = F.pad(torch.cat([q_nope, q_pe], dim=-1), [0, pad - cfg.qk_head_dim]) * boost
    k = F.pad(torch.cat([k_nope, k_pe.expand(T, H, Drope)], dim=-1), [0, pad - cfg.qk_head_dim])
    v = F.pad(v, [0, pad - Dv])
    return q, k, v


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_yarn_rope_matches_reference():
    torch.manual_seed(0)
    # mscale != mscale_all_dim so the cos/sin amplitude factor is non-trivial.
    rd, base, factor, max_pos = 8, 50000.0, 32.0, 4096
    ms, msad = 1.0, 0.5
    rope = KimiYarnRotaryEmbedding(rd, base, factor, max_pos, 32, 1, ms, msad).to(DEVICE)
    pos = torch.arange(6, device=DEVICE)
    q = torch.randn(6, 4, rd, device=DEVICE)
    k = torch.randn(6, 1, rd, device=DEVICE)

    gq, gk = rope(pos, q, k)
    rq, rk = _ref_yarn_rope(pos, q, k, rd, base, factor, max_pos, 32, 1, ms, msad)
    torch.testing.assert_close(gq, rq, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gk, rk, rtol=1e-4, atol=1e-4)
    assert abs(rope.mscale - 1.0) > 1e-3  # amplitude path is exercised


def _build_attention(cfg, dtype):
    attn = KimiMLAAttention(cfg).to(device=DEVICE, dtype=dtype)
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    return attn


def test_mla_qkv_assembly_matches_reference():
    torch.manual_seed(1)
    cfg = KimiK2Config.reduced()
    dtype = torch.bfloat16
    attn = _build_attention(cfg, dtype)
    cache = _MockMLACache(cfg.padded_head_dim)
    h = torch.randn(5, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.1
    pos = torch.arange(5, device=DEVICE)

    attn(h, cache, pos)  # populates cache.captured with the assembled q/k/v
    ref_q, ref_k, ref_v = _ref_mla(attn, cfg, h, pos, cache.scale, attn.softmax_scale_boost)

    torch.testing.assert_close(cache.captured["q"], ref_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache.captured["k"], ref_k, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache.captured["v"], ref_v, rtol=2e-2, atol=2e-2)


def test_mla_attention_forward_matches_reference():
    torch.manual_seed(2)
    cfg = KimiK2Config.reduced()
    dtype = torch.bfloat16
    attn = _build_attention(cfg, dtype)
    cache = _MockMLACache(cfg.padded_head_dim)
    h = torch.randn(7, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.1
    pos = torch.arange(7, device=DEVICE)

    got = attn(h, cache, pos)

    ref_q, ref_k, ref_v = _ref_mla(attn, cfg, h, pos, cache.scale, attn.softmax_scale_boost)
    ref_attn = _MockMLACache(cfg.padded_head_dim).run_attention(ref_q, ref_k, ref_v)
    ref_out = ref_attn[..., : cfg.v_head_dim].reshape(7, attn.num_heads * cfg.v_head_dim)
    expected = F.linear(ref_out, attn.o_proj.weight)

    assert got.shape == (7, cfg.hidden_size)
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)
