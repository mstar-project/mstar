"""Phase-A GPU check: the REAL ``KimiMLAAttention.forward`` absorbed branch.

The CPU gate (``test/modular/test_kimi_mla_absorb.py``) proves the absorption
algebra with pure-torch references. This test drives the actual wired forward —
``config.mla_absorb=True`` -> ``_forward_absorbed`` -> ``run_attention_mla`` — so
it needs a GPU (MLA RMSNorm uses a FlashInfer kernel). A ``_MockMLALatentCache``
stands in for the Phase-B paged latent backend: its ``run_attention_mla`` does a
causal SDPA over ``[kv_c | k_pe]`` (value = ``kv_c``) at the DeepSeek scale, which
is exactly what the FlashInfer MLA kernel will compute. The real kernel is locked
to ckv=512/kpe=64 so it can't run at the reduced dims — that path is validated in
Phase B on real dims.

Matching the independent DeepSeek MLA (materialized k_nope/v, no absorption) proves
the wired absorbed forward is numerically the naive path.

Run:  pytest test/integration/test_kimi_mla_absorb_forward.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.kimi_k2_7.components.attention import KimiMLAAttention
from mstar.model.kimi_k2_7.components.rope import (
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    rotate_gptj,
    yarn_get_mscale,
)
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the absorbed forward runs MLA RMSNorm (a FlashInfer kernel) on GPU",
)

DEVICE = "cuda"


# --------------------------------------------------------------------------
# References (device-aware; mirror test_kimi_mla_paged.py).
# --------------------------------------------------------------------------

def _ref_rmsnorm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def _ref_yarn_rope(pos, q_pe, k_pe, cfg):
    r = cfg.rope_scaling
    rotary_dim, base, factor = cfg.qk_rope_head_dim, cfg.rope_theta, r["factor"]
    max_pos = r["original_max_position_embeddings"]
    beta_fast, beta_slow = r.get("beta_fast", 32), r.get("beta_slow", 1)
    mscale, mscale_all_dim = r.get("mscale", 1.0), r.get("mscale_all_dim", 0.0)
    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, device=q_pe.device).float() / rotary_dim)
    ext, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, rotary_dim, base, max_pos)
    mask = 1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, torch.float).to(q_pe.device)
    inv_freq = interp * (1 - mask) + ext * mask
    amp = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
    freqs = torch.outer(pos.float(), inv_freq)
    cos = (freqs.cos() * amp).repeat_interleave(2, -1).unsqueeze(-2)
    sin = (freqs.sin() * amp).repeat_interleave(2, -1).unsqueeze(-2)
    qr = q_pe.float() * cos + rotate_gptj(q_pe.float()) * sin
    kr = k_pe.float() * cos + rotate_gptj(k_pe.float()) * sin
    return qr.to(q_pe.dtype), kr.to(k_pe.dtype)


def _sdpa_causal(q, k, v, scale):
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))  # (H,T,D)
    t = q.shape[0]
    causal = torch.triu(torch.full((t, t), float("-inf"), device=q.device), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


def _deepseek_scale(cfg):
    r = cfg.rope_scaling
    mscale = yarn_get_mscale(r["factor"], r.get("mscale_all_dim", 0.0))
    return cfg.qk_head_dim ** -0.5 * mscale * mscale


def _ref_deepseek_mla(attn, cfg, h, pos):
    """Naive DeepSeek MLA (materialized k_nope/v; no absorption) — ground truth."""
    t, heads = h.shape[0], attn.num_heads
    d_nope, d_rope, d_v, latent = (
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank)
    eps = cfg.rms_norm_eps
    q = _ref_rmsnorm(F.linear(h, attn.q_a_proj.weight), attn.q_a_layernorm.weight, eps)
    q = F.linear(q, attn.q_b_proj.weight).view(t, heads, cfg.qk_head_dim)
    q_nope, q_pe = q.split([d_nope, d_rope], dim=-1)
    lat = F.linear(h, attn.kv_a_proj_with_mqa.weight)
    kv_a, k_pe = lat.split([latent, d_rope], dim=-1)
    kv = F.linear(_ref_rmsnorm(kv_a, attn.kv_a_layernorm.weight, eps),
                  attn.kv_b_proj.weight).view(t, heads, d_nope + d_v)
    k_nope, v = kv.split([d_nope, d_v], dim=-1)
    k_pe = k_pe.view(t, 1, d_rope)
    q_pe, k_pe = _ref_yarn_rope(pos, q_pe, k_pe, cfg)
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(t, heads, d_rope)], dim=-1)
    out = _sdpa_causal(q, k, v, _deepseek_scale(cfg)).reshape(t, heads * d_v)
    return F.linear(out, attn.o_proj.weight)


class _MockMLALatentCache:
    """Phase-B latent-backend stand-in: causal SDPA over [kv_c|k_pe], value=kv_c."""

    def __init__(self, sm_scale):
        self.sm_scale = sm_scale

    def set_layer_idx(self, _i):
        pass

    def set_active_label(self, _l):
        pass

    def advance_seq_lens(self, *_a, **_k):
        pass

    def run_attention_mla(self, q_nope, q_pe, kv_c, k_pe):
        t, heads, latent = q_nope.shape
        d_rope = q_pe.shape[-1]
        query = torch.cat([q_nope, q_pe], dim=-1)                 # (T,H,L+Drope)
        kv_c_h = kv_c.expand(t, heads, latent)                    # MQA broadcast
        key = torch.cat([kv_c_h, k_pe.expand(t, heads, d_rope)], dim=-1)
        return _sdpa_causal(query, key, kv_c_h, self.sm_scale)    # (T,H,L)


def _build_attention(cfg, dtype):
    attn = KimiMLAAttention(cfg).to(device=DEVICE, dtype=dtype)
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    attn.process_weights_after_loading()  # build w_kc / w_vc on-device
    return attn


def test_absorbed_forward_matches_deepseek():
    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    cfg.mla_absorb = True
    dtype = torch.bfloat16
    attn = _build_attention(cfg, dtype)
    assert attn.w_kc is not None and attn.w_vc is not None

    t = 7
    h = torch.randn(t, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.1
    pos = torch.arange(t, device=DEVICE)

    cache = _MockMLALatentCache(_deepseek_scale(cfg))
    with torch.no_grad():
        got = attn(h, cache, pos)

    expected = _ref_deepseek_mla(attn, cfg, h, pos)
    assert got.shape == (t, cfg.hidden_size)
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)
