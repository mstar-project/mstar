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
from mstar.model.kimi_k2_7.kimi_model import KimiK2Model


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
    pos_freqs = base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    ext, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, rotary_dim, base, max_pos)
    mask = 1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, torch.float)
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
    causal = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


def _q_and_latent(attn, cfg, h, pos):
    t, heads = h.shape[0], attn.num_heads
    d_nope, d_rope, latent = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.kv_lora_rank
    eps = cfg.rms_norm_eps
    q = _ref_rmsnorm(F.linear(h, attn.q_a_proj.weight), attn.q_a_layernorm.weight, eps)
    q = F.linear(q, attn.q_b_proj.weight).view(t, heads, cfg.qk_head_dim)
    q_nope, q_pe = q.split([d_nope, d_rope], dim=-1)
    lat = F.linear(h, attn.kv_a_proj_with_mqa.weight)
    kv_a, k_pe = lat.split([latent, d_rope], dim=-1)
    kv_c = _ref_rmsnorm(kv_a, attn.kv_a_layernorm.weight, eps)  # (T, L)
    k_pe = k_pe.view(t, 1, d_rope)
    q_pe, k_pe = _ref_yarn_rope(pos, q_pe, k_pe, cfg)
    return q_nope, q_pe, kv_c, k_pe


def _deepseek_scale(cfg):
    r = cfg.rope_scaling
    mscale = yarn_get_mscale(r["factor"], r.get("mscale_all_dim", 0.0))
    return cfg.qk_head_dim ** -0.5 * mscale * mscale


def _ref_deepseek_mla(attn, cfg, h, pos):
    t, heads = h.shape[0], attn.num_heads
    d_nope, d_rope, d_v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    q_nope, q_pe, kv_c, k_pe = _q_and_latent(attn, cfg, h, pos)
    kv = F.linear(kv_c, attn.kv_b_proj.weight).view(t, heads, d_nope + d_v)
    k_nope, v = kv.split([d_nope, d_v], dim=-1)
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(t, heads, d_rope)], dim=-1)
    out = _sdpa_causal(q, k, v, _deepseek_scale(cfg)).reshape(t, heads * d_v)
    return F.linear(out, attn.o_proj.weight)


def _absorbed_mla(attn, cfg, h, pos):
    t, heads = h.shape[0], attn.num_heads
    d_rope, d_v, latent = cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    q_nope, q_pe, kv_c, k_pe = _q_and_latent(attn, cfg, h, pos)
    q_nope = torch.einsum("thd,hdl->thl", q_nope, attn.w_kc)  # (T,H,L)
    query = torch.cat([q_nope, q_pe], dim=-1)                 # (T,H,L+Drope)
    kv_c_h = kv_c.unsqueeze(1).expand(t, heads, latent)       # MQA: shared over heads
    key = torch.cat([kv_c_h, k_pe.expand(t, heads, d_rope)], dim=-1)
    attn_latent = _sdpa_causal(query, key, kv_c_h, _deepseek_scale(cfg))  # (T,H,L)
    out = torch.einsum("thl,hdl->thd", attn_latent, attn.w_vc).reshape(t, heads * d_v)
    return F.linear(out, attn.o_proj.weight)


def _build_attention_cpu(seed=0):
    torch.manual_seed(seed)
    cfg = KimiK2Config.reduced()
    cfg.mla_absorb = True
    attn = KimiMLAAttention(cfg)  # CPU, float32
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    attn.process_weights_after_loading()  # split kv_b_proj -> w_kc / w_vc
    return attn, cfg


def test_absorb_reconstructs_kv_b_proj():
    attn, cfg = _build_attention_cpu(seed=1)
    heads, d_nope, d_v, latent = (
        attn.num_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank)
    assert tuple(attn.w_kc.shape) == (heads, d_nope, latent)
    assert tuple(attn.w_vc.shape) == (heads, d_v, latent)

    kv_c = torch.randn(5, latent)
    kv = F.linear(kv_c, attn.kv_b_proj.weight).view(5, heads, d_nope + d_v)
    k_nope_ref, v_ref = kv.split([d_nope, d_v], dim=-1)

    k_nope_abs = torch.einsum("tl,hdl->thd", kv_c, attn.w_kc)
    v_abs = torch.einsum("tl,hdl->thd", kv_c, attn.w_vc)
    torch.testing.assert_close(k_nope_abs, k_nope_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_abs, v_ref, rtol=1e-5, atol=1e-5)


def test_fused_qkv_a_proj():
    attn, cfg = _build_attention_cpu(seed=3)
    expected = torch.cat(
        [attn.q_a_proj.weight, attn.kv_a_proj_with_mqa.weight], dim=0)
    assert tuple(attn.fused_qkv_a_proj_weight.shape) == (
        cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size)
    torch.testing.assert_close(
        attn.fused_qkv_a_proj_weight, expected, rtol=0, atol=0)


def test_absorbed_math_matches_deepseek():
    attn, cfg = _build_attention_cpu(seed=2)
    t = 7
    h = torch.randn(t, cfg.hidden_size) * 0.1
    pos = torch.arange(t)

    absorbed = _absorbed_mla(attn, cfg, h, pos)
    reference = _ref_deepseek_mla(attn, cfg, h, pos)

    assert absorbed.shape == (t, cfg.hidden_size)
    # Pure fp32 algebra; residual comes only from op ordering.
    torch.testing.assert_close(absorbed, reference, rtol=1e-4, atol=1e-4)


def _kv_cfg(mla_absorb):
    m = object.__new__(KimiK2Model)          # skip __init__ (tokenizer/weights/GPU)
    m.config = KimiK2Config.reduced()
    m.config.mla_absorb = mla_absorb
    return m.get_kv_cache_config()[0]


def test_kv_cache_config_absorbed_shrinks_latent():
    cfg = KimiK2Config.reduced()
    kv = _kv_cfg(mla_absorb=True)
    assert kv.num_kv_heads == 1
    assert kv.head_dim == cfg.kv_lora_rank + cfg.qk_rope_head_dim  # 32 + 8 = 40
    assert kv.num_qo_heads == cfg.num_attention_heads              # 4 (q still sharded)
    assert kv.attention_backend == "mla_absorb"
    naive_elems = 2 * cfg.num_attention_heads * cfg.padded_head_dim
    absorbed_elems = kv.num_kv_heads * kv.head_dim
    assert naive_elems == 512 and absorbed_elems == 40


def test_kv_cache_config_flag_off_is_naive():
    cfg = KimiK2Config.reduced()
    kv = _kv_cfg(mla_absorb=False)
    assert kv.num_kv_heads == cfg.num_attention_heads   # 4
    assert kv.head_dim == cfg.padded_head_dim           # 64
    assert kv.attention_backend == "flashinfer"         # default naive backend
