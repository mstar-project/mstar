import torch
import torch.nn.functional as F

from mstar.model.glm52.components.attention import Glm52MLAAttention
from mstar.model.glm52.components.rope import rotate_gptj
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.glm52_model import Glm52Model


def _ref_rmsnorm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def _ref_plain_rope(pos, q_pe, k_pe, cfg):
    """Textbook interleaved RoPE: no Yarn, no mscale — GLM-5.2's regime."""
    rotary_dim, base = cfg.qk_rope_head_dim, cfg.rope_theta
    inv_freq = 1.0 / base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    freqs = torch.outer(pos.float(), inv_freq)
    cos = freqs.cos().repeat_interleave(2, -1).unsqueeze(-2)
    sin = freqs.sin().repeat_interleave(2, -1).unsqueeze(-2)
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
    q_pe, k_pe = _ref_plain_rope(pos, q_pe, k_pe, cfg)
    return q_nope, q_pe, kv_c, k_pe


def _ref_dense_mla(attn, cfg, h, pos):
    t, heads = h.shape[0], attn.num_heads
    d_nope, d_rope, d_v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    q_nope, q_pe, kv_c, k_pe = _q_and_latent(attn, cfg, h, pos)
    kv = F.linear(kv_c, attn.kv_b_proj.weight).view(t, heads, d_nope + d_v)
    k_nope, v = kv.split([d_nope, d_v], dim=-1)
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(t, heads, d_rope)], dim=-1)
    out = _sdpa_causal(q, k, v, cfg.qk_head_dim ** -0.5).reshape(t, heads * d_v)
    return F.linear(out, attn.o_proj.weight)


def _absorbed_mla(attn, cfg, h, pos):
    t, heads = h.shape[0], attn.num_heads
    d_rope, d_v, latent = cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    q_nope, q_pe, kv_c, k_pe = _q_and_latent(attn, cfg, h, pos)
    q_nope = torch.einsum("thd,hdl->thl", q_nope, attn.w_kc)  # (T,H,L)
    query = torch.cat([q_nope, q_pe], dim=-1)                 # (T,H,L+Drope)
    kv_c_h = kv_c.unsqueeze(1).expand(t, heads, latent)       # MQA: shared over heads
    key = torch.cat([kv_c_h, k_pe.expand(t, heads, d_rope)], dim=-1)
    attn_latent = _sdpa_causal(query, key, kv_c_h, cfg.qk_head_dim ** -0.5)  # (T,H,L)
    out = torch.einsum("thl,hdl->thd", attn_latent, attn.w_vc).reshape(t, heads * d_v)
    return F.linear(out, attn.o_proj.weight)


def _build_attention_cpu(seed=0):
    torch.manual_seed(seed)
    cfg = Glm52ModelConfig.reduced()
    cfg.mla_absorb = True
    attn = Glm52MLAAttention(cfg)  # CPU, float32
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


def test_absorbed_math_matches_dense():
    attn, cfg = _build_attention_cpu(seed=2)
    t = 7
    h = torch.randn(t, cfg.hidden_size) * 0.1
    pos = torch.arange(t)

    absorbed = _absorbed_mla(attn, cfg, h, pos)
    reference = _ref_dense_mla(attn, cfg, h, pos)

    assert absorbed.shape == (t, cfg.hidden_size)
    # Pure fp32 algebra; residual comes only from op ordering.
    torch.testing.assert_close(absorbed, reference, rtol=1e-4, atol=1e-4)


def test_rope_position_zero_is_identity():
    attn, cfg = _build_attention_cpu(seed=4)
    q_pe = torch.randn(1, attn.num_heads, cfg.qk_rope_head_dim)
    k_pe = torch.randn(1, 1, cfg.qk_rope_head_dim)
    q_rot, k_rot = attn.rotary(torch.zeros(1, dtype=torch.long), q_pe, k_pe)
    torch.testing.assert_close(q_rot, q_pe, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k_rot, k_pe, rtol=1e-6, atol=1e-6)


def _kv_cfg(config):
    m = object.__new__(Glm52Model)  # skip __init__ (tokenizer/weights)
    m.config = config
    return m.get_kv_cache_config()[0]


def test_kv_cache_config_absorbed_is_latent_with_backend():
    cfg = Glm52ModelConfig()  # full model, mla_absorb default True
    kv = _kv_cfg(cfg)
    assert kv.num_kv_heads == 1
    assert kv.head_dim == cfg.kv_lora_rank + cfg.qk_rope_head_dim == 576
    assert kv.attention_backend == "mla_absorb"
    assert kv.softmax_scale == cfg.qk_head_dim ** -0.5  # 256**-0.5, no mscale
    assert kv.mla_ckv_dim == cfg.kv_lora_rank == 512


def test_kv_cache_config_flag_off_is_naive():
    cfg = Glm52ModelConfig.reduced()  # mla_absorb False
    kv = _kv_cfg(cfg)
    assert kv.num_kv_heads == cfg.num_attention_heads == 4
    assert kv.head_dim == cfg.padded_head_dim == 64  # qk 24 -> FlashInfer 64
    assert kv.attention_backend == "flashinfer"      # default naive backend


def test_full_model_head_geometry_needs_no_pad():
    cfg = Glm52ModelConfig()
    assert cfg.qk_head_dim == 256
    assert cfg.padded_head_dim == 256  # pad and softmax boost are no-ops
    attn_boost = (cfg.padded_head_dim / cfg.qk_head_dim) ** 0.5
    assert attn_boost == 1.0
