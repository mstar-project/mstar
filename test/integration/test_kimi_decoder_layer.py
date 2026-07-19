"""M4 golden tests for the Kimi-K2.7 / DeepSeek-V3 decoder layer.

One golden per feed-forward variant — a dense layer (``layer_idx=0``, below
``first_k_dense_replace``) and a MoE layer (``layer_idx=1``) — each compared to a
self-contained inline reference that re-derives the whole block:
pre-norm → naive-MLA self-attention → residual → pre-norm → (dense-or-MoE) FFN →
residual. The inner attention and FFN references are the same ones the M2/M3
goldens use, cited to vLLM; what this test adds is the residual/norm *wiring*,
matching vLLM ``models/deepseek_v2.py::DeepseekV2DecoderLayer.forward``.

A ``_MockMLACache`` stands in for the paged cache (causal SDPA at the fixed
``1/sqrt(qk_head_dim)`` scale FlashInfer uses); the real paged ``run_attention``
is exercised separately in ``test_kimi_flashinfer_attention.py``.

GPU test (mstar RMSNorm + the fused expert GEMM are CUDA/half-precision only);
skips without a GPU.

Run:  pytest test/integration/test_kimi_decoder_layer.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.kimi_k2_7.components.decoder_layer import KimiDecoderLayer
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.components.rope import (
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    rotate_gptj,
    yarn_get_mscale,
)
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="M4 golden tests need a GPU (RMSNorm + fused expert GEMM are CUDA-only)",
)

DEVICE = "cuda"


# --------------------------------------------------------------------------
# Inline references (cited to vLLM deepseek_v2.py / deepseek_scaling_rope.py /
# cpu_fused_moe.py) — self-contained, no dependency on the golden harness.
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
    """Causal SDPA at a fixed scale (mirrors _MockMLACache / FlashInfer)."""
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))  # (H,T,D)
    T = q.shape[0]
    causal = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


def _ref_attn_forward(attn, cfg, h_normed, pos):
    """Independent naive-MLA forward matching KimiMLAAttention.forward."""
    T, H = h_normed.shape[0], attn.num_heads
    Dnope, Drope, Dv, L = (
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank)
    eps = cfg.rms_norm_eps
    q = _ref_rmsnorm(F.linear(h_normed, attn.q_a_proj.weight), attn.q_a_layernorm.weight, eps)
    q = F.linear(q, attn.q_b_proj.weight).view(T, H, cfg.qk_head_dim)
    q_nope, q_pe = q.split([Dnope, Drope], dim=-1)
    latent = F.linear(h_normed, attn.kv_a_proj_with_mqa.weight)
    kv_a, k_pe = latent.split([L, Drope], dim=-1)
    kv = F.linear(_ref_rmsnorm(kv_a, attn.kv_a_layernorm.weight, eps),
                  attn.kv_b_proj.weight).view(T, H, Dnope + Dv)
    k_nope, v = kv.split([Dnope, Dv], dim=-1)
    k_pe = k_pe.view(T, 1, Drope)
    r = cfg.rope_scaling
    q_pe, k_pe = _ref_yarn_rope(
        pos, q_pe, k_pe, Drope, cfg.rope_theta, r["factor"],
        r["original_max_position_embeddings"], r.get("beta_fast", 32),
        r.get("beta_slow", 1), r.get("mscale", 1.0), r.get("mscale_all_dim", 0.0))
    q = torch.cat([q_nope, q_pe], dim=-1) * attn.softmax_scale_boost
    k = torch.cat([k_nope, k_pe.expand(T, H, Drope)], dim=-1)
    v = F.pad(v, [0, cfg.qk_head_dim - Dv])
    out = _sdpa_causal(q, k, v, cfg.qk_head_dim ** -0.5)
    out = out[..., :Dv].reshape(T, H * Dv)
    return F.linear(out, attn.o_proj.weight)


def _ref_grouped_topk(logits, bias, n_group, topk_group, top_k,
                      norm_topk_prob, routed_scaling_factor):
    scores = logits.float().sigmoid()
    T = scores.shape[0]
    original = scores
    scores = scores + bias.unsqueeze(0)
    group_scores = scores.view(T, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(T, n_group, scores.shape[-1] // n_group)
        .reshape(T, -1)
    )
    masked = scores.masked_fill(~score_mask.bool(), float("-inf"))
    ids = torch.topk(masked, k=top_k, dim=-1, sorted=False)[1]
    weights = original.gather(1, ids)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * routed_scaling_factor
    return weights, ids


def _ref_routed_experts(h, gate_up, down, weights, ids):
    T, H = h.shape
    inter = down.shape[-1]
    out = torch.zeros(T, H, dtype=h.dtype, device=h.device)
    for t in range(T):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            gu = gate_up[e] @ h[t]
            g, u = gu[:inter], gu[inter:]
            out[t] += weights[t, j] * (down[e] @ (F.silu(g) * u))
    return out


def _ref_swiglu(x, gate_w, up_w, down_w):
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


def _ref_mlp_forward(mlp, cfg, h_normed):
    """Dense SwiGLU or MoE (routed + ungated shared), matching the module."""
    if isinstance(mlp, KimiSparseMoeBlock):
        I = cfg.moe_intermediate_size
        si = cfg.moe_intermediate_size * cfg.n_shared_experts
        logits = F.linear(h_normed.float(), mlp.gate.weight.float())
        weights, ids = _ref_grouped_topk(
            logits, mlp.gate.e_score_correction_bias, cfg.n_group, cfg.topk_group,
            cfg.num_experts_per_tok, cfg.norm_topk_prob, cfg.routed_scaling_factor)
        routed = _ref_routed_experts(
            h_normed, mlp.experts.gate_up_proj, mlp.experts.down_proj,
            weights.to(h_normed.dtype), ids)
        sh = mlp.shared_expert
        shared = _ref_swiglu(
            h_normed, sh.gate_up_proj.weight[:si], sh.gate_up_proj.weight[si:],
            sh.down_proj.weight)
        return routed + shared
    i = cfg.intermediate_size
    return _ref_swiglu(
        h_normed, mlp.gate_up_proj.weight[:i], mlp.gate_up_proj.weight[i:],
        mlp.down_proj.weight)


def _ref_decoder_layer(layer, cfg, h, pos):
    eps = cfg.rms_norm_eps
    attn_in = _ref_rmsnorm(h, layer.input_layernorm.weight, eps)
    h1 = h + _ref_attn_forward(layer.self_attn, cfg, attn_in, pos)
    mlp_in = _ref_rmsnorm(h1, layer.post_attention_layernorm.weight, eps)
    return h1 + _ref_mlp_forward(layer.mlp, cfg, mlp_in)


# --------------------------------------------------------------------------
# Mock paged cache: causal SDPA at 1/sqrt(head_dim), no cross-layer history
# (a single prefill forward; each layer attends its own q/k/v).
# --------------------------------------------------------------------------

class _MockMLACache:
    def __init__(self, head_dim):
        self.scale = head_dim ** -0.5

    def set_layer_idx(self, _i):
        pass

    def advance_seq_lens(self, *_a, **_k):
        pass

    def run_attention(self, q, k, v):
        return _sdpa_causal(q, k, v, self.scale)


def _build_layer(cfg, layer_idx, dtype):
    layer = KimiDecoderLayer(cfg, layer_idx).to(device=DEVICE, dtype=dtype)
    a = layer.self_attn
    for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa, a.kv_b_proj, a.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (a.q_a_layernorm, a.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    layer.input_layernorm.weight.data.normal_(1.0, 0.02)
    layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
    mlp = layer.mlp
    if isinstance(mlp, KimiSparseMoeBlock):
        # Keep the router fp32 (deterministic selection); experts/shared bf16.
        mlp.gate.weight.data = torch.randn(
            cfg.n_routed_experts, cfg.hidden_size, device=DEVICE)
        mlp.gate.e_score_correction_bias.data = torch.randn(
            cfg.n_routed_experts, device=DEVICE)
        mlp.experts.gate_up_proj.data.normal_(0, 0.05)
        mlp.experts.down_proj.data.normal_(0, 0.05)
        mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
    else:
        mlp.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.down_proj.weight.data.normal_(0, 0.05)
    return layer


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_dense_decoder_layer_matches_reference():
    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    dtype = torch.bfloat16
    layer = _build_layer(cfg, layer_idx=0, dtype=dtype)  # dense (< first_k_dense_replace)
    assert not isinstance(layer.mlp, KimiSparseMoeBlock)

    T = 6
    h = torch.randn(T, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.1
    pos = torch.arange(T, device=DEVICE)

    got = layer(h, _MockMLACache(cfg.qk_head_dim), pos)
    expected = _ref_decoder_layer(layer, cfg, h, pos)

    assert got.shape == (T, cfg.hidden_size)
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)


def test_moe_decoder_layer_matches_reference():
    torch.manual_seed(1)
    cfg = KimiK2Config.reduced()
    dtype = torch.bfloat16
    layer = _build_layer(cfg, layer_idx=1, dtype=dtype)  # MoE (>= first_k_dense_replace)
    assert isinstance(layer.mlp, KimiSparseMoeBlock)

    T = 7
    h = torch.randn(T, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.1
    pos = torch.arange(T, device=DEVICE)

    got = layer(h, _MockMLACache(cfg.qk_head_dim), pos)
    expected = _ref_decoder_layer(layer, cfg, h, pos)

    assert got.shape == (T, cfg.hidden_size)
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)
