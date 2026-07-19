"""M4 full-forward golden test for Kimi-K2.7 / DeepSeek-V3 (assembled backbone).

Runs ``KimiForCausalLM`` end to end on the reduced config — token ids →
embedding → stacked ``KimiDecoderLayer`` blocks (including the dense→MoE
transition at ``first_k_dense_replace=1``) → final RMSNorm → untied LM head →
logits — and compares against a self-contained inline reference that re-derives
every step. The inner attention / FFN / router references are the same ones the
M2/M3 goldens use (cited to vLLM); this test verifies the *assembly*: embedding,
the per-layer cache-handle contract (``set_layer_idx`` each layer,
``advance_seq_lens`` once after), the layer stack, and the LM head.

A ``_MockMLACache`` stands in for the paged cache (causal SDPA at the fixed
``1/sqrt(qk_head_dim)`` scale). Because a single prefill forward re-attends the
same tokens each layer with no cross-layer history, the mock — which attends the
q/k/v of each ``run_attention`` call independently — reproduces the paged
prefill exactly. The real FlashInfer paged path is validated separately in
``test_kimi_flashinfer_attention.py``.

Refs: vLLM ``models/deepseek_v2.py`` (``DeepseekV2Model`` / ``DecoderLayer`` /
``DeepseekV2MoE``), ``rotary_embedding/deepseek_scaling_rope.py``,
``fused_moe/cpu_fused_moe.py::grouped_topk``.

Run:  pytest test/integration/test_kimi_forward.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.kimi_k2_7.components.causal_lm import KimiForCausalLM
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
    reason="M4 full-forward golden needs a GPU (RMSNorm + fused expert GEMM)",
)

DEVICE = "cuda"


# --------------------------------------------------------------------------
# Inline references (self-contained; cited to vLLM). Same math as the M2/M3
# component goldens, assembled here into a whole-model forward.
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
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))
    T = q.shape[0]
    causal = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


def _ref_attn_forward(attn, cfg, h_normed, pos):
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
    # M6 mitigation: q/k padded from Dqk and v from Dv up to padded_head_dim; the
    # softmax_scale_boost compensates so run_attention's padded_head_dim**-0.5
    # scale reproduces the DeepSeek qk_head_dim**-0.5 * mscale**2 scale.
    pad = cfg.padded_head_dim
    q = F.pad(torch.cat([q_nope, q_pe], dim=-1), [0, pad - cfg.qk_head_dim]) * attn.softmax_scale_boost
    k = F.pad(torch.cat([k_nope, k_pe.expand(T, H, Drope)], dim=-1), [0, pad - cfg.qk_head_dim])
    v = F.pad(v, [0, pad - Dv])
    out = _sdpa_causal(q, k, v, cfg.padded_head_dim ** -0.5)
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
    if isinstance(mlp, KimiSparseMoeBlock):
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


def _ref_forward(model, cfg, ids, pos):
    h = F.embedding(ids, model.model.embed_tokens.weight)
    for layer in model.model.layers:
        h = _ref_decoder_layer(layer, cfg, h, pos)
    h = _ref_rmsnorm(h, model.model.norm.weight, cfg.rms_norm_eps)
    return F.linear(h, model.lm_head.weight)


# --------------------------------------------------------------------------
# Mock paged cache (causal SDPA at 1/sqrt(head_dim)) + weight init
# --------------------------------------------------------------------------

class _MockMLACache:
    def __init__(self, head_dim):
        self.scale = head_dim ** -0.5
        self.layer_idx = 0
        self.advance_calls = 0

    def set_layer_idx(self, i):
        self.layer_idx = i

    def advance_seq_lens(self, *_a, **_k):
        self.advance_calls += 1

    def run_attention(self, q, k, v):
        return _sdpa_causal(q, k, v, self.scale)


def _fill_layer(layer, cfg):
    a = layer.self_attn
    for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa, a.kv_b_proj, a.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (a.q_a_layernorm, a.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    layer.input_layernorm.weight.data.normal_(1.0, 0.02)
    layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
    mlp = layer.mlp
    if isinstance(mlp, KimiSparseMoeBlock):
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


def _build_model(cfg, dtype):
    model = KimiForCausalLM(cfg).to(device=DEVICE, dtype=dtype)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        _fill_layer(layer, cfg)
    return model.eval()


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------

def test_full_forward_logits_match_reference():
    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    dtype = torch.bfloat16
    model = _build_model(cfg, dtype)
    # The stack spans the dense->MoE transition (first_k_dense_replace=1).
    assert not isinstance(model.model.layers[0].mlp, KimiSparseMoeBlock)
    assert isinstance(model.model.layers[1].mlp, KimiSparseMoeBlock)

    T = 8
    ids = torch.randint(0, cfg.vocab_size, (T,), device=DEVICE)
    pos = torch.arange(T, device=DEVICE)

    cache = _MockMLACache(cfg.padded_head_dim)
    with torch.no_grad():
        got = model(ids, cache, pos)
    expected = _ref_forward(model, cfg, ids, pos)

    # advance_seq_lens is called exactly once per forward (after the layer loop),
    # not once per layer — the cache-handle contract mirrored from Orpheus.
    assert cache.advance_calls == 1
    assert got.shape == (T, cfg.vocab_size)
    torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)
