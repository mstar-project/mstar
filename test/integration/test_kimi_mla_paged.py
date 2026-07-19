"""M6 step 1: the real paged MLA path, end-to-end, at the DeepSeek scale.

This is the test that finally validates ``KimiMLAAttention`` over mstar's REAL
paged ``FlashInferCacheManager`` (genuine ``PagedAllocationManager`` + KV cache),
not the MockCacheHandle SDPA stand-in the M3/M4/M5 goldens use.

It closes the M4 FlashInfer-192 blocker. The naive MLA pads q/k (from
``qk_head_dim``) and v (from ``v_head_dim``) up to ``padded_head_dim`` — the
smallest FlashInfer-SM90-supported head_dim {64,128,256} >= ``qk_head_dim`` — so
the reduced ``qk_head_dim=24`` becomes 64 (real Kimi 192 -> 256). The Hopper
prefill kernel ``static_assert``s ``head_dim_vo in {64,128,256}``, so the raw 24
(and 192) fail to JIT-build; 64 builds and runs.

The correctness crux is the **softmax-scale compensation**. run_attention applies
a fixed ``1/sqrt(padded_head_dim)`` scale, but DeepSeek's intended softmax scale
is ``qk_head_dim**-0.5 * mscale**2``. The zero-pad dims contribute 0 to q·k, so
we fold ``boost = mscale**2 * sqrt(padded_head_dim / qk_head_dim)`` into q:

    scores = (q*boost)·k * padded_head_dim**-0.5
           = q·k * mscale**2 * sqrt(padded/qk) * padded**-0.5
           = q·k * mscale**2 * qk**-0.5            (the DeepSeek scale).

The reference below is the **independent DeepSeek computation** — projections +
YARN RoPE + causal SDPA at ``qk_head_dim**-0.5 * mscale**2`` over the UNPADDED q/k
(Dqk) and v (Dv), then output slice + o_proj. Matching it proves the padded paged
run + scale compensation reproduce the intended result exactly.

Run:  pytest test/integration/test_kimi_mla_paged.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.cache_manager import WorkspaceBufferManager, create_cache_manager
from mstar.engine.kv_store import (
    KVCacheConfig,
    PagedAllocationManager,
    TransferEngineInfo,
)
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
    reason="real FlashInfer paged MLA needs a GPU",
)

DEVICE = torch.device("cuda")


# --------------------------------------------------------------------------
# Real paged cache manager (mirrors test_kimi_flashinfer_attention.py).
# --------------------------------------------------------------------------

def _make_real_cache_manager(num_heads, head_dim, dtype, page_size=128, max_num_pages=8):
    kv_cache = torch.zeros(
        2, max_num_pages, 2, page_size, num_heads, head_dim,
        dtype=dtype, device=DEVICE,
    ).contiguous()
    kv_cfg = KVCacheConfig(
        num_layers=2, num_kv_heads=num_heads, head_dim=head_dim,
        max_seq_len=page_size * max_num_pages, max_num_pages=max_num_pages,
        page_size=page_size, num_qo_heads=num_heads,
    )
    transfer_info = TransferEngineInfo(
        my_entity_id="kimi_mla_paged_test",
        my_session_id="kimi_session",
        transfer_engine=LocalTransferEngine("localhost"),
    )
    alloc = PagedAllocationManager(
        config=kv_cfg, kv_cache=kv_cache, transfer_engine_info=transfer_info,
    )
    alloc.add_request("r0", ["main"])
    buffers = WorkspaceBufferManager(64 * 1024 * 1024, device=DEVICE)
    cm = create_cache_manager(
        request_ids=["r0"],
        active_labels_per_request={"r0": "main"},
        kv_cache=kv_cache,
        alloc_manager=alloc,
        buffer_manager=buffers,
        kv_cache_config=kv_cfg,
        device=DEVICE,
    )
    return cm, alloc


# --------------------------------------------------------------------------
# Independent DeepSeek reference (no pad; scale = qk_head_dim**-0.5 * mscale**2).
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
    qt, kt, vt = (t.transpose(0, 1).float() for t in (q, k, v))  # (H,T,D)
    T = q.shape[0]
    causal = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    attn = (torch.einsum("hqd,hkd->hqk", qt, kt) * scale + causal).softmax(-1)
    return torch.einsum("hqk,hkd->hqd", attn, vt).transpose(0, 1).to(q.dtype)


def _ref_deepseek_mla(attn: KimiMLAAttention, cfg, h, pos):
    """The intended DeepSeek MLA output: NO padding, scale = qk**-0.5 * mscale**2."""
    T, H = h.shape[0], attn.num_heads
    Dnope, Drope, Dv, L = (
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank)
    eps = cfg.rms_norm_eps
    q = _ref_rmsnorm(F.linear(h, attn.q_a_proj.weight), attn.q_a_layernorm.weight, eps)
    q = F.linear(q, attn.q_b_proj.weight).view(T, H, cfg.qk_head_dim)
    q_nope, q_pe = q.split([Dnope, Drope], dim=-1)
    latent = F.linear(h, attn.kv_a_proj_with_mqa.weight)
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
    q = torch.cat([q_nope, q_pe], dim=-1)  # (T, H, Dqk) — NOT padded
    k = torch.cat([k_nope, k_pe.expand(T, H, Drope)], dim=-1)  # (T, H, Dqk)
    mscale = yarn_get_mscale(r["factor"], r.get("mscale_all_dim", 0.0))
    deepseek_scale = cfg.qk_head_dim ** -0.5 * mscale * mscale
    out = _sdpa_causal(q, k, v, deepseek_scale)  # v is Dv-wide, output Dv-wide
    out = out.reshape(T, H * Dv)
    return F.linear(out, attn.o_proj.weight)


def _build_attention(cfg, dtype):
    attn = KimiMLAAttention(cfg).to(device=DEVICE, dtype=dtype)
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    return attn


def test_paged_mla_matches_deepseek_sdpa():
    """KimiMLAAttention through the REAL paged FlashInferCacheManager (head_dim =
    padded_head_dim = 64) == the independent DeepSeek MLA at qk**-0.5 * mscale**2.

    This validates both (a) the real paged path builds+runs at the padded head_dim
    (the M4 FlashInfer-192 blocker mitigation), and (b) the scale compensation is
    exactly right — the padded run reproduces the unpadded DeepSeek scale.
    """
    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    assert cfg.qk_head_dim == 24 and cfg.padded_head_dim == 64  # the mitigation
    dtype = torch.bfloat16
    attn = _build_attention(cfg, dtype)

    T = 6
    h = torch.randn(T, cfg.hidden_size, device=DEVICE, dtype=dtype) * 0.1
    pos = torch.arange(T, device=DEVICE)

    cm, alloc = _make_real_cache_manager(cfg.num_attention_heads, cfg.padded_head_dim, dtype)
    try:
        cm.set_active_label("main")
        cm.plan_attention(seq_lens=[T], is_causal=True, dtype=dtype)
        cm.set_layer_idx(0)
        with torch.no_grad():
            got = attn(h, cm, pos)
        torch.cuda.synchronize()
    finally:
        alloc.cleanup()

    expected = _ref_deepseek_mla(attn, cfg, h, pos)
    assert got.shape == (T, cfg.hidden_size)
    # bf16 through the real FlashInfer kernel; the scale compensation is exact in
    # exact arithmetic, so any residual is pure bf16 rounding.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
