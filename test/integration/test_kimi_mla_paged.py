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
    # Any residual after exact scale compensation is bf16 FlashInfer rounding.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
