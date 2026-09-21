"""Absorbed MLA (w_kc / w_vc folded into Q and O over the latent cache) vs
the naive per-head K/V path: the pure algebra against a from-scratch dense
reference, then both engine paths through their real resources — the
``KVLayout.MLA`` cache + the MLA attention resource on its SDPA fallback,
and the NHD cache + the reference K/V attention — over a prefill and
decode steps that cross a page boundary.
"""
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


if "flashinfer" not in sys.modules:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.resources import (  # noqa: E402
    AttentionStep,
    KVStep,
    MlaAttentionStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.model.glm52._testing import build_cpu_resources  # noqa: E402
from mstar.model.glm52.components.attention import Glm52MLAAttention  # noqa: E402
from mstar.model.glm52.components.rope import rotate_gptj  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    KV_RESOURCE,
    Glm52ModelConfig,
)


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


def _source_weights(attn):
    """The projections the absorbed path folds away, snapshotted BEFORE
    process_weights_after_loading releases them (their storage is dropped
    once w_kc / w_vc / fused_qkv_a_proj_weight exist)."""
    return {
        "q_a": attn.q_a_proj.weight.detach().clone(),
        "kv_a": attn.kv_a_proj_with_mqa.weight.detach().clone(),
        "kv_b": attn.kv_b_proj.weight.detach().clone(),
    }


def _q_and_latent(attn, cfg, h, pos, src):
    t, heads = h.shape[0], attn.num_heads
    d_nope, d_rope, latent = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.kv_lora_rank
    eps = cfg.rms_norm_eps
    q = _ref_rmsnorm(F.linear(h, src["q_a"]), attn.q_a_layernorm.weight, eps)
    q = F.linear(q, attn.q_b_proj.weight).view(t, heads, cfg.qk_head_dim)
    q_nope, q_pe = q.split([d_nope, d_rope], dim=-1)
    lat = F.linear(h, src["kv_a"])
    kv_a, k_pe = lat.split([latent, d_rope], dim=-1)
    kv_c = _ref_rmsnorm(kv_a, attn.kv_a_layernorm.weight, eps)  # (T, L)
    k_pe = k_pe.view(t, 1, d_rope)
    q_pe, k_pe = _ref_plain_rope(pos, q_pe, k_pe, cfg)
    return q_nope, q_pe, kv_c, k_pe


def _ref_dense_mla(attn, cfg, h, pos, src):
    t, heads = h.shape[0], attn.num_heads
    d_nope, d_rope, d_v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    q_nope, q_pe, kv_c, k_pe = _q_and_latent(attn, cfg, h, pos, src)
    kv = F.linear(kv_c, src["kv_b"]).view(t, heads, d_nope + d_v)
    k_nope, v = kv.split([d_nope, d_v], dim=-1)
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(t, heads, d_rope)], dim=-1)
    out = _sdpa_causal(q, k, v, cfg.qk_head_dim ** -0.5).reshape(t, heads * d_v)
    return F.linear(out, attn.o_proj.weight)


def _absorbed_mla(attn, cfg, h, pos, src):
    t, heads = h.shape[0], attn.num_heads
    d_rope, d_v, latent = cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    q_nope, q_pe, kv_c, k_pe = _q_and_latent(attn, cfg, h, pos, src)
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
    src = _source_weights(attn)
    attn.process_weights_after_loading()  # split kv_b_proj -> w_kc / w_vc, free the sources
    return attn, cfg, src


def test_absorb_reconstructs_kv_b_proj():
    attn, cfg, src = _build_attention_cpu(seed=1)
    heads, d_nope, d_v, latent = (
        attn.num_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank)
    assert tuple(attn.w_kc.shape) == (heads, d_nope, latent)
    assert tuple(attn.w_vc.shape) == (heads, d_v, latent)

    kv_c = torch.randn(5, latent)
    kv = F.linear(kv_c, src["kv_b"]).view(5, heads, d_nope + d_v)
    k_nope_ref, v_ref = kv.split([d_nope, d_v], dim=-1)

    k_nope_abs = torch.einsum("tl,hdl->thd", kv_c, attn.w_kc)
    v_abs = torch.einsum("tl,hdl->thd", kv_c, attn.w_vc)
    torch.testing.assert_close(k_nope_abs, k_nope_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_abs, v_ref, rtol=1e-5, atol=1e-5)


def test_fused_qkv_a_proj():
    attn, cfg, src = _build_attention_cpu(seed=3)
    expected = torch.cat([src["q_a"], src["kv_a"]], dim=0)
    assert tuple(attn.fused_qkv_a_proj_weight.shape) == (
        cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size)
    torch.testing.assert_close(
        attn.fused_qkv_a_proj_weight, expected, rtol=0, atol=0)


def test_absorbed_math_matches_dense():
    attn, cfg, src = _build_attention_cpu(seed=2)
    t = 7
    h = torch.randn(t, cfg.hidden_size) * 0.1
    pos = torch.arange(t)

    absorbed = _absorbed_mla(attn, cfg, h, pos, src)
    reference = _ref_dense_mla(attn, cfg, h, pos, src)

    assert absorbed.shape == (t, cfg.hidden_size)
    # Pure fp32 algebra; residual comes only from op ordering.
    torch.testing.assert_close(absorbed, reference, rtol=1e-4, atol=1e-4)


def test_rope_position_zero_is_identity():
    attn, cfg, _ = _build_attention_cpu(seed=4)
    q_pe = torch.randn(1, attn.num_heads, cfg.qk_rope_head_dim)
    k_pe = torch.randn(1, 1, cfg.qk_rope_head_dim)
    q_rot, k_rot = attn.rotary(torch.zeros(1, dtype=torch.long), q_pe, k_pe)
    torch.testing.assert_close(q_rot, q_pe, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(k_rot, k_pe, rtol=1e-6, atol=1e-6)


def test_processing_releases_the_folded_sources_and_is_idempotent():
    """After fusing, q_a_proj / kv_a_proj_with_mqa / kv_b_proj are dead
    weight on the absorbed path (~2.8 GB per rank on the TP8 full model):
    their storage goes, the Parameters and the Linear metadata the forward
    still reads stay. A second call must not try to re-fold from nothing."""
    attn, cfg, src = _build_attention_cpu(seed=8)
    for lin in (attn.q_a_proj, attn.kv_a_proj_with_mqa, attn.kv_b_proj):
        assert lin.weight.numel() == 0
        assert isinstance(lin.weight, torch.nn.Parameter)
        assert lin.weight.dtype == src["q_a"].dtype
    assert attn.q_a_proj.out_features == cfg.q_lora_rank  # split point still read
    assert attn.absorbed_sources_released
    # the still-needed projections and the built buffers are intact
    assert attn.q_b_proj.weight.numel() > 0 and attn.o_proj.weight.numel() > 0
    assert attn.fused_qkv_a_proj_weight.numel() == (
        cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim) * cfg.hidden_size
    # w_kc / w_vc own their storage: nothing pins the released kv_b_proj
    assert attn.w_kc.untyped_storage().data_ptr() != attn.w_vc.untyped_storage().data_ptr()
    assert attn.w_kc.is_contiguous() and attn.w_vc.is_contiguous()

    w_kc, w_vc, fused = attn.w_kc, attn.w_vc, attn.fused_qkv_a_proj_weight
    attn.process_weights_after_loading()  # the root hook may visit twice
    assert attn.w_kc is w_kc and attn.w_vc is w_vc
    assert attn.fused_qkv_a_proj_weight is fused


def test_naive_path_keeps_its_projections():
    torch.manual_seed(9)
    cfg = Glm52ModelConfig.reduced()  # mla_absorb False
    attn = Glm52MLAAttention(cfg)
    before = _source_weights(attn)
    attn.process_weights_after_loading()  # no-op off the absorbed path
    assert torch.equal(attn.q_a_proj.weight, before["q_a"])
    assert torch.equal(attn.kv_a_proj_with_mqa.weight, before["kv_a"])
    assert torch.equal(attn.kv_b_proj.weight, before["kv_b"])
    assert not attn.mla_absorb


# ── both engine paths through their real resources ──


class _Path:
    """One attention layer bound to the node's resources for its
    ``mla_absorb`` value, driven a step at a time: declare -> admit ->
    plan -> layer forward -> commit, what the engine does around a
    forward.
    """

    def __init__(self, attn: Glm52MLAAttention, cfg: Glm52ModelConfig, page_size: int):
        self.attn, self.cfg = attn, cfg
        self.resources, self.runner = build_cpu_resources(cfg, ["r0"], page_size=page_size)
        attn.bind_resources(self.resources)

    def step(self, h: torch.Tensor, start: int) -> torch.Tensor:
        n = h.shape[0]
        attn_step = MlaAttentionStep() if self.cfg.mla_absorb else AttentionStep(causal=True)
        step = SubmoduleStep(
            segments=[Segment("r0", "main", n)],
            steps={KV_RESOURCE: KVStep(), ATTN_RESOURCE: attn_step},
        )
        step.set_ctx(StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        with torch.no_grad():
            out = self.attn(h, torch.arange(start, start + n))
        self.runner.commit(step)
        assert self.resources[KV_RESOURCE].stored_len("r0") == start + n
        return out


def _build_pair(seed):
    """The same weights on both paths: the naive layer's state dict loaded
    into an absorbed layer, whose w_kc / w_vc / fused_qkv_a are then built
    from it (non-persistent buffers, so the state dicts match)."""
    torch.manual_seed(seed)
    naive_cfg = Glm52ModelConfig.reduced()  # mla_absorb False
    naive = Glm52MLAAttention(naive_cfg)
    for lin in (naive.q_a_proj, naive.q_b_proj, naive.kv_a_proj_with_mqa,
                naive.kv_b_proj, naive.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (naive.q_a_layernorm, naive.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    absorbed_cfg = Glm52ModelConfig.reduced()
    absorbed_cfg.mla_absorb = True
    absorbed = Glm52MLAAttention(absorbed_cfg)
    absorbed.load_state_dict(naive.state_dict())
    absorbed.process_weights_after_loading()
    return (naive, naive_cfg), (absorbed, absorbed_cfg)


def test_absorbed_resource_path_matches_naive_and_dense_reference():
    """Prefill 7 tokens, then decode at positions 7, 8, 9 (page size 8: the last two land
    on a second page).
    """
    (naive, naive_cfg), (absorbed, absorbed_cfg) = _build_pair(seed=5)
    page = 8
    naive_path = _Path(naive, naive_cfg, page)
    absorbed_path = _Path(absorbed, absorbed_cfg, page)
    # reduced dims: the MLA resource is on its SDPA fallback, not the kernel
    assert not absorbed_path.resources[ATTN_RESOURCE].uses_kernel

    torch.manual_seed(6)
    total = 10
    h_all = torch.randn(total, naive_cfg.hidden_size) * 0.1
    with torch.no_grad():
        reference = _ref_dense_mla(
            naive, naive_cfg, h_all, torch.arange(total), _source_weights(naive))

    chunks = [(0, 7), (7, 8), (8, 9), (9, 10)]
    for lo, hi in chunks:
        out_naive = naive_path.step(h_all[lo:hi], lo)
        out_absorbed = absorbed_path.step(h_all[lo:hi], lo)
        assert out_naive.shape == out_absorbed.shape == (hi - lo, naive_cfg.hidden_size)
        torch.testing.assert_close(out_absorbed, out_naive, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out_absorbed, reference[lo:hi], rtol=1e-2, atol=2e-3)
        torch.testing.assert_close(out_naive, reference[lo:hi], rtol=1e-2, atol=2e-3)

    # what each cache holds: one 40-wide latent row per token on the
    # absorbed path (32 ckv + 8 rope), padded per-head K/V on the naive one
    kv_abs = absorbed_path.resources[KV_RESOURCE]
    kv_naive = naive_path.resources[KV_RESOURCE]
    assert kv_abs.layer_view(0).shape[-1] == absorbed_cfg.cache_latent_dim == 40
    assert kv_naive.layer_view(0).shape[1:] == (2, page, naive_cfg.num_attention_heads, 64)
    assert len(kv_abs._streams["r0"]["main"].page_indices) == 2


def test_absorbed_path_needs_the_processed_weights():
    # the naive twin still holds the full projections; the processed one
    # has released them, so its state dict is not a load source anymore
    (naive, _), (_, absorbed_cfg) = _build_pair(seed=7)
    fresh = Glm52MLAAttention(absorbed_cfg)
    fresh.load_state_dict(naive.state_dict())
    with pytest.raises(RuntimeError, match="process_weights_after_loading"):
        fresh(torch.randn(2, absorbed_cfg.hidden_size), torch.arange(2))


def test_full_model_head_geometry_needs_no_pad():
    cfg = Glm52ModelConfig()
    assert cfg.qk_head_dim == 256
    assert cfg.padded_head_dim == 256  # pad and softmax boost are no-ops
    attn_boost = (cfg.padded_head_dim / cfg.qk_head_dim) ** 0.5
    assert attn_boost == 1.0
