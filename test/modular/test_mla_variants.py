"""MLAAttention's options against Kimi K3's ``KimiMLAAttention`` (``modeling_kimi_linear.py``, HF
f831ab66814), transcribed: the decoupled dims are never rotated (``mla_use_nope``), the output is
gated by ``sigmoid(g_proj(x))`` (``mla_use_output_gate``) and the latent norms are ``KimiRMSNorm``'s
default 1e-6 (``mla_latent_norm_eps``). Both the naive and the absorbed path run on CPU, with the
attention itself supplied through the class's hooks (as a sparse attention plugs in), so no KV or
attention resource is needed. The same for GLM-5.3's layout, no decoupled dims at all, whose cache
rows carry a zero rope slot."""
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from mstar.model.components.mla import MLAAttention

HIDDEN, HEADS, Q_LORA, KV_LORA, NOPE, ROPE, V = 64, 4, 32, 32, 16, 8, 16
T = 7


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


@pytest.fixture(autouse=True)
def _cpu_flashinfer(monkeypatch):
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    torch.manual_seed(0)


def _config(absorb: bool, **extra) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=HIDDEN, num_attention_heads=HEADS, q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, qk_head_dim=NOPE + ROPE, v_head_dim=V,
        padded_head_dim=32, rms_norm_eps=1e-5, softmax_scale=(NOPE + ROPE) ** -0.5,
        mla_absorb=absorb, **extra)


def _causal_softmax(scores):
    t = scores.shape[-1]
    mask = torch.ones(t, t, dtype=torch.bool).triu(1)
    return scores.masked_fill(mask, float("-inf")).softmax(dim=-1)


class _DenseNaive(MLAAttention):
    def _attend_naive(self, q, k, v, **sparse):
        # q carries softmax_scale * sqrt(padded); the backend divides by sqrt(padded)
        scores = torch.einsum("qhd,khd->hqk", q, k) / self.padded_head_dim ** 0.5
        return torch.einsum("hqk,khd->qhd", _causal_softmax(scores), v)


class _DenseAbsorbed(MLAAttention):
    def _select(self, hidden_states, q_c, position_ids, **sparse):
        return "every row"

    def _attend_selected(self, selection, q_nope, q_pe, latent, **sparse):
        kv_c, k_pe = latent.split([self.kv_lora_rank, self.mla_cache_kpe], dim=-1)
        scores = (torch.einsum("qhl,kl->hqk", q_nope, kv_c)
                  + torch.einsum("qhr,kr->hqk", q_pe, k_pe)) * self.softmax_scale
        self.rows = latent
        return torch.einsum("hqk,kl->qhl", _causal_softmax(scores), kv_c)


def _k3_reference(params, x, eps=1e-6):
    """KimiMLAAttention.forward, eager attention, one sequence. Its latent norms are
    ``KimiRMSNorm(q_lora_rank)`` / ``KimiRMSNorm(kv_lora_rank)``: eps 1e-6, not rms_norm_eps."""
    def rms(v, w):
        v32 = v.float()
        return w * (v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + eps)).to(v.dtype)

    q = rms(x @ params["q_a_proj.weight"].T, params["q_a_layernorm.weight"])
    q = (q @ params["q_b_proj.weight"].T).view(T, HEADS, NOPE + ROPE)
    q_pass, q_rot = q.split([NOPE, ROPE], dim=-1)
    compressed = x @ params["kv_a_proj_with_mqa.weight"].T
    k_pass, k_rot = compressed.split([KV_LORA, ROPE], dim=-1)
    kv = (rms(k_pass, params["kv_a_layernorm.weight"]) @ params["kv_b_proj.weight"].T)
    k_pass, value = kv.view(T, HEADS, NOPE + V).split([NOPE, V], dim=-1)
    k_rot = k_rot.view(T, 1, ROPE).expand(T, HEADS, ROPE)
    query, key = torch.cat([q_pass, q_rot], -1), torch.cat([k_pass, k_rot], -1)
    scores = torch.einsum("qhd,khd->hqk", query, key) * (NOPE + ROPE) ** -0.5
    out = torch.einsum("hqk,khd->qhd", _causal_softmax(scores), value).reshape(T, HEADS * V)
    out = out * torch.sigmoid(x @ params["g_proj.weight"].T)
    return out @ params["o_proj.weight"].T


def _module(cls, absorb, params=None):
    m = cls(_config(absorb, mla_use_nope=True, mla_use_output_gate=True, mla_latent_norm_eps=1e-6),
            layer_idx=0, kv_key="kv", attn_key="attn", pos_key=None)
    if params is None:
        with torch.no_grad():
            for name, p in m.named_parameters():
                p.copy_(1.0 + 0.1 * torch.randn_like(p) if "layernorm" in name
                        else 0.1 * torch.randn_like(p))
    else:
        m.load_state_dict(params, strict=False)
    return m


def test_k3_mla_naive_and_absorbed_match_the_reference():
    naive = _module(_DenseNaive, absorb=False)
    params = {k: v.clone() for k, v in naive.state_dict().items()}
    absorbed = _module(_DenseAbsorbed, absorb=True, params=params)
    absorbed.process_weights_after_loading()
    x = torch.randn(T, HIDDEN)
    pos = torch.arange(T)
    ref = _k3_reference(params, x)
    with torch.no_grad():
        torch.testing.assert_close(naive(x, pos), ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(absorbed(x, pos), ref, rtol=1e-5, atol=1e-6)


def test_the_gate_and_nope_are_what_change_the_output():
    m = _module(_DenseNaive, absorb=False)
    x, pos = torch.randn(T, HIDDEN), torch.arange(T)
    with torch.no_grad():
        gated = m(x, pos)
        g = m.g_proj
        m.g_proj = None
        ungated = m(x, pos)
        m.g_proj = g
    assert not torch.allclose(gated, ungated)


def test_defaults_are_the_rotating_ungated_mla():
    m = MLAAttention(_config(False), kv_key="kv", attn_key="attn", pos_key="pos")
    assert m.use_rope and m.g_proj is None
    assert m.q_a_layernorm.variance_epsilon == m.kv_a_layernorm.variance_epsilon == 1e-5
    with pytest.raises(ValueError, match="pos_key"):
        MLAAttention(_config(False), kv_key="kv", attn_key="attn", pos_key=None)
    eps = MLAAttention(_config(False, mla_latent_norm_eps=1e-6), kv_key="kv", attn_key="attn",
                       pos_key="pos")
    assert eps.q_a_layernorm.variance_epsilon == eps.kv_a_layernorm.variance_epsilon == 1e-6


def test_unrotated_mla_binds_no_position_resource():
    m = _module(_DenseNaive, absorb=False)
    m.bind_resources({"kv": object(), "attn": object()})
    q, k = torch.randn(T, HEADS, ROPE), torch.randn(T, 1, ROPE)
    q2, k2 = m._rope(torch.arange(T), q, k)
    assert q2 is q and k2 is k


def _latent_only_config(absorb: bool, kpe: int) -> SimpleNamespace:
    cfg = _config(absorb, mla_use_nope=True, mla_cache_kpe=kpe)
    cfg.qk_rope_head_dim, cfg.qk_head_dim = 0, NOPE
    cfg.softmax_scale = NOPE ** -0.5
    return cfg


def _mha_reference(m, x):
    """Plain multi-head attention from the module's own weights: no decoupled dims, no gate."""
    q = m.q_b_proj(m.q_a_layernorm(m.q_a_proj(x))).view(T, HEADS, NOPE)
    kv = m.kv_b_proj(m.kv_a_layernorm(m.kv_a_proj_with_mqa(x))).view(T, HEADS, NOPE + V)
    k, v = kv.split([NOPE, V], dim=-1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * NOPE ** -0.5
    out = torch.einsum("hqk,khd->qhd", _causal_softmax(scores), v).reshape(T, HEADS * V)
    return m.o_proj(out)


@pytest.mark.parametrize("kpe", [0, 4])
def test_latent_only_mla_matches_plain_attention(kpe):
    """GLM-5.3's layout: no decoupled dims, so the cache rows are the normed latent and a zero
    rope slot ``mla_cache_kpe`` wide, and the query's rope part is ``kpe_query_fill``."""
    naive = _DenseNaive(_latent_only_config(False, kpe), kv_key="kv", attn_key="attn", pos_key=None)
    with torch.no_grad():
        for name, p in naive.named_parameters():
            p.copy_(1.0 + 0.1 * torch.randn_like(p) if "layernorm" in name
                    else 0.1 * torch.randn_like(p))
    params = {k: v.clone() for k, v in naive.state_dict().items()}
    absorbed = _DenseAbsorbed(_latent_only_config(True, kpe), kv_key="kv", attn_key="attn",
                              pos_key=None)
    absorbed.load_state_dict(params, strict=False)
    absorbed.kpe_query_fill = 1.0  # adds 0 against the zero slot
    x = torch.randn(T, HIDDEN)
    with torch.no_grad():
        ref = _mha_reference(naive, x)
        absorbed.process_weights_after_loading()
        torch.testing.assert_close(naive(x, None), ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(absorbed(x, None), ref, rtol=1e-5, atol=1e-6)
    assert absorbed.rows.shape == (T, KV_LORA + kpe)
    assert torch.equal(absorbed.rows[:, KV_LORA:], torch.zeros(T, kpe))


class _Recorder:
    """A KV and an MLA attention resource that record what the layer hands them."""

    def __init__(self):
        self.calls = []

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.calls.append(("write", layer_idx, label, k.shape))

    def layer_view(self, layer_idx):
        return layer_idx

    def run(self, q, label=None, kv_cache_layer=None, q_pe=None):
        self.calls.append(("run", kv_cache_layer, label, q_pe.unique().tolist()))
        return q.new_zeros(q.shape)


def test_a_packed_plane_and_a_label_reach_the_resources():
    """``cache_layer`` addresses the KV layer in place of the layer's own index, and
    ``kv_label`` names the stream every write and attention runs under."""
    m = MLAAttention(_latent_only_config(True, 4), layer_idx=7, kv_key="kv", attn_key="attn",
                     pos_key=None, kv_label="main", cache_layer=1)
    m.process_weights_after_loading()
    rec = _Recorder()
    m.bind_resources({"kv": rec, "attn": rec})
    assert m.cache_layer_idx == 1
    with torch.no_grad():
        m(torch.randn(T, HIDDEN), None)
    assert rec.calls == [("write", 1, "main", (T, KV_LORA + 4)), ("run", 1, "main", [0.0])]
