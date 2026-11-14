"""MLAAttention's options past GLM-5.2's layout, on CPU: no decoupled dims (``mla_use_nope``,
GLM-5.3's), whose entries are the normed latent alone, against plain attention from the same
weights, with the naive attention supplied through the class's hook and the absorbed one by a
stub handle; and the layer's own id reaching the handle's calls."""
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


class _StubLatentHandle:
    """Records the layer's calls, and keeps the step's latent entries and attends them densely."""

    def __init__(self, m: MLAAttention):
        self.m, self.calls, self.rows = m, [], None

    def add(self, layer, keep=None, **fields):
        self.calls.append(("add", None if layer is None else int(layer), tuple(fields)))
        self.rows = fields["ckv"]

    def read(self, layer, q_nope, **kwargs):
        self.calls.append(("read", None if layer is None else int(layer), tuple(kwargs)))
        scores = torch.einsum("qhl,kl->hqk", q_nope, self.rows) * self.m.softmax_scale
        return torch.einsum("hqk,kl->qhl", _causal_softmax(scores), self.rows)


def test_nope_means_no_decoupled_dims():
    with pytest.raises(ValueError, match="no decoupled dims"):
        MLAAttention(_config(False, mla_use_nope=True), kv_key="kv")


def test_nope_mla_rotates_nothing():
    m = _DenseNaive(_latent_only_config(False), kv_key="kv")
    q, k = torch.randn(T, HEADS, 0), torch.randn(T, 1, 0)
    q2, k2 = m._rope(torch.arange(T), q, k)
    assert q2 is q and k2 is k


def _latent_only_config(absorb: bool) -> SimpleNamespace:
    cfg = _config(absorb, mla_use_nope=True)
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


def test_latent_only_mla_matches_plain_attention():
    """GLM-5.3's layout: no decoupled dims, so an entry is the normed latent alone and the read
    takes no rope query."""
    naive = _DenseNaive(_latent_only_config(False), kv_key="kv")
    with torch.no_grad():
        for name, p in naive.named_parameters():
            p.copy_(1.0 + 0.1 * torch.randn_like(p) if "layernorm" in name
                    else 0.1 * torch.randn_like(p))
    params = {k: v.clone() for k, v in naive.state_dict().items()}
    absorbed = MLAAttention(_latent_only_config(True), kv_key="kv")
    absorbed.load_state_dict(params, strict=False)
    handle = _StubLatentHandle(absorbed)
    absorbed.bind_state({"kv": handle})
    x = torch.randn(T, HIDDEN)
    with torch.no_grad():
        ref = _mha_reference(naive, x)
        absorbed.process_weights_after_loading()
        torch.testing.assert_close(naive(x, None), ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(absorbed(x, None), ref, rtol=1e-5, atol=1e-6)
    assert handle.rows.shape == (T, KV_LORA), "an entry with no decoupled dims is the latent alone"


def test_the_layer_adds_and_reads_at_its_own_id():
    """The absorbed path adds the step's latents and reads them at the layer's id, with no rope
    query where there are no decoupled dims."""
    m = MLAAttention(_latent_only_config(True), layer_idx=7, kv_key="kv")
    m.process_weights_after_loading()
    handle = _StubLatentHandle(m)
    m.bind_state({"kv": handle})
    with torch.no_grad():
        m(torch.randn(T, HIDDEN), None)
    assert handle.calls == [("add", 7, ("ckv",)), ("read", 7, ())], handle.calls
