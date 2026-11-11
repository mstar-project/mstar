"""``write_kv``: the K/V an attention layer's ``forward`` would write, without attending."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.model.components.attention import Attention
from mstar.model.components.distributed.attention import ParallelAttention


class _KV:
    """A KV resource that records what it is asked to write."""

    default_label = "main"

    def __init__(self):
        self.written = []

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def layer_view(self):
        return None

    def write_kv(self, k, v):
        self.written.append((k.clone(), v.clone()))


class _Attn:
    requires_kv_write = True

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def run(self, q, kv_cache_layer=None, k=None, v=None):
        return torch.zeros_like(q)


class _Pos:
    """1D RoPE stand-in: a per-token rotation of q and k, so a write that skips it differs."""

    def apply_qk(self, q, k, label=None, **kwargs):
        scale = torch.arange(1, q.shape[0] + 1, dtype=q.dtype)[:, None, None]
        return q * scale, k * scale


@pytest.mark.parametrize("cls", [Attention, ParallelAttention], ids=lambda c: c.__name__)
def test_write_kv_writes_what_forward_writes(cls):
    torch.manual_seed(0)
    kwargs = dict(hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=True,
                  attn_key="attn", kv_key="kv", pos_key="rope")
    layer = cls(**kwargs) if cls is Attention else cls(comm_group=None, **kwargs)
    # ParallelAttention's projections come from torch.empty; NaN would make torch.equal fail at random
    with torch.no_grad():
        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.02)
    kv = _KV()
    layer.bind_resources({"attn": _Attn(), "kv": kv, "rope": _Pos()})
    x = torch.randn(5, 32)
    with torch.no_grad():
        layer(x)
        layer.write_kv(x)
    (k_fwd, v_fwd), (k_w, v_w) = kv.written
    assert torch.equal(k_fwd, k_w) and torch.equal(v_fwd, v_w)


def test_a_stacks_write_kv_moves_each_layers_cursor_to_its_own_index():
    from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig

    config = Qwen3LMConfig(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
                           rope_theta=1e4, max_position_embeddings=64)
    lm = Qwen3DenseLM(config, attn_key="attn", kv_key="kv", pos_key="rope")
    seen = []
    kv = _KV()
    kv.set_default_layer_idx = seen.append
    for layer in lm.layers:
        layer.self_attn.bind_resources({"attn": _Attn(), "kv": kv, "rope": SimpleNamespace(
            apply_qk=_Pos().apply_qk, fusable_rope_theta=lambda head_dim: None)})
    with torch.no_grad():
        lm.write_kv(torch.randn(4, 32), label="main")
    assert seen == [0, 1, 2] and len(kv.written) == 3
