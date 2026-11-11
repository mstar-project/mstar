"""Ministral attention contracts and tiny CPU parity; no pretrained weights or GPU."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.position.config import PositionConfig, YaRNConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.ministral3.components.attention import Ministral3Attention


def attention(**overrides):
    config = dict(hidden_size=96, num_heads=4, num_kv_heads=2, head_dim=32,
                  original_max_position_embeddings=16384)
    return Ministral3Attention(**(config | overrides))


class ShiftPosition:
    """A non-identity test transform makes use of the resource's output observable."""

    def __init__(self, positions):
        self.positions = positions
        self.labels = []

    def apply_qk(self, q, k, *, label):
        self.labels.append(label)
        return q + 2, k - 3

    def pos_ids(self, label):
        assert label == "main"
        return self.positions


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scale_uses_each_rows_position_after_rotation(dtype):
    pos = ShiftPosition(torch.tensor([0, 16383, 16384, 32768, 262143, 999999]))
    model = attention()
    model.bind_resources(dict(attn=object(), kv=object(), rope=pos))
    q = torch.ones(5, 4, 32, dtype=dtype)
    k = torch.ones(5, 2, 32, dtype=dtype)
    actual_q, actual_k = model._apply_rope(q, k, "main")
    # Known boundary buckets, including a padded position tail not represented in Q.
    scale = (1 + 0.1 * torch.log(torch.tensor([1., 1., 2., 3., 16.]))).to(dtype)
    torch.testing.assert_close(actual_q, (q + 2) * scale[:, None, None], rtol=0, atol=0)
    assert torch.equal(actual_k, k - 3)
    assert pos.labels == ["main"]
    assert not torch.equal(actual_q[2:], (q + 2)[2:])  # omitted scaling must fail


def test_scaling_hook_traces_as_one_tensor_graph():
    model = attention()
    pos = ShiftPosition(torch.tensor([16383, 16384]))
    model.bind_resources(dict(attn=object(), kv=object(), rope=pos))
    q, k = torch.randn(2, 4, 32), torch.randn(2, 2, 32)
    expected = model._apply_rope(q, k, "main")
    compiled = torch.compile(model._apply_rope, backend="eager", fullgraph=True)
    for actual, target in zip(compiled(q, k, "main"), expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)


def test_real_geometry_on_meta_has_no_qk_norms_or_biases():
    with torch.device("meta"):
        model = attention(hidden_size=3072, num_heads=32, num_kv_heads=8, head_dim=128)
    assert model.qkv_proj.weight.shape == (6144, 3072)
    assert model.o_proj.weight.shape == (3072, 4096)
    assert model.q_norm is None and model.k_norm is None
    assert model.qkv_proj.bias is None and model.o_proj.bias is None


@pytest.mark.parametrize("key", ["attn", "kv", "rope"])
def test_missing_resource_is_refused(key):
    resources = dict(attn=object(), kv=object(), rope=object())
    del resources[key]
    with pytest.raises(ValueError, match=key):
        attention().bind_resources(resources)


def test_missing_positions_are_refused():
    model = attention()
    model.bind_resources(dict(attn=object(), kv=object(), rope=ShiftPosition(None)))
    with pytest.raises(RuntimeError, match="no planned positions"):
        model._apply_rope(torch.ones(1, 4, 32), torch.ones(1, 2, 32), "main")


class DenseAttention:
    """Test-only causal attention over one prefill, without a persistent KV cache."""

    requires_kv_write = False

    def run(self, q, kv_cache_layer, k, v):
        del kv_cache_layer
        repeats = q.shape[1] // k.shape[1]
        q = q.transpose(0, 1)
        k = k.repeat_interleave(repeats, dim=1).transpose(0, 1)
        v = v.repeat_interleave(repeats, dim=1).transpose(0, 1)
        logits = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
        mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).triu(1)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return (logits.softmax(-1) @ v).transpose(0, 1)


def cpu_table_rotation(q, k, positions, table, *args, **kwargs):
    """Stand in only for the GPU rotation kernel; use the real resource's table."""
    cos, sin = table[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    for x in (q, k):
        a, b = x.chunk(2, dim=-1)
        x.copy_(torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1))


@pytest.mark.parametrize("start", [0, 16382, 32766])
def test_forward_matches_transformers_with_yarn_and_query_scaling(monkeypatch, start):
    pytest.importorskip("transformers.models.ministral3.modeling_ministral3")
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
    from transformers.models.ministral3.modeling_ministral3 import (
        Ministral3Attention as ReferenceAttention,
    )
    from transformers.models.ministral3.modeling_ministral3 import (
        Ministral3RotaryEmbedding,
    )

    torch.manual_seed(42)
    config = Ministral3Config(hidden_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=32)
    config._attn_implementation = "eager"
    reference = ReferenceAttention(config, layer_idx=0).eval()
    model = attention().eval()
    for name in ("q", "k", "v"):
        model.qkv_proj.weight_loader(model.qkv_proj.weight, getattr(reference, f"{name}_proj").weight, name)
    model.o_proj.weight_loader(model.o_proj.weight, reference.o_proj.weight)

    positions = torch.arange(start, start + 4)
    pos = RopeManager(PositionConfig(kv_cache="kv", rotary_dim=32, rope_theta=1e6,
                                    rope_dtype=torch.float32,
                                    yarn=YaRNConfig(factor=16, original_max_position_embeddings=16384)),
                      torch.device("cpu"), max_seq_len=start + 4)
    pos._current_pos_ids["main"] = positions
    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", cpu_table_rotation)
    kv = SimpleNamespace(default_label="main", layer_view=lambda: None)
    model.bind_resources(dict(attn=DenseAttention(), kv=kv, rope=pos))
    hidden = torch.randn(1, 4, 96)
    mask = torch.zeros(1, 1, 4, 4).masked_fill(
        torch.ones(4, 4, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min,
    )
    with torch.no_grad():
        embeddings = Ministral3RotaryEmbedding(config)(hidden, positions[None])
        expected = reference(hidden, position_embeddings=embeddings, attention_mask=mask,
                             position_ids=positions[None])[0][0]
        actual = model(hidden[0])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if start > 0:
            model.llama_4_scaling_beta = 0
            assert not torch.allclose(model(hidden[0]), expected, atol=1e-6, rtol=1e-5)
