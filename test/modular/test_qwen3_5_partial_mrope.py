"""Qwen3.5's partial MRoPE, written over the whole head, against the slice-and-concatenate form.

The whole-head form must give the same bits: the rotated dims run the same ops, and the passed
dims see ``x * 1 + 0 * 0``.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_5.components import language_model, vision
from mstar.model.qwen3_5.components.language_model import Qwen3_5LanguageModel
from mstar.model.qwen3_5.components.rope import apply_partial_mrope, compute_inv_freq
from mstar.model.qwen3_5.components.vision import VisionRotaryEmbedding
from mstar.model.qwen3_5.config import FULL_ATTENTION, Qwen3_5Config


def _reference(q, k, cos, sin):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    r = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    out = []
    for x in (q, k):
        rot = x[..., :r] * cos + rotate_half(x[..., :r]) * sin
        out.append(torch.cat((rot, x[..., r:]), dim=-1))
    return tuple(out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("tokens", "rotary_dim"), [(1, 64), (7, 64), (33, 128)])
def test_whole_head_form_is_bit_exact(dtype, tokens, rotary_dim):
    g = torch.Generator().manual_seed(tokens + rotary_dim)
    q = torch.randn(tokens, 8, 256, generator=g).to(dtype)
    k = torch.randn(tokens, 2, 256, generator=g).to(dtype)
    angles = torch.rand(tokens, rotary_dim // 2, generator=g) * 100
    emb = torch.cat((angles, angles), dim=-1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    for got, want in zip(apply_partial_mrope(q, k, cos, sin), _reference(q, k, cos, sin), strict=True):
        assert torch.equal(got, want)


def _tiny_language_model() -> Qwen3_5LanguageModel:
    config = Qwen3_5Config(
        num_hidden_layers=1, hidden_size=64, intermediate_size=128, layer_types=[FULL_ATTENTION],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, vocab_size=16, rms_norm_eps=1e-6, max_position_embeddings=64,
        rope_theta=10_000_000.0, mrope_section=[2, 1, 1], mrope_interleaved=True, partial_rotary_factor=0.25,
    )
    return Qwen3_5LanguageModel(config)


def test_rope_frequencies_survive_the_dtype_cast_on_rocm(monkeypatch):
    monkeypatch.setattr(language_model, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    lm = _tiny_language_model().to(dtype=torch.bfloat16)
    want = compute_inv_freq(lm.config.rotary_dim, lm.config.rope_theta)
    assert lm.inv_freq.dtype is torch.float32
    assert torch.equal(lm.inv_freq, want)
    # far along a long prompt a rounded frequency lands on another angle
    position = 20_000.0
    cos, _ = lm.build_cos_sin(torch.full((3, 1), position), torch.float32)
    assert torch.equal(cos[0, 1], (want[1] * position).cos())


def test_rope_frequencies_keep_the_cast_elsewhere(monkeypatch):
    monkeypatch.setattr(language_model, "current_platform", SimpleNamespace(is_rocm=lambda: False))
    lm = _tiny_language_model().to(dtype=torch.bfloat16)
    assert lm.inv_freq.dtype is torch.bfloat16


def _vision_rope() -> VisionRotaryEmbedding:
    return VisionRotaryEmbedding(SimpleNamespace(head_dim=72, rope_theta=10000.0))


def test_vision_rope_frequencies_are_upstreams():
    dim = 36
    want = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    assert torch.equal(_vision_rope().inv_freq, want)


def test_vision_rope_frequencies_survive_the_dtype_cast_on_rocm(monkeypatch):
    monkeypatch.setattr(vision, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    rope = _vision_rope().to(dtype=torch.bfloat16)
    want = compute_inv_freq(36, 10000.0)
    assert rope.inv_freq.dtype is torch.float32
    assert torch.equal(rope.inv_freq, want)
    position_ids = torch.tensor([[0, 255]])
    cos, _ = rope(position_ids)
    assert torch.equal(cos[0, 18:36], (255.0 * want).cos())


def test_vision_rope_frequencies_keep_the_cast_elsewhere(monkeypatch):
    monkeypatch.setattr(vision, "current_platform", SimpleNamespace(is_rocm=lambda: False))
    rope = _vision_rope().to(dtype=torch.bfloat16)
    assert rope.inv_freq.dtype is torch.bfloat16
