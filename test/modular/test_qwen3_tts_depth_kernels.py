"""Qwen3-TTS's depth loop: the fused q/k norm + RoPE + KV write (``qk_norm_rope_kv_write``) with its guard and eager
fallback.

The CPU tests stand a Python reference in for the Triton kernels and pin the selection and the guard. The GPU tests hold
the kernels to the eager path at bf16 tolerance on the predictor's own geometry (16 groups, 16 q and 8 KV heads of 128,
rows 1/8/28).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import mstar.model.qwen3_tts.components.talker as talker_module
from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor
from mstar.model.qwen3_tts.config import (
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.utils import attention

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA (Triton + FlashInfer)")

GROUPS = 4


def _config(layers=2, heads=4, kv_heads=2, head_dim=8, hidden=16, groups=GROUPS) -> Qwen3TTSModelConfig:
    cp = Qwen3TTSCodePredictorConfig(
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads, hidden_size=hidden,
        intermediate_size=2 * hidden, head_dim=head_dim, vocab_size=32, num_code_groups=groups,
    )
    talker = Qwen3TTSTalkerConfig(
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, hidden_size=2 * hidden,
        intermediate_size=32, head_dim=hidden, vocab_size=64, text_hidden_size=16, text_vocab_size=128,
        num_code_groups=groups, code_predictor=cp,
    )
    return Qwen3TTSModelConfig(talker=talker)


def _init_weights(predictor) -> None:
    """The parallel linears start uninitialized."""
    for name, param in predictor.named_parameters():
        if "norm" in name:
            torch.nn.init.normal_(param, 1.0, 0.1)
        elif param.dim() > 1:
            torch.nn.init.normal_(param, 0.0, 0.02)
        else:
            torch.nn.init.zeros_(param)


def _reference_rope(q, k, position_ids, rope_theta):
    """``apply_rope_pos_ids`` in Python, returning new tensors."""

    def rotate(x):
        half = x.shape[-1] // 2
        inv = rope_theta ** (-torch.arange(0, x.shape[-1], 2, dtype=torch.float32) / x.shape[-1])
        angle = position_ids.float()[:, None, None] * inv
        a, b = x[..., :half], x[..., half:]
        return torch.cat([a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()], dim=-1)

    return rotate(q), rotate(k)


def _reference_attention(q, k_cache, v_cache, cache_len):
    """``decode_attn_nhd`` in Python: token t of T over the cache up to and including itself."""
    tokens = q.shape[1]
    rows = []
    for t in range(tokens):
        n = cache_len - tokens + t + 1
        rows.append(F.scaled_dot_product_attention(
            q[:, t:t + 1].transpose(1, 2), k_cache[:, :n].transpose(1, 2), v_cache[:, :n].transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2))
    return torch.cat(rows, dim=1)


@pytest.fixture
def python_kernels(monkeypatch):
    monkeypatch.setattr(talker_module, "apply_rope_pos_ids", _reference_rope)
    monkeypatch.setattr(talker_module, "decode_attn_nhd", _reference_attention)


def _cache(config, batch):
    cp = config.talker.code_predictor
    return torch.zeros(cp.num_hidden_layers, batch, 2, GROUPS, cp.num_key_value_heads, cp.head_dim)


def _fake_tensors(head_dim=128, dtype=torch.bfloat16, cache_dtype=None, q_stride=1, cache_stride=1, device_type="cuda"):
    device = SimpleNamespace(type=device_type)
    q = SimpleNamespace(shape=(4, 16, head_dim), device=device, dtype=dtype, stride=lambda _dim: q_stride)
    cache = SimpleNamespace(dtype=cache_dtype or dtype, stride=lambda _dim: cache_stride)
    return q, cache


@pytest.fixture
def blackwell(monkeypatch):
    seen = []

    def has_capability(major, minor, device):
        seen.append((major, minor))
        return (major, minor) <= (10, 3)

    monkeypatch.setattr(attention, "current_platform", SimpleNamespace(has_cuda_capability=has_capability))
    return seen


def test_guard_takes_what_the_predictor_runs(blackwell):
    assert attention.qk_norm_rope_kv_write_applies(*_fake_tensors())
    assert attention.qk_norm_rope_kv_write_applies(*_fake_tensors(dtype=torch.float32))
    assert attention.qk_norm_rope_kv_write_applies(*_fake_tensors(head_dim=64))
    assert blackwell == [(10, 0)] * 3


@pytest.mark.parametrize("kwargs", [
    dict(head_dim=72),                     # tl.arange can't span a half head that is no power of two
    dict(head_dim=1),
    dict(dtype=torch.float64),
    dict(cache_dtype=torch.float32),       # the kernel stores k and v in the cache's dtype, q in its own
    dict(q_stride=2),                      # heads have to be contiguous
    dict(cache_stride=2),
    dict(device_type="cpu"),
], ids=str)
def test_guard_refuses(blackwell, kwargs):
    assert not attention.qk_norm_rope_kv_write_applies(*_fake_tensors(**kwargs))


def test_guard_refuses_an_older_gpu(monkeypatch):
    platform = SimpleNamespace(has_cuda_capability=lambda major, minor, device: False)
    monkeypatch.setattr(attention, "current_platform", platform)
    assert not attention.qk_norm_rope_kv_write_applies(*_fake_tensors())


def _attend_inputs(config, batch):
    cp = config.talker.code_predictor
    predictor = Qwen3TTSCodePredictor(config)
    _init_weights(predictor)
    attn = predictor.model.layers[0].self_attn
    q = torch.randn(batch, cp.num_attention_heads, cp.head_dim)
    k = torch.randn(batch, cp.num_key_value_heads, cp.head_dim)
    v = torch.randn(batch, cp.num_key_value_heads, cp.head_dim)
    return predictor, attn, q, k, v, _cache(config, batch)[0]


def test_applies_selects_the_one_kernel(monkeypatch):
    config = _config()
    batch = 3
    predictor, attn, q, k, v, kv_layer = _attend_inputs(config, batch)
    positions, rows = torch.full((batch,), 2), torch.arange(batch)
    calls = []

    def fake_kernel(*args):
        calls.append(args)

    monkeypatch.setattr(talker_module, "qk_norm_rope_kv_write_applies", lambda q, layer: True)
    monkeypatch.setattr(talker_module, "qk_norm_rope_kv_write", fake_kernel)
    monkeypatch.setattr(talker_module, "decode_attn_nhd", _reference_attention)
    monkeypatch.setattr(talker_module, "apply_rope_pos_ids", lambda *a: pytest.fail("the eager RoPE ran"))
    with torch.no_grad():
        predictor._norm_rope_attend(attn, q, k, v, positions, rows, kv_layer, 2, 3)

    (args,) = calls
    assert args[0] is q and args[1] is k and args[2] is v
    assert args[3] is attn.q_norm.weight and args[4] is attn.k_norm.weight
    # a token's depth position is also its cache slot
    assert args[5] is positions and args[8] is positions
    assert args[6] is kv_layer and args[7] is rows and args[9] == batch
    assert args[12] == attn.rope_theta


def test_fallback_is_the_eager_norm_rope_and_write(monkeypatch, python_kernels):
    config = _config()
    batch = 2
    predictor, attn, q, k, v, kv_layer = _attend_inputs(config, batch)
    positions, rows = torch.full((batch,), 1), torch.arange(batch)
    monkeypatch.setattr(talker_module, "qk_norm_rope_kv_write_applies", lambda q, layer: False)
    monkeypatch.setattr(talker_module, "qk_norm_rope_kv_write", lambda *a: pytest.fail("the fused kernel ran"))
    with torch.no_grad():
        out = predictor._norm_rope_attend(attn, q, k.clone(), v, positions, rows, kv_layer, 1, 2)
        normed = attn.k_norm(k.reshape(-1, k.shape[-1])).view(k.shape)
        _, want_k = _reference_rope(normed, normed, positions, attn.rope_theta)
    assert out.shape == (batch, 1, q.shape[1], q.shape[2])
    torch.testing.assert_close(kv_layer[:, 0, 1], want_k)
    torch.testing.assert_close(kv_layer[:, 1, 1], v)
    assert kv_layer[:, :, 0].abs().sum() == 0 and kv_layer[:, :, 2:].abs().sum() == 0


# ---------------------------------------------------------------------------
# The kernels on the real geometry.
# ---------------------------------------------------------------------------

HQ, HKV, D, EPS, THETA = 16, 8, 128, 1e-6, 1_000_000.0
ROWS = [1, 8, 28]


def _assert_bf16_close(got, want, min_equal=0.5, steps=1.0):
    """Within ``steps`` bf16 steps of the eager result, taken at unit scale too since
    an output that cancels toward zero keeps the absolute error of its O(1) terms."""
    assert got.shape == want.shape and got.dtype == want.dtype
    step = (want.float().abs() + 1.0) * 2.0**-7
    assert ((got.float() - want.float()).abs() <= steps * step).all()
    assert (got == want).float().mean().item() >= min_equal


def _qkv(rows, tokens, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qkv = torch.randn(rows * tokens, (HQ + 2 * HKV) * D, generator=g).cuda().bfloat16()
    wq = (torch.rand(D, generator=g) * 0.5 + 0.75).cuda().bfloat16()
    wk = (torch.rand(D, generator=g) * 0.5 + 0.75).cuda().bfloat16()
    return qkv, wq, wk


def _split(qkv):
    n = qkv.shape[0]
    q, k, v = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
    return q.view(n, HQ, D), k.view(n, HKV, D), v.view(n, HKV, D)


@needs_cuda
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("tokens,cache_pos", [(1, 0), (1, 2), (1, 15)])
def test_kernel_matches_eager_norm_rope_write(rows, tokens, cache_pos):
    from mstar.engine.resources.rms_norm import run_rms_norm

    qkv, wq, wk = _qkv(rows, tokens)
    positions = (cache_pos + torch.arange(tokens, device="cuda")).repeat(rows)
    batch_rows = torch.arange(rows, device="cuda").repeat_interleave(tokens)
    cache = torch.zeros(rows, 2, 16, HKV, D, device="cuda", dtype=torch.bfloat16)

    q, k, v = _split(qkv.clone())
    want_q, want_k = attention.apply_rope_pos_ids(
        run_rms_norm(q.reshape(-1, D), wq, EPS).view(q.shape),
        run_rms_norm(k.reshape(-1, D), wk, EPS).view(k.shape),
        positions.to(torch.int32), THETA,
    )

    q, k, v = _split(qkv.clone())
    assert attention.qk_norm_rope_kv_write_applies(q, cache)
    attention.qk_norm_rope_kv_write(
        q, k, v, wq, wk, positions, cache, batch_rows, positions, q.shape[0], EPS, EPS, THETA,
    )
    _assert_bf16_close(q, want_q)
    end = cache_pos + tokens
    _assert_bf16_close(cache[:, 0, cache_pos:end], want_k.view(rows, tokens, HKV, D))
    assert torch.equal(cache[:, 1, cache_pos:end], v.view(rows, tokens, HKV, D))
    # nothing else in the frame's cache moved
    assert cache[:, :, :cache_pos].abs().sum() == 0 and cache[:, :, end:].abs().sum() == 0


def _predictor(rows):
    config = _config(layers=5, heads=HQ, kv_heads=HKV, head_dim=D, hidden=1024, groups=16)
    torch.manual_seed(0)
    predictor = Qwen3TTSCodePredictor(config)
    _init_weights(predictor)
    predictor = predictor.cuda().bfloat16()
    return predictor, torch.zeros(5, rows, 2, 16, HKV, D, device="cuda", dtype=torch.bfloat16)


@needs_cuda
@pytest.mark.parametrize("rows", ROWS)
def test_depth_passes_match_the_eager_passes(rows, monkeypatch):
    """All 16 positions through five real-geometry layers, the fused path against the eager one."""
    predictor, cache = _predictor(rows)
    g = torch.Generator(device="cpu").manual_seed(3)
    inputs = torch.randn(rows, 16, 2048, generator=g).cuda().bfloat16()
    batch_rows = torch.arange(rows, device="cuda")

    def run(kv_cache):
        return [
            predictor.forward_depth_unrolled(
                inputs[:, pos:pos + 1].clone(), batch_rows * 0 + pos, batch_rows, kv_cache, cache_pos=pos,
            )[:, 0]
            for pos in range(16)
        ]

    with torch.inference_mode():
        fused = run(cache)
        monkeypatch.setattr(talker_module, "qk_norm_rope_kv_write_applies", lambda q, layer: False)
        eager = run(cache.clone().zero_())

    for got, want in zip(fused, eager, strict=True):
        # five layers of rounding, each at most a step or so
        _assert_bf16_close(got, want, min_equal=0.0, steps=8.0)
