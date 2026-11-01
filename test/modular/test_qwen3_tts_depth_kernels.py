"""Qwen3-TTS's depth loop: the fused q/k norm + RoPE + KV write (``qk_norm_rope_kv_write``) with its eager fallback, the
decode attention over several tokens of a row, the two-token first pass and the deferred residual adds.

The CPU tests stand a Python reference in for the Triton kernels and pin what a two-token pass means. The GPU tests hold
the kernels to the eager path at bf16 tolerance on the predictor's own geometry (16 groups, 16 q and 8 KV heads of 128,
rows 1/8/28).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import mstar.model.qwen3_tts.components.talker as talker_module
from mstar.model.components.norm import RMSNorm
from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from mstar.model.qwen3_tts.config import (
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.model.qwen3_tts.submodules import FIRST_PASS_TOKENS, TalkerSubmodule
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


def test_two_token_pass_is_two_one_token_passes(python_kernels):
    config = _config()
    torch.manual_seed(0)
    predictor = Qwen3TTSCodePredictor(config)
    _init_weights(predictor)
    batch = 3
    inputs = torch.randn(batch, FIRST_PASS_TOKENS, config.talker.hidden_size)
    rows = torch.arange(batch)

    one = _cache(config, batch)
    with torch.no_grad():
        predictor.forward_depth_unrolled(inputs[:, :1].clone(), rows * 0, rows, one, cache_pos=0)
        second = predictor.forward_depth_unrolled(inputs[:, 1:].clone(), rows * 0 + 1, rows, one, cache_pos=1)

        two = _cache(config, batch)
        out = predictor.forward_depth_unrolled(
            inputs.clone(),
            torch.arange(FIRST_PASS_TOKENS).repeat(batch),
            rows.repeat_interleave(FIRST_PASS_TOKENS),
            two,
            cache_pos=0,
        )
    assert out.shape == (batch, FIRST_PASS_TOKENS, config.code_predictor.hidden_size)
    torch.testing.assert_close(out[:, 1:], second)
    torch.testing.assert_close(two, one)


def addressing_dtypes(submodule: TalkerSubmodule) -> set[torch.dtype]:
    full = submodule._depth_addressing
    return {full.first_positions.dtype, full.positions.dtype}


@pytest.mark.parametrize("two_tokens", [True, False], ids=["two_tokens", "one_token"])
def test_depth_loop_passes(monkeypatch, two_tokens):
    config = _config()
    submodule = TalkerSubmodule(Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config)
    submodule.code_predictor.consolidate_stacked_weights()
    submodule.two_token_first_pass = two_tokens
    cp = config.talker.code_predictor
    calls = []

    def fake_pass(inputs_embeds, position_ids, batch_rows, kv_cache, cache_pos):
        calls.append((inputs_embeds.shape[:2], position_ids.tolist(), batch_rows.tolist(), cache_pos))
        return torch.zeros(*inputs_embeds.shape[:2], cp.hidden_size)

    monkeypatch.setattr(submodule.code_predictor, "forward_depth_unrolled", fake_pass)
    batch = 2
    hidden = torch.randn(batch, config.talker.hidden_size)
    layer0 = torch.tensor([1, 2])
    codes, embed_sum = submodule._depth_loop(hidden, layer0, lambda logits: logits.argmax(-1))

    if two_tokens:
        first = [((batch, 2), [0, 1, 0, 1], [0, 0, 1, 1], 0)]
    else:
        first = [((batch, 1), [group, group], [0, 1], group) for group in range(2)]
    assert calls == first + [((batch, 1), [group, group], [0, 1], group) for group in range(2, GROUPS)]
    assert addressing_dtypes(submodule) == {torch.int32}
    assert codes.shape == (batch, GROUPS)
    assert codes[:, 0].tolist() == [1, 2]
    assert embed_sum.shape == (batch, config.talker.hidden_size)


class _ScaleAttention(torch.nn.Module):
    """Stands in for the paged attention: the residual stream is what's under test."""

    def __init__(self):
        super().__init__()
        self.attend = SimpleNamespace(bind_step=lambda label: None, set_layer_idx=lambda idx: None)

    def forward(self, hidden_states):
        return 0.5 * hidden_states


def _talker_language_model(config):
    model = Qwen3TTSTalkerModel(config).model
    for layer in model.layers:
        layer.self_attn = _ScaleAttention()
        layer.mlp = torch.nn.Identity()
    return model


def test_talker_keeps_its_input_out_of_the_in_place_residual(monkeypatch):
    """``RMSNorm.forward_residual`` only writes in place on CUDA, so stand that in: a replay's input is a static buffer
    the next replay reads."""

    def in_place(self, hidden_states, residual, comm_group=None):
        residual.add_(hidden_states)
        hidden_states.copy_(self(residual))
        return hidden_states, residual

    config = _config()
    torch.manual_seed(0)
    fused = _talker_language_model(config)
    fused.fused_add_norm = True
    plain = _talker_language_model(config)
    plain.load_state_dict(fused.state_dict())
    embeds = torch.randn(3, config.talker.hidden_size)
    before = embeds.clone()
    with torch.no_grad():
        want = plain(embeds, label="main")
        monkeypatch.setattr(RMSNorm, "forward_residual", in_place)
        got = fused(embeds, label="main")
    assert torch.equal(embeds, before)
    torch.testing.assert_close(got, want)


def _attend_inputs(config, batch):
    cp = config.talker.code_predictor
    predictor = Qwen3TTSCodePredictor(config)
    _init_weights(predictor)
    attn = predictor.model.layers[0].self_attn
    q = torch.randn(batch, cp.num_attention_heads, cp.head_dim)
    k = torch.randn(batch, cp.num_key_value_heads, cp.head_dim)
    v = torch.randn(batch, cp.num_key_value_heads, cp.head_dim)
    return predictor, attn, q, k, v, _cache(config, batch)[0]


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
@pytest.mark.parametrize("tokens,cache_pos", [(2, 0), (1, 2), (1, 15)])
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


@needs_cuda
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cache_len", [1, 2, 16])
def test_one_token_attention_matches_the_reference(rows, cache_len):
    g = torch.Generator(device="cpu").manual_seed(cache_len)
    q = torch.randn(rows, 1, HQ, D, generator=g).cuda().bfloat16()
    k = torch.randn(rows, 16, HKV, D, generator=g).cuda().bfloat16()
    v = torch.randn(rows, 16, HKV, D, generator=g).cuda().bfloat16()
    got = attention.decode_attn_nhd(q, k, v, cache_len)
    _assert_bf16_close(got, _reference_attention(q, k, v, cache_len), min_equal=0.3)


@needs_cuda
@pytest.mark.parametrize("rows", ROWS)
def test_two_token_attention_matches_one_token_calls(rows):
    g = torch.Generator(device="cpu").manual_seed(2)
    q = torch.randn(rows, 2, HQ, D, generator=g).cuda().bfloat16()
    k = torch.randn(rows, 16, HKV, D, generator=g).cuda().bfloat16()
    v = torch.randn(rows, 16, HKV, D, generator=g).cuda().bfloat16()
    got = attention.decode_attn_nhd(q, k, v, 2)
    want = torch.cat([attention.decode_attn_nhd(q[:, t:t + 1], k, v, t + 1) for t in range(2)], dim=1)
    _assert_bf16_close(got, want, min_equal=0.99)


def _predictor(rows):
    config = _config(layers=5, heads=HQ, kv_heads=HKV, head_dim=D, hidden=1024, groups=16)
    torch.manual_seed(0)
    predictor = Qwen3TTSCodePredictor(config)
    _init_weights(predictor)
    predictor = predictor.cuda().bfloat16()
    return predictor, torch.zeros(5, rows, 2, 16, HKV, D, device="cuda", dtype=torch.bfloat16)


@needs_cuda
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("measured", [True, False], ids=["blackwell", "older"])
def test_depth_passes_match_the_eager_one_token_passes(rows, measured, monkeypatch):
    """All 16 positions through five real-geometry layers against the eager norm, RoPE, write and add with one token per
    pass. ``measured`` runs the fused add+norms and the two-token first pass over the kernel; without it, the kernel
    with a plain add and one token per pass."""
    predictor, cache = _predictor(rows)
    predictor.fused_add_norm = measured
    g = torch.Generator(device="cpu").manual_seed(3)
    inputs = torch.randn(rows, 16, 2048, generator=g).cuda().bfloat16()
    batch_rows = torch.arange(rows, device="cuda")

    with torch.inference_mode():
        fused = []
        if measured:
            fused.append(predictor.forward_depth_unrolled(
                inputs[:, :2].clone(), torch.arange(2, device="cuda").repeat(rows), batch_rows.repeat_interleave(2),
                cache, cache_pos=0,
            )[:, 1])
        for pos in range(2 if measured else 0, 16):
            out = predictor.forward_depth_unrolled(
                inputs[:, pos:pos + 1].clone(), batch_rows * 0 + pos, batch_rows, cache, cache_pos=pos,
            )[:, 0]
            if pos:
                fused.append(out)

        monkeypatch.setattr(talker_module, "qk_norm_rope_kv_write_applies", lambda q, layer: False)
        eager_predictor, eager_cache = _predictor(rows)
        eager = []
        for pos in range(16):
            out = eager_predictor.forward_depth_unrolled(
                inputs[:, pos:pos + 1].clone(), batch_rows * 0 + pos, batch_rows, eager_cache, cache_pos=pos,
            )[:, 0]
            if pos:
                eager.append(out)

    for got, want in zip(fused, eager, strict=True):
        # five layers of rounding, each at most a step or so
        _assert_bf16_close(got, want, min_equal=0.0, steps=8.0)
