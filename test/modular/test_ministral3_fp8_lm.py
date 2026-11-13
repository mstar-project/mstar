"""Tiny FP8 decoder wiring against a Transformers architecture oracle.

This is synthetic CPU validation, not pretrained or GPU kernel parity. The
reference uses independent scalar/static FP8 arithmetic because Transformers'
finegrained-fp8 v4 static kernel does not support this checkpoint's scalar scales.
Its rotary arithmetic is explicitly aligned to FP32 tables with one BF16 output
rounding, and RMSNorm to the native portable kernel's rounding order. Native
attention/cache and device matmul are CPU test doubles; production
loading, FP8 quantization, projection wiring and decoder layers run unchanged.
"""

import pytest
import torch
from test_ministral3_fp8 import QUANTIZATION, TINY, build, checkpoint, oracle_linear
from test_ministral3_lm import DenseCacheAttention, Harness, cpu_table_rotation, reference
from torch import nn

import mstar.model.components.scalar_fp8_linear as fp8_component
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.ministral3.fp8 import load_fp8_text_weights


class ReferenceLinear(nn.Module):
    """Test-only explicit quantization and FP64 dot, rounded to BF16."""

    def __init__(self, weights, stem):
        super().__init__()
        self.weights = weights
        self.stem = stem

    def forward(self, x):
        return oracle_linear(x, self.weights, self.stem)


def synthetic_checkpoint():
    weights = checkpoint()
    generator = torch.Generator().manual_seed(83)
    # Non-unit norm weights exercise both normalization and learned scaling.
    for name, value in weights.items():
        if name.endswith(("layernorm.weight", ".norm.weight")):
            weights[name] = (0.5 + torch.rand(value.shape, generator=generator)).bfloat16()
    return weights


@torch.no_grad()
def fp8_reference(weights):
    hf = reference()
    # Convert parameters only. A whole-module .bfloat16() would also round the
    # nonpersistent rotary frequency buffers before their FP32 sinusoid.
    consumed = set()
    for name, module in list(hf.named_modules()):
        if isinstance(module, nn.Linear) and name != "lm_head":
            stem = "language_model." + name
            hf.set_submodule(name, ReferenceLinear(weights, stem))
            consumed.update(stem + suffix for suffix in (".weight", ".weight_scale_inv", ".activation_scale"))
    for name, param in hf.named_parameters():
        key = "language_model." + name
        param.data = weights[key].clone()
        consumed.add(key)
    assert consumed == set(weights), "the independent reference must consume the entire synthetic checkpoint"
    assert hf.lm_head.weight is hf.model.embed_tokens.weight
    assert hf.model.rotary_emb.inv_freq.dtype == torch.float32
    return hf


class FP8Harness(Harness):
    """Reuse packed execution, substituting the strictly loaded native FP8 stack."""

    def __init__(self, weights, *, starts=None):
        self.model = build()
        load_fp8_text_weights(self.model, weights.items(), QUANTIZATION)
        self.cache = DenseCacheAttention()
        self.positions = dict(starts or {})
        self.pos = RopeManager(PositionConfig(
            kv_cache="kv", rotary_dim=TINY.head_dim, rope_theta=TINY.rope_theta, yarn=TINY.yarn,
        ), torch.device("cpu"), max_seq_len=33000)
        for layer in self.model.layers:
            layer.self_attn.bind_resources(dict(attn=self.cache, kv=self.cache, rope=self.pos))


@pytest.fixture
def cpu_fp8(monkeypatch):
    """Keep production operand preparation, replace the unsupported device dot."""
    calls = []

    def cpu_linear(x, weight, weight_scale, activation_scale):
        fp8_component._check_inputs(x, weight, weight_scale, activation_scale)
        return fp8_component._scaled_fp8_mm(x, weight, weight_scale, activation_scale)

    def cpu_mm(a, b, *, scale_a, scale_b, out_dtype, use_fast_accum):
        assert a.dtype == b.dtype == fp8_component.FP8_DTYPE
        assert out_dtype == torch.bfloat16 and not use_fast_accum
        calls.append(a.shape[0])
        return ((a.double() * scale_a.double()) @ (b.double() * scale_b.double())).to(out_dtype)

    monkeypatch.setattr(fp8_component, "scalar_fp8_linear", cpu_linear)
    monkeypatch.setattr(torch, "_scaled_mm", cpu_mm)
    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", cpu_table_rotation)
    return calls


def align_reference_arithmetic(monkeypatch, *, rotation=True, normalization=True):
    """Align rounding explicitly; do not disguise this as untouched HF parity."""
    from transformers.models.ministral3 import modeling_ministral3 as modeling

    original_forward = modeling.Ministral3RotaryEmbedding.forward
    original_apply = modeling.apply_rotary_pos_emb

    def forward(self, x, position_ids):
        return original_forward(self, x.float(), position_ids)

    def apply(q, k, cos, sin, unsqueeze_dim=1):
        qr, kr = original_apply(q.float(), k.float(), cos, sin, unsqueeze_dim)
        return qr.to(q.dtype), kr.to(k.dtype)

    def norm(self, x):
        variance = x.float().square().mean(-1, keepdim=True)
        # The portable native kernel rounds the reciprocal RMS before the
        # multiply. HF eager instead rounds the normalized hidden state.
        reciprocal_rms = torch.rsqrt(variance + self.variance_epsilon).to(x.dtype)
        return (x * reciprocal_rms) * self.weight

    if rotation:
        monkeypatch.setattr(modeling.Ministral3RotaryEmbedding, "forward", forward)
        monkeypatch.setattr(modeling, "apply_rotary_pos_emb", apply)
    if normalization:
        monkeypatch.setattr(modeling.Ministral3RMSNorm, "forward", norm)


@pytest.fixture
def aligned_reference_arithmetic(monkeypatch):
    align_reference_arithmetic(monkeypatch)


def assert_logits(actual, expected):
    # No tolerance masks wiring mistakes when arithmetic and rounding align.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_dtype=False)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize("start", [0, 16382, 32766])
@torch.no_grad()
def test_fp8_prefill_and_cached_decode(cpu_fp8, aligned_reference_arithmetic, start):
    weights = synthetic_checkpoint()
    hf, native = fp8_reference(weights), FP8Harness(weights, starts={1: start})
    sequence = torch.tensor([3, 8, 12, 4])
    oracle = hf(sequence[None], position_ids=torch.arange(start, start + 4)[None], use_cache=True)
    actual = native.step([(1, sequence)])[0]
    assert_logits(actual, oracle.logits[0])
    for _ in range(3):
        token = actual[-1].argmax().reshape(1)
        sequence = torch.cat((sequence, token))
        oracle = hf(token[None], position_ids=torch.tensor([[start + len(sequence) - 1]]),
                    past_key_values=oracle.past_key_values, use_cache=True)
        actual = native.step([(1, token)])[0]
        assert_logits(actual, oracle.logits[0])
        full = hf(sequence[None], position_ids=torch.arange(start, start + len(sequence))[None], use_cache=False)
        assert_logits(actual, full.logits[0, -1:])
    assert cpu_fp8 == [4] * 14 + [1] * 42  # Seven projections per layer, two layers.
    assert all(k.dtype == v.dtype == torch.bfloat16 and len(k) == len(sequence)
               for k, v in native.cache.cache.values())


@torch.no_grad()
def test_fp8_chunked_prefill_and_reordered_requests(cpu_fp8, aligned_reference_arithmetic):
    weights = synthetic_checkpoint()
    hf = fp8_reference(weights)
    starts = {1: 0, 2: 16382}
    native = FP8Harness(weights, starts=starts)
    sequences = {1: torch.tensor([], dtype=torch.long), 2: torch.tensor([], dtype=torch.long)}
    steps = [
        [(1, torch.tensor([3, 8])), (2, torch.tensor([4, 19, 27]))],
        [(2, torch.tensor([5, 22])), (1, torch.tensor([12]))],
        [(1, torch.tensor([6])), (2, torch.tensor([7]))],
    ]
    for requests in steps:
        for (rid, ids), actual in zip(requests, native.step(requests), strict=True):
            sequences[rid] = torch.cat((sequences[rid], ids))
            positions = torch.arange(starts[rid], starts[rid] + len(sequences[rid]))[None]
            expected = hf(sequences[rid][None], position_ids=positions, use_cache=False).logits[0, -len(ids):]
            assert_logits(actual, expected)


@pytest.mark.parametrize("broken", ["shared_scale", "no_activation_quantization"])
@torch.no_grad()
def test_full_stack_parity_detects_quantization_errors(cpu_fp8, aligned_reference_arithmetic, monkeypatch, broken):
    weights = synthetic_checkpoint()
    hf = fp8_reference(weights)
    ids = torch.tensor([3, 8, 12, 4])
    expected = hf(ids[None], use_cache=False).logits[0]
    assert_logits(FP8Harness(weights).step([(1, ids)])[0], expected)
    native = FP8Harness(weights)
    if broken == "shared_scale":
        for layer in native.model.layers:
            projections = layer.self_attn.qkv_proj.projections
            projections.k.activation_scale.copy_(projections.q.activation_scale)
    else:
        def unquantized(x, weight, weight_scale, activation_scale):
            return (x.double() @ (weight.view(fp8_component.FP8_DTYPE).double() * weight_scale.double()).T).bfloat16()

        monkeypatch.setattr(fp8_component, "scalar_fp8_linear", unquantized)
    actual = native.step([(1, ids)])[0]
    with pytest.raises(AssertionError):
        assert_logits(actual, expected)


@pytest.mark.parametrize("unaligned", ["rotation", "normalization"])
@torch.no_grad()
def test_rounding_differences_are_observable(cpu_fp8, monkeypatch, unaligned):
    """Each alignment is necessary for exactness, rather than just assumed so."""
    weights = synthetic_checkpoint()
    ids = torch.tensor([3, 8, 12, 4])
    start = 16382
    actual = FP8Harness(weights, starts={1: start}).step([(1, ids)])[0]
    with monkeypatch.context() as partial:
        align_reference_arithmetic(partial, **{unaligned: False})
        expected = fp8_reference(weights)(
            ids[None], position_ids=torch.arange(start, start + len(ids))[None], use_cache=False,
        ).logits[0]
    print(f"unaligned {unaligned}: max_abs_logit_error={(actual.float() - expected.float()).abs().max().item()}")
    with pytest.raises(AssertionError):
        assert_logits(actual, expected)
    # Restoring only the differing arithmetic boundary removes the discrepancy.
    align_reference_arithmetic(monkeypatch)
    expected = fp8_reference(weights)(
        ids[None], position_ids=torch.arange(start, start + len(ids))[None], use_cache=False,
    ).logits[0]
    assert_logits(actual, expected)
