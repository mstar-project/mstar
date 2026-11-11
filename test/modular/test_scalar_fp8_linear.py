"""Scalar/static FP8 contracts; CPU arithmetic is a test oracle, not a fallback.

Reference: kernels-community/finegrained-fp8 revision
3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8/build/torch-cuda/matmul.py,
w8a8_block_static_fp8_matmul_kernel (lines 243-352: FP32 division then
saturating E4M3 cast). The static wrapper at line 477 rejects tensor mode.
Transformers 5.22.2 integrations/hub_kernels.py pins v4. Its matmul_2d
routes static activation_scale into a block-only implementation, so these tests
do not claim scalar/static Transformers GPU parity. The CUDA tests exercise the
native component against explicitly quantized arithmetic when hardware exists.
An offline Triton 3.9 compilation of x.to(tl.float8e4nv) for sm90 emitted
cvt.rn.satfinite.e4m3x2.f32, establishing saturation before the torch FP8 cast.
"""
from __future__ import annotations

import pytest
import torch

from mstar.model.components.scalar_fp8_linear import (
    FP8_DTYPE,
    ScalarFp8Linear,
    _scaled_fp8_mm,
    quantize_static_fp8,
)
from mstar.model.loader.base import load_weights_into
from mstar.platforms import current_platform


def _loaded_linear(in_features=32, out_features=16, *, activation_scale=0.125):
    layer = ScalarFp8Linear(in_features, out_features)
    generator = torch.Generator().manual_seed(321)
    weight = torch.randn(out_features, in_features, generator=generator).to(FP8_DTYPE)
    values = {
        "weight": weight,
        "weight_scale_inv": torch.tensor(0.03125, dtype=torch.bfloat16),
        "activation_scale": torch.tensor(activation_scale, dtype=torch.bfloat16),
    }
    for name, value in values.items():
        param = getattr(layer, name)
        param.weight_loader(param, value)
    return layer, values


def _arithmetic_oracle(x, weight, weight_scale, activation_scale):
    # Explicitly reconstruct quantized values in FP64 for a high precision dot.
    # This deliberately exists only in tests, never in the served component.
    a = (x.double() / activation_scale.double()).clamp(-448, 448).to(FP8_DTYPE)
    return ((a.double() * activation_scale.double()) @
            (weight.double() * weight_scale.double()).t()).to(torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_dtype_moves_preserve_weight_bytes_and_scale_values(dtype):
    layer, values = _loaded_linear()
    # These FP32 values would change or overflow if _apply cast and then restored.
    layer.weight_scale_inv.weight_loader(
        layer.weight_scale_inv, torch.tensor(1.000123, dtype=torch.float32))
    layer.activation_scale.weight_loader(
        layer.activation_scale, torch.tensor(100000.0, dtype=torch.float32))
    expected_scales = [p.detach().clone() for p in
                       (layer.weight_scale_inv, layer.activation_scale)]
    layer.to(dtype=dtype)
    torch.testing.assert_close(layer.weight, values["weight"].view(torch.uint8), rtol=0, atol=0)
    for name, expected in zip(("weight_scale_inv", "activation_scale"), expected_scales, strict=True):
        param = getattr(layer, name)
        assert param.dtype == torch.float32
        assert callable(param.weight_loader)
        torch.testing.assert_close(param, expected, rtol=0, atol=0)
    assert callable(layer.weight.weight_loader)


def test_all_fp8_byte_patterns_survive_loading_and_dtype_moves():
    layer = ScalarFp8Linear(16, 16)
    bits = torch.arange(256).to(torch.uint8).reshape(16, 16)
    layer.weight.weight_loader(layer.weight, bits.view(FP8_DTYPE))
    layer.to(dtype=torch.bfloat16)
    torch.testing.assert_close(layer.weight, bits, rtol=0, atol=0)


def test_meta_materialization_and_state_dict_round_trip():
    original, _ = _loaded_linear()
    layer = ScalarFp8Linear(32, 16, device="meta").to(dtype=torch.bfloat16)
    layer.to_empty(device="cpu")
    layer.load_state_dict(original.state_dict(), strict=True)
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(layer.state_dict()[name], expected, rtol=0, atol=0)
    incomplete = dict(original.state_dict())
    del incomplete["activation_scale"]
    with pytest.raises(RuntimeError, match="activation_scale"):
        layer.load_state_dict(incomplete, strict=True)


@pytest.mark.parametrize("missing", [None, "weight_scale_inv", "activation_scale"])
def test_shared_strict_loader_dispatches_all_checkpoint_parameters(missing):
    _, values = _loaded_linear(activation_scale=0.0137)
    layer = ScalarFp8Linear(32, 16).to(dtype=torch.bfloat16)
    container = torch.nn.ModuleDict({"q_proj": layer})
    weights = [(f"q_proj.{name}", value) for name, value in values.items()
               if name != missing]
    if missing is not None:
        with pytest.raises(ValueError, match=f"q_proj.{missing}"):
            load_weights_into(container, weights, strict=True)
        return
    loaded = load_weights_into(container, weights, strict=True)
    assert loaded == {f"q_proj.{name}" for name in values}
    torch.testing.assert_close(layer.weight, values["weight"].view(torch.uint8), rtol=0, atol=0)
    for name in ("weight_scale_inv", "activation_scale"):
        torch.testing.assert_close(getattr(layer, name), values[name].float(), rtol=0, atol=0)


def test_checkpoint_bf16_scales_are_lossless():
    layer, values = _loaded_linear(activation_scale=0.0137)
    for name in ("weight_scale_inv", "activation_scale"):
        assert getattr(layer, name).item() == values[name].float().item()


@pytest.mark.parametrize("name", ["weight_scale_inv", "activation_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scales_rejected_at_loading(name, value):
    layer = ScalarFp8Linear(32, 16)
    param = getattr(layer, name)
    with pytest.raises(ValueError, match="finite and positive"):
        param.weight_loader(param, torch.tensor(value))


def test_rejects_wrong_checkpoint_shapes_and_dtypes():
    layer = ScalarFp8Linear(32, 16)
    with pytest.raises(TypeError, match="original E4M3"):
        layer.weight.weight_loader(layer.weight, torch.zeros(16, 32, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="checkpoint weight"):
        layer.weight.weight_loader(layer.weight, torch.zeros(32, 32).to(FP8_DTYPE))
    for bad_scale in (torch.ones(1), torch.tensor(1.0, dtype=torch.float64)):
        with pytest.raises(TypeError, match="checkpoint scalar"):
            layer.activation_scale.weight_loader(layer.activation_scale, bad_scale)
    with pytest.raises(ValueError, match="multiples of 16"):
        ScalarFp8Linear(31, 16)


def test_static_quantization_rounding_and_overflow():
    # E4M3 ties choose an even mantissa: 1.0625 -> 1, 1.1875 -> 1.25.
    normalized = torch.tensor([0., 1., 1.0625, 1.1875, -1.1875, 448., 480., -1000.])
    scale = torch.tensor(0.125)
    actual = quantize_static_fp8(normalized * scale, scale).float()
    expected = torch.tensor([0., 1., 1., 1.25, -1.25, 448., 448., -448.])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Negative control: torch's unsaturated cast produces NaNs outside the range.
    assert torch.isnan(normalized.to(FP8_DTYPE).float()[-2:]).all()


def test_fp8_matmul_operands_and_supplied_scales(monkeypatch):
    layer, values = _loaded_linear()
    x = torch.linspace(-3, 3, 64).reshape(2, 32).to(torch.bfloat16)
    calls = []

    def check_mm(a, b, *, scale_a, scale_b, out_dtype, use_fast_accum):
        assert a.dtype == b.dtype == FP8_DTYPE
        assert a.is_contiguous() and b.t().is_contiguous()
        assert out_dtype == torch.bfloat16 and not use_fast_accum
        assert scale_a is layer.activation_scale
        assert scale_b is layer.weight_scale_inv
        torch.testing.assert_close(b.t().view(torch.uint8), values["weight"].view(torch.uint8))
        calls.append(True)
        return ((a.double() * scale_a.double()) @
                (b.double() * scale_b.double())).to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", check_mm)
    actual = _scaled_fp8_mm(x, layer.weight, layer.weight_scale_inv, layer.activation_scale)
    expected = _arithmetic_oracle(x, values["weight"], values["weight_scale_inv"],
                                  values["activation_scale"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [True]
    unquantized = (x.float() @ (values["weight"].float() *
                              values["weight_scale_inv"].float()).t()).to(torch.bfloat16)
    assert not torch.equal(actual, unquantized)


def test_separate_projections_keep_distinct_activation_scales():
    x = torch.linspace(-1, 1, 32).to(torch.bfloat16)
    a, _ = _loaded_linear(activation_scale=0.125)
    b, _ = _loaded_linear(activation_scale=0.019)
    qa = quantize_static_fp8(x, a.activation_scale).float() * a.activation_scale
    qb = quantize_static_fp8(x, b.activation_scale).float() * b.activation_scale
    assert not torch.equal(qa, qb)
    assert a.activation_scale.data_ptr() != b.activation_scale.data_ptr()


def test_cpu_forward_has_no_dequantization_fallback():
    layer, _ = _loaded_linear()
    with pytest.raises(RuntimeError, match="requires NVIDIA"):
        layer(torch.ones(2, 32, dtype=torch.bfloat16))


def test_export_uses_fake_contract_without_executing_cpu_fallback():
    layer, _ = _loaded_linear()
    exported = torch.export.export(layer, (torch.empty(2, 3, 32, dtype=torch.bfloat16),))
    op_nodes = [n for n in exported.graph.nodes
                if n.target == torch.ops.mstar.scalar_fp8_linear.default]
    assert len(op_nodes) == 1
    output = op_nodes[0].meta["val"]
    assert output.shape == (2, 3, 16) and output.dtype == torch.bfloat16


@pytest.mark.skipif(not current_platform.has_cuda_capability(8, 9),
                    reason="requires NVIDIA SM89+ FP8 hardware")
@pytest.mark.parametrize("tokens", [1, 2, 17])
def test_gpu_fp8_mm_matches_quantized_arithmetic(tokens):
    layer, values = _loaded_linear(64, 32)
    x = torch.randn(tokens, 64, generator=torch.Generator().manual_seed(42)).to(torch.bfloat16)
    expected = _arithmetic_oracle(x, values["weight"], values["weight_scale_inv"],
                                  values["activation_scale"])
    layer.to(device="cuda", dtype=torch.bfloat16)
    actual = layer(x.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
