"""``linear_dtype: fp8``: the platform's e4m3 format, per-row quantisation, the GEMM op's
reference, and the format on main's ``QuantLinear``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar import platforms
from mstar.kernels.fp8_gemm import FP8_ROWWISE_MM, quantize_rows
from mstar.model.components.fp8_rowwise import Fp8RowwiseFormat
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.components.quant_linear import LINEAR_DTYPES, QuantLinear, check_linear_dtype, linear_format
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform

FP8_DTYPES = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


class _Rocm(RocmPlatform):
    def __init__(self, arch: str):
        self._arch = arch

    def arch(self, device=None) -> str:
        return self._arch


class _Cuda(CudaPlatform):
    def __init__(self, capability: tuple[int, int]):
        self._capability = capability

    def cuda_capability(self, device=None) -> tuple[int, int]:
        return self._capability


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm())


@pytest.fixture
def on_gfx942(monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Rocm("gfx942"))


def test_the_fp8_format_comes_from_the_architecture():
    assert _Rocm("gfx942").fp8_dtype() == torch.float8_e4m3fnuz
    assert _Rocm("gfx950").fp8_dtype() == torch.float8_e4m3fn
    assert _Rocm("gfx90a").fp8_dtype() is None
    assert _Cuda((9, 0)).fp8_dtype() == torch.float8_e4m3fn
    assert _Cuda((8, 9)).fp8_dtype() == torch.float8_e4m3fn
    assert _Cuda((8, 0)).fp8_dtype() is None
    assert CpuPlatform().fp8_dtype() is None


@pytest.mark.parametrize("fp8", FP8_DTYPES, ids=str)
def test_quantize_rows_maps_each_rows_largest_value_to_the_format_max(fp8):
    torch.manual_seed(0)
    t = torch.randn(64, 256) * torch.logspace(-3, 3, 64)[:, None]
    t[5] = 0
    q, scale = quantize_rows(t, fp8)
    assert q.dtype == fp8 and scale.dtype == torch.float32 and scale.shape == (64, 1)
    peak = q.float().abs().amax(dim=-1)
    assert torch.all(peak[torch.arange(64) != 5] == torch.finfo(fp8).max)
    assert torch.all(q[5].float() == 0) and torch.isfinite(scale).all()
    # e4m3 keeps 3 mantissa bits: rounding moves a normal value by at most 2^-4 of itself
    back = q.float() * scale
    normal = t.abs() >= scale * torch.finfo(fp8).smallest_normal
    assert torch.all(((back - t).abs() <= t.abs() * 2 ** -4)[normal])


@pytest.mark.parametrize("bias", [False, True])
def test_the_ops_reference_is_the_dequantised_product(bias):
    torch.manual_seed(0)
    xq, x_scale = quantize_rows(torch.randn(48, 64), torch.float8_e4m3fnuz)
    wq, w_scale = quantize_rows(torch.randn(32, 64), torch.float8_e4m3fnuz)
    b = torch.randn(32).to(torch.bfloat16) if bias else None
    out = FP8_ROWWISE_MM(xq, x_scale, wq, w_scale.T.contiguous(), b, torch.bfloat16)
    expected = (xq.float() * x_scale) @ (wq.float() * w_scale).T + (b.float() if bias else 0)
    assert out.dtype == torch.bfloat16 and out.shape == (48, 32)
    torch.testing.assert_close(out, expected.to(torch.bfloat16), rtol=0, atol=0)


def test_fp8_is_a_registered_linear_format():
    assert "fp8" in LINEAR_DTYPES and check_linear_dtype("fp8") == "fp8"
    fmt = linear_format("fp8")
    assert isinstance(fmt, Fp8RowwiseFormat) and fmt.buffers == ("weight", "weight_scale")
    with pytest.raises(ValueError, match="linear_dtype must be one of"):
        check_linear_dtype("fp16")


def test_without_an_fp8_gemm_the_format_says_why(monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Rocm("gfx90a"))
    assert "rocm gfx90a has no fp8 GEMM" in linear_format("fp8").unavailable_reason(torch.device("cpu"))


@pytest.mark.parametrize("bias", [False, True])
def test_an_fp8_quant_linear_tracks_the_bf16_linear(on_gfx942, bias):
    torch.manual_seed(0)
    linear = nn.Linear(128, 96, bias=bias).to(torch.bfloat16)
    x = torch.randn(2, 7, 128).to(torch.bfloat16)
    quantized = QuantLinear.from_linear(linear, linear_format("fp8"))
    out = quantized(x)
    assert out.shape == (2, 7, 96) and out.dtype == torch.bfloat16
    assert quantized.weight.dtype == torch.float8_e4m3fnuz and quantized.weight_scale.shape == (1, 96)
    assert _rel(out, F.linear(x, linear.weight, linear.bias)) < 0.05


def test_a_fused_column_linear_quantises_per_output_row(on_gfx942):
    torch.manual_seed(0)
    fused = FusedColumnLinear(64, {"q": 32, "k": 32, "v": 48}).to(torch.bfloat16)
    nn.init.normal_(fused.weight)
    with torch.no_grad():
        fused.weight[32:64] *= 100  # one shard far larger than the others
    quantized = QuantLinear.from_linear(fused, linear_format("fp8"))
    x = torch.randn(5, 64).to(torch.bfloat16)
    reference = fused(x)
    out = quantized(x)
    for rows in (slice(0, 32), slice(32, 64), slice(64, 112)):
        assert _rel(out[:, rows], reference[:, rows]) < 0.05


def test_a_linear_the_gemm_cannot_take_is_refused(on_gfx942):
    with pytest.raises(ValueError, match="multiples of 16"):
        QuantLinear.from_linear(nn.Linear(32, 40).to(torch.bfloat16), linear_format("fp8"))


def test_an_fp8_quant_linear_is_one_op_under_dynamo(on_gfx942):
    torch.manual_seed(0)
    quantized = QuantLinear.from_linear(nn.Linear(64, 32).to(torch.bfloat16), linear_format("fp8"))
    x = torch.randn(3, 64).to(torch.bfloat16)
    compiled = torch.compile(quantized, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), quantized(x), rtol=0, atol=0)


def test_the_models_take_linear_dtype_fp8():
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
    from mstar.model.z_image.z_image_model import ZImageModel

    assert Flux2KleinModel(model_path_hf="x").linear_dtype == "bf16"
    assert ZImageModel(model_path_hf="x", linear_dtype="fp8").linear_dtype == "fp8"
    assert Flux2KleinModel(model_path_hf="x", linear_dtype="fp8", linear_dtype_steps=2).step_dtypes.steps == (
        "bf16", "bf16", "fp8",
    )
