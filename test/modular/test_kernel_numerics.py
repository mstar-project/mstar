"""The parity bar of ``mstar.kernels.numerics``: one round-to-nearest-even of
an fp32 result passes ``rounded_once``, a truncating store does not though
every element is within an ulp, and ``accumulated`` and ``exact`` mean what
they say."""

import pytest
import torch

from mstar.kernels.numerics import Tolerance, compare, reference_outputs, ulp


def _ref(n=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dtype=torch.float64, generator=g) * 3


def _truncate_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp32 to bf16 by dropping the low mantissa bits, as a truncating store does."""
    bits = x.float().view(torch.int32) & -65536
    return bits.view(torch.float32).to(torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_math_rounded_once_passes(dtype):
    ref = _ref()
    result = compare(ref.float().to(dtype), ref, Tolerance.rounded_once())
    assert result.ok, result.summary()
    assert result.max_ulps <= 1.0 and result.mismatched == 0


def test_a_truncating_store_fails_on_the_norm_alone():
    ref = _ref()
    result = compare(_truncate_to_bf16(ref), ref, Tolerance.rounded_once())
    assert not result.ok
    assert result.mismatched == 0, "every element is within one ulp"
    assert result.rel_norm > 1e-3


def test_an_element_two_ulps_off_fails():
    ref = _ref()
    out = ref.to(torch.bfloat16)
    out[7] = (ref[7] + 2 * ulp(ref[7:8], torch.bfloat16)[0]).to(torch.bfloat16)
    result = compare(out, ref, Tolerance.rounded_once(rel_norm=1.0))
    assert not result.ok and result.mismatched == 1 and result.max_ulps >= 1.5


def test_accumulated_allows_summation_order_but_not_a_wrong_result():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 2048, dtype=torch.float64, generator=g)
    w = torch.randn(512, 2048, dtype=torch.float64, generator=g) / 2048 ** 0.5
    ref = x @ w.T
    gemm = (x.bfloat16().float() @ w.bfloat16().float().T).bfloat16()
    assert compare(gemm, ref, Tolerance.accumulated()).ok
    assert not compare(gemm * 1.05, ref, Tolerance.accumulated()).ok


def test_exact_is_bit_for_bit():
    ref = _ref()
    out = ref.to(torch.bfloat16)
    assert compare(out, ref, Tolerance.exact()).ok
    out[0] = (ref[0] + ulp(ref[:1], torch.bfloat16)[0]).to(torch.bfloat16)
    assert not compare(out, ref, Tolerance.exact()).ok


def test_a_nan_where_the_reference_is_finite_fails():
    ref = _ref(16)
    out = ref.to(torch.bfloat16)
    out[3] = float("nan")
    result = compare(out, ref, Tolerance.rounded_once(rel_norm=1.0))
    assert not result.ok and result.mismatched == 1


def test_integer_outputs_compare_exactly():
    ref = torch.arange(10)
    assert compare(ref.int(), ref, Tolerance.rounded_once()).ok
    assert not compare(ref.int() + 1, ref, Tolerance.rounded_once()).ok


def test_shapes_must_match():
    with pytest.raises(ValueError, match="shape"):
        compare(torch.zeros(3), torch.zeros(4, dtype=torch.float64), Tolerance.exact())


def test_ulp_of_bf16():
    values = torch.tensor([1.0, 1.5, 0.75, -2.0, 0.0], dtype=torch.float64)
    spacing = ulp(values, torch.bfloat16)
    assert spacing[:4].tolist() == [2 ** -7, 2 ** -7, 2 ** -8, 2 ** -6]
    assert 0 < spacing[4] < 1e-38


def test_reference_outputs_run_in_fp64_and_return_mutated_args():
    def add_into(out, x):
        out.add_(x)
        return x * 2

    out, x = torch.ones(3, dtype=torch.bfloat16), torch.full((3,), 0.5, dtype=torch.bfloat16)
    doubled, mutated = reference_outputs(add_into, [out, x], mutated=(0,))
    assert doubled.dtype == mutated.dtype == torch.float64
    assert mutated.tolist() == [1.5] * 3 and doubled.tolist() == [1.0] * 3
    assert out.tolist() == [1.0] * 3, "the caller's tensors are left alone"
