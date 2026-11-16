"""Published E2M1/E8M0 expert weight layout and FP8 conversion guard."""

import pytest
import torch

from mstar.utils.deepseek_v41_expert_weights import (
    convert_expert_fp4_to_fp8,
    dequantize_expert_fp4,
)
from mstar.utils.fp8_block import dequantize_fp8_block_weight


def _scales(exponents: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.tensor(2.0), exponents.float()).to(torch.float8_e8m0fnu)


def test_dequantize_low_nibble_first_and_per_row_per_32_scales():
    packed = torch.full((32, 32), -15, dtype=torch.int8)  # 0xF1: +0.5, -6
    exponents = torch.zeros((32, 2), dtype=torch.int64)
    exponents[0, 1] = 2
    exponents[1, 0] = -1
    got = dequantize_expert_fp4(packed, _scales(exponents), dtype=torch.float32)
    assert got.shape == (32, 64)
    assert got[0, :2].tolist() == [0.5, -6.0]
    assert got[0, 32:34].tolist() == [2.0, -24.0]
    assert got[1, :2].tolist() == [0.25, -3.0]


@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (64, 96), (96, 128)])
def test_fp8_blocks_reconstruct_fp4_exactly(shape):
    generator = torch.Generator().manual_seed(17)
    rows, cols = shape
    packed = torch.randint(-128, 128, (rows, cols // 2), generator=generator, dtype=torch.int8)
    exponents = torch.randint(-3, 4, (rows, cols // 32), generator=generator)
    scales = _scales(exponents)
    weight, block_scales = convert_expert_fp4_to_fp8(packed, scales)
    assert weight.dtype == torch.float8_e4m3fn
    assert block_scales.dtype == torch.float32
    assert block_scales.shape == (rows // 32, cols // 32)
    reconstructed = dequantize_fp8_block_weight(
        weight, block_scales, block_size=(32, 32), out_dtype=torch.float32,
    )
    assert torch.equal(reconstructed, dequantize_expert_fp4(packed, scales, dtype=torch.float32))


def test_fp8_refuses_row_scale_range_that_loses_nonzero_weights():
    packed = torch.full((32, 16), 0x11, dtype=torch.int8)
    exponents = torch.full((32, 1), -20, dtype=torch.int64)
    exponents[0] = 10
    scales = _scales(exponents)
    expected = dequantize_expert_fp4(packed, scales, dtype=torch.float32)
    assert expected[1, 0] > 0
    with pytest.raises(ValueError, match="not exactly representable"):
        convert_expert_fp4_to_fp8(packed, scales)


@pytest.mark.parametrize("packed,scales", [
    (torch.zeros((0, 16), dtype=torch.int8), _scales(torch.zeros((0, 1)))),
    (torch.zeros((31, 16), dtype=torch.int8), _scales(torch.zeros((31, 1)))),
    (torch.zeros((32, 15), dtype=torch.int8), _scales(torch.zeros((32, 1)))),
    (torch.zeros((32, 16), dtype=torch.uint8), _scales(torch.zeros((32, 1)))),
    (torch.zeros((32, 16), dtype=torch.int8), torch.ones((32, 1))),
])
def test_fp8_rejects_invalid_expert_layouts(packed, scales):
    with pytest.raises(ValueError):
        convert_expert_fp4_to_fp8(packed, scales)
