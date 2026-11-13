"""MX rounding in torch (``mstar.utils.mx_round``): the FP8 and FP4 grids DeepSeek-V4.1 rounds to."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.utils import mx_round


class TestMxRounding:
    def test_fp4_rounds_to_nearest_with_ties_to_the_even_code(self):
        v = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.76, 5.1, 0.2, -6.0])
        assert mx_round.round_e2m1(v).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0, 0.0, -6.0]

    def test_fp4_packing_round_trips(self):
        q = mx_round.round_e2m1((torch.randn(8, 64, generator=torch.Generator().manual_seed(0)) * 2).clamp(-6, 6))
        packed = mx_round.pack_e2m1(q)
        assert packed.shape == (8, 32) and torch.equal(mx_round.unpack_e2m1(packed), q)

    def test_scales_round_up_to_a_power_of_two(self):
        v = torch.tensor([1.0, 1.0000001, 0.75, 0.5, 3.0, 2.0 ** -20])
        assert mx_round.pow2_ceil(v).tolist() == [1.0, 2.0, 1.0, 0.5, 4.0, 2.0 ** -20]

    @pytest.mark.parametrize("round_", [
        lambda x: mx_round.round_fp8(x, 32),
        lambda x: mx_round.round_fp4(x, 32),
        lambda x: mx_round.round_fp4(x, 16, e4m3_scale=True),
    ])
    def test_rounded_values_are_exact_in_bf16_and_rounding_them_again_changes_nothing(self, round_):
        x = torch.randn(16, 64, generator=torch.Generator().manual_seed(1)) * 3
        once = round_(x)
        assert torch.equal(once.to(torch.bfloat16).float(), once)
        assert torch.equal(round_(once), once)

    def test_fp8_rounding_stays_within_half_a_step_of_the_block_scale(self):
        x = torch.randn(4, 64, generator=torch.Generator().manual_seed(2))
        q, scale = mx_round.quant_fp8(x, 32)
        dense = q.float().unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)
        # E4M3 keeps 4 significant bits: a relative error of at most 1/16, plus the subnormals' step
        assert ((dense.flatten(-2) - x).abs() <= x.abs() / 16 + scale.repeat_interleave(32, -1) * 2.0 ** -10).all()
