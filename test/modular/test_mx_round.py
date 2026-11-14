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

    def test_an_fp8_scale_is_the_block_maximum_times_the_fp32_reciprocal_of_448_rounded_up(self):
        """The product is taken in fp32, as the checkpoint's reference takes it: maxima on, just
        under and just over those whose product is a power of two, over 40 binades, and around
        the 1e-4 floor."""
        edge = 448.0 * torch.exp2(torch.arange(-20.0, 20.0))
        amax = torch.cat([edge, torch.nextafter(edge, edge * 0), torch.nextafter(edge, edge * 2),
                          torch.tensor([0.0, 9e-5, 1e-4, 1.1e-4])])
        x = torch.zeros(amax.numel(), 32)
        x[:, 5] = -amax
        reciprocal = torch.tensor(1 / 448.0, dtype=torch.float32)
        for dtype in (torch.float32, torch.bfloat16):
            held = x.to(dtype).float().abs().amax(-1)  # what the dtype holds of each maximum
            _, scale = mx_round.quant_fp8(x.to(dtype), 32)
            assert torch.equal(scale[:, 0], mx_round.pow2_ceil(held.clamp_min(1e-4) * reciprocal))
    def test_a_first_call_in_inference_mode_leaves_later_calls_under_autograd_working(self):
        """The helpers keep their constants a device. One made under inference mode would be an
        inference tensor, which autograd refuses to save."""
        for cached in (mx_round._scalar, mx_round._e2m1_grid, mx_round._e2m1_bytes):
            cached.cache_clear()
        x = torch.randn(4, 64, generator=torch.Generator().manual_seed(4))
        with torch.inference_mode():
            first = mx_round.round_fp8(x, 32), mx_round.round_fp4(x, 32)
            rows = mx_round.pack_fp4_rows(x, 32)
        tracked = x.clone().requires_grad_()
        assert torch.equal(mx_round.round_fp8(tracked, 32), first[0])
        assert torch.equal(mx_round.round_fp4(tracked, 32), first[1])
        assert torch.equal(mx_round.unpack_fp4_rows(rows.clone(), 64, 32), first[1])

    def test_fp8_rounding_stays_within_half_a_step_of_the_block_scale(self):
        x = torch.randn(4, 64, generator=torch.Generator().manual_seed(2))
        q, scale = mx_round.quant_fp8(x, 32)
        dense = q.float().unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)
        # E4M3 keeps 4 significant bits: a relative error of at most 1/16, plus the subnormals' step
        assert ((dense.flatten(-2) - x).abs() <= x.abs() / 16 + scale.repeat_interleave(32, -1) * 2.0 ** -10).all()


class TestFp4RowStore:
    """A row's FP4 bytes read back as the row rounded, bit for bit."""

    # the scale formats' ranges: any power of two as E8M0; E4M3 from 2 ** -9 to 448
    @pytest.mark.parametrize("block, e4m3_scale, span", [(32, False, 20.0), (16, True, 6.0)],
                             ids=["e8m0-per-32", "e4m3-per-16"])
    def test_the_bytes_hold_the_rounded_row(self, block, e4m3_scale, span):
        torch.manual_seed(0)
        x = torch.randn(4000, 64) * torch.exp(torch.linspace(-span, span, 4000))[:, None]
        # blocks whose largest value rounds down, all-zero blocks, exact grid values and midpoints
        x[::7, :32] = x[::7, :32].clamp(-3.2, 3.2)
        x[::11, 32:] = 0
        x[::13, :8] = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -6.0])
        x[::17, 8:16] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -5.0])
        for dtype in (torch.float32, torch.bfloat16):
            packed = mx_round.pack_fp4_rows(x.to(dtype), block, e4m3_scale)
            assert packed.dtype == torch.uint8 and packed.shape == (4000, 32 + 64 // block)
            want = mx_round.round_fp4(x.to(dtype), block, e4m3_scale).float()
            assert torch.isfinite(want).all()
            assert torch.equal(mx_round.unpack_fp4_rows(packed, 64, block, e4m3_scale), want)
            # bf16 holds every FP4 value: unpacked straight into it, the same values
            assert torch.equal(mx_round.unpack_fp4_rows(packed, 64, block, e4m3_scale, torch.bfloat16).float(), want)
