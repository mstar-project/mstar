"""CPU noise and flow times for the Praxis-1 window sampler."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

VIDEO_SALT = 0
ACTION_SALT = 1


def seeded_normal(shape: Sequence[int], seed: int, salt: int) -> torch.Tensor:
    """Draw float32 CPU noise with the reference's independent stream salt."""
    generator_seed = (int(seed) * 1_000_003 + int(salt)) & 0x7FFF_FFFF_FFFF_FFFF
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    return torch.randn(tuple(shape), generator=generator)


def window_time_grid(num_steps: int, shift: float) -> tuple[float, ...]:
    """Descending float64 Euler times, including the final zero."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("t_shift must be finite and positive")
    times = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    if shift != 1.0:
        times = shift * times / (1.0 + (shift - 1.0) * times)
    return tuple(float(t) for t in times)
