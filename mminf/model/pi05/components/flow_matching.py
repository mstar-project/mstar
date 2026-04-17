"""Flow-matching utilities for Pi0.5: timestep embedding, Euler step, state discretization."""

import math

import torch


def sincos_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (matches the openpi reference).

    Args:
        t: scalar or 1D tensor of timesteps in [0, 1].
        dim: embedding dimension. Must be even.
        min_period / max_period: frequency range for the sinusoidal basis.

    Returns:
        Tensor of shape ``(*t.shape, dim)`` with the sin/cos embedding.
    """
    if dim % 2 != 0:
        raise ValueError(f"sincos embedding requires even dim, got {dim}")
    if t.dim() == 0:
        t = t[None]
    half = dim // 2
    # Geometric progression of frequencies between min_period and max_period.
    # Use float64 for the frequency computation to match the openpi reference;
    # bf16 has only ~3 digits of precision and rounds higher-frequency
    # components, which compounds through time_mlp -> adaRMS -> 18 layers.
    fraction = torch.linspace(0.0, 1.0, half, device=t.device, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    omega = (2.0 * math.pi / period).to(t.dtype)
    angles = t[..., None] * omega
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def euler_step(
    x: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """One Euler integration step: x_{t+dt} = x_t + dt * v(x_t, t)."""
    return x + dt * velocity


def discretize_state(
    state: torch.Tensor,
    num_bins: int = 256,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> torch.Tensor:
    """Map continuous robot state values in [value_min, value_max] to bin indices.

    Used by Pi0.5 to tokenize proprioceptive state into language tokens that
    PaliGemma can embed alongside image and language tokens.

    Args:
        state: 1D tensor of state values.
        num_bins: number of discrete bins.
        value_min / value_max: assumed range of normalized state values.

    Returns:
        1D ``torch.long`` tensor of bin indices in ``[0, num_bins)``.
    """
    clamped = state.clamp(value_min, value_max)
    normalized = (clamped - value_min) / (value_max - value_min)
    indices = (normalized * (num_bins - 1)).round().to(torch.long)
    return indices.clamp(0, num_bins - 1)
