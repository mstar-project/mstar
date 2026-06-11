"""Flow-matching utilities for Pi0.5: timestep embedding, Euler step, state discretization."""


import torch


def sincos_timestep_embedding(
    t: torch.Tensor,
    dim: int,
    fraction: torch.Tensor,
    output_buffer: torch.Tensor,
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
    
    period = min_period * torch.pow(max_period / min_period, fraction)
    omega = (2.0 * torch.pi / period).to(t.dtype)

    K = dim // 2
    # angles: (N, K)
    angles = t.unsqueeze(-1) * omega  # omega should be (K,)

    # Fill in-place instead of cat
    output_buffer[:, :K] = torch.sin(angles)
    output_buffer[:, K:] = torch.cos(angles)

    return output_buffer


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
