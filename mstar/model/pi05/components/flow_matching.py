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
        fraction: ``(dim // 2,)`` frequency ladder, caller-owned so it is
            allocated once rather than per step (see
            ``Pi05LLMSubmodule._get_timestep_emb_fraction``).
        output_buffer: ``(len(t), dim)`` destination, written in place and
            returned; preallocated so the action-gen step stays capturable.
        min_period / max_period: frequency range for the sinusoidal basis.

    Returns:
        ``output_buffer``, filled with the sin/cos embedding.
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


def discretize_state(state: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    """Bin a normalized robot state the way openpi's PaligemmaTokenizer does.

    That is ``np.digitize(state, np.linspace(-1, 1, num_bins + 1)[:-1]) - 1``:
    equal bins over [-1, 1], with values below -1 giving -1 and values at or
    above 1 giving ``num_bins - 1``. lerobot's pi0.5 processor bins the same way.
    """
    edges = torch.linspace(-1.0, 1.0, num_bins + 1, dtype=torch.float64, device=state.device)[:-1]
    return torch.bucketize(state.to(torch.float64), edges, right=True) - 1
