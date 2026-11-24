"""Flow-matching utilities for Pi0.5: timestep embedding, Euler step, state discretization, and Flow-SDE."""

import math
from collections.abc import Mapping
from typing import Any

import torch

_ROLLOUT_ALLOWED_KEYS = frozenset({"sde_type", "noise_level", "sde_step", "return_state", "ignore_last"})
_ROLLOUT_SDE_TYPES = frozenset({"flow_sde", "flow_ode"})


def normalize_rollout_config(raw: Any, num_steps: int) -> dict[str, Any] | None:
    """Validate and normalize an opt-in Pi0.5 ``rollout`` request block."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"rollout must be a mapping, got {type(raw).__name__}")
    unknown = sorted(k for k in raw if k not in _ROLLOUT_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"rollout has unsupported keys: {unknown}")

    sde_type = raw.get("sde_type")
    if not isinstance(sde_type, str) or sde_type not in _ROLLOUT_SDE_TYPES:
        raise ValueError(
            f"rollout.sde_type must be one of {sorted(_ROLLOUT_SDE_TYPES)}, got {sde_type!r}"
        )

    noise_level = raw.get("noise_level", 0.5)
    if isinstance(noise_level, bool) or not isinstance(noise_level, (int, float)):
        raise ValueError(f"rollout.noise_level must be a finite non-negative float, got {noise_level!r}")
    noise_level_f = float(noise_level)
    if not math.isfinite(noise_level_f) or noise_level_f < 0.0:
        raise ValueError(f"rollout.noise_level must be finite and >= 0.0, got {noise_level!r}")

    sde_step = raw.get("sde_step", "random")
    if sde_step != "random":
        if isinstance(sde_step, bool) or not isinstance(sde_step, int) or not 0 <= sde_step < num_steps:
            raise ValueError(
                f"rollout.sde_step must be 'random' or an int in [0, {num_steps - 1}], got {sde_step!r}"
            )

    return_state = raw.get("return_state", True)
    if not isinstance(return_state, bool):
        raise ValueError(f"rollout.return_state must be a bool, got {return_state!r}")

    ignore_last = raw.get("ignore_last", False)
    if not isinstance(ignore_last, bool):
        raise ValueError(f"rollout.ignore_last must be a bool, got {ignore_last!r}")

    return {
        "sde_type": sde_type,
        "noise_level": noise_level_f,
        "sde_step": sde_step,
        "return_state": return_state,
        "ignore_last": ignore_last,
    }


def flow_sde_timesteps(num_steps: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Descending fp32 schedule ``[1.0, ..., 1/N, 0.0]`` of length ``num_steps + 1``."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    return torch.cat(
        [
            torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=torch.float32),
            torch.zeros(1, device=device, dtype=torch.float32),
        ]
    )


def flow_sde_sigmas(timesteps: torch.Tensor, noise_level: float | torch.Tensor) -> torch.Tensor:
    """Flow-SDE noise schedule in fp32, handling ``t=1.0`` via ``timesteps[1]``."""
    timesteps_f = timesteps.to(torch.float32)
    denom_timesteps = torch.where(timesteps_f == 1.0, timesteps_f[1], timesteps_f)
    return noise_level * torch.sqrt(timesteps_f / (1.0 - denom_timesteps))


def sample_mean_var(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    t: torch.Tensor | float,
    t_next: torch.Tensor | float,
    sigma_i: torch.Tensor | float,
    is_sde: torch.Tensor | bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fp32 transition mean and std for one Flow-ODE or Flow-SDE step."""
    x_t_f = x_t.to(torch.float32)
    v_t_f = v_t.to(torch.float32)
    t_f = torch.as_tensor(t, device=x_t_f.device, dtype=torch.float32)
    t_next_f = torch.as_tensor(t_next, device=x_t_f.device, dtype=torch.float32)
    sigma_f = torch.as_tensor(sigma_i, device=x_t_f.device, dtype=torch.float32)
    dt = t_f - t_next_f

    x0_pred = x_t_f - v_t_f * t_f
    x1_pred = x_t_f + v_t_f * (1.0 - t_f)
    w0 = 1.0 - t_next_f
    sde_w1 = t_next_f - (sigma_f * sigma_f * dt) / (2.0 * t_f)
    sde_std = torch.sqrt(dt) * sigma_f
    if isinstance(is_sde, torch.Tensor):
        w1 = torch.where(is_sde, sde_w1, t_next_f)
        x_t_std = torch.where(is_sde, sde_std, torch.zeros_like(sde_std))
    elif is_sde:
        w1 = sde_w1
        x_t_std = sde_std
    else:
        w1 = t_next_f
        x_t_std = torch.zeros_like(sde_std)
    x_t_mean = x0_pred * w0 + x1_pred * w1
    return x_t_mean, x_t_std


def gaussian_logprob(
    x_next: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor | float,
) -> torch.Tensor:
    """Elementwise fp32 Gaussian log-probability, returning ``0.0`` where ``x_std == 0``."""
    x_next_f = x_next.to(torch.float32)
    x_mean_f = x_mean.to(torch.float32)
    x_std_f = torch.as_tensor(x_std, device=x_next_f.device, dtype=torch.float32)
    zero_std = x_std_f == 0
    safe_std = torch.where(zero_std, torch.ones_like(x_std_f), x_std_f)
    logprob = (
        -torch.log(safe_std)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * torch.square((x_next_f - x_mean_f) / safe_std)
    )
    return torch.where(zero_std, torch.zeros_like(logprob), logprob)


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
            ``Pi05ActionSubmodule._get_timestep_emb_fraction``).
        output_buffer: ``(len(t), dim)`` destination, written in place and
            returned.
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
