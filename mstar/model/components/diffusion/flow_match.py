"""Flow-matching sigma schedules and the Euler step, as the DiT scaffold's step math.

One schedule object per request: built once from the checkpoint's scheduler
config and the request's ``(num_inference_steps, image_seq_len)``, then read
per step. Every denoise loop in the scaffold (FLUX.2-klein, Z-Image, ...) goes
through this module so the schedule math lives in one place and is tested
against the reference scheduler once.

Numerics follow diffusers ``FlowMatchEulerDiscreteScheduler`` to the bit:

* the base grid is ``np.linspace(1.0, 1 / N, N)`` (float64) cast to float32, as
  the FLUX.2 pipelines hand it to ``set_timesteps``, or float32 ``torch.linspace``
  (``torch_linspace=True``) as the Z-Image pipeline does; the two differ by one
  ulp for most ``N`` (they agree at 1, 2, 4, 8, 16);
* the shift is applied in float32 numpy arithmetic with a Python-float ``mu``
  (NumPy's NEP-50 promotion keeps float32), so the sigmas equal the reference's
  ``self.sigmas`` exactly;
* ``timesteps = sigmas * num_train_timesteps`` and a terminal ``0`` is appended
  to ``sigmas``;
* the Euler step upcasts the sample to fp32, uses ``dt = sigma_next - sigma``
  and casts the result back to the model output's dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class ShiftMode(Enum):
    """How the uniform sigma grid is bent toward the noisy end."""

    NONE = "none"                    # sigmas as sampled
    LINEAR = "linear"                # shift * s / (1 + (shift - 1) * s)   (use_dynamic_shifting=False)
    DYNAMIC = "dynamic"              # exp(mu) / (exp(mu) + (1/s - 1)), mu from calculate_shift (FLUX.1 style)
    EMPIRICAL_MU = "empirical_mu"    # same time shift; mu from the FLUX.2 empirical (seq_len, steps) fit


@dataclass(frozen=True)
class FlowMatchConfig:
    """The scheduler facts a checkpoint pins (``scheduler/scheduler_config.json``)."""

    num_train_timesteps: int = 1000
    shift: float = 3.0
    shift_mode: ShiftMode = ShiftMode.NONE
    # DYNAMIC (calculate_shift) parameters
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    # The pipeline's base grid: float64 numpy linspace cast to float32 (FLUX.2) or
    # float32 torch.linspace (Z-Image). Not in the scheduler config; the pipeline picks.
    torch_linspace: bool = False

    @classmethod
    def from_scheduler_config(cls, cfg: dict, empirical_mu: bool = False) -> "FlowMatchConfig":
        """Build from a diffusers ``FlowMatchEulerDiscreteScheduler`` config dict.

        ``empirical_mu`` selects the FLUX.2 pipelines' ``compute_empirical_mu``
        over the FLUX.1-style ``calculate_shift`` when dynamic shifting is on;
        the scheduler config alone cannot tell the two apart (the pipeline picks).
        """
        for key in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "invert_sigmas"):
            if cfg.get(key):
                raise NotImplementedError(f"scheduler option {key}=True is not supported by FlowMatchSchedule")
        if cfg.get("shift_terminal"):
            raise NotImplementedError("scheduler shift_terminal is not supported by FlowMatchSchedule")
        if cfg.get("time_shift_type", "exponential") != "exponential":
            raise NotImplementedError("only the exponential time shift is supported")
        if cfg.get("use_dynamic_shifting", False):
            mode = ShiftMode.EMPIRICAL_MU if empirical_mu else ShiftMode.DYNAMIC
        else:
            mode = ShiftMode.LINEAR if float(cfg.get("shift", 1.0)) != 1.0 else ShiftMode.NONE
        return cls(
            num_train_timesteps=int(cfg.get("num_train_timesteps", 1000)),
            shift=float(cfg.get("shift", 1.0)),
            shift_mode=mode,
            base_shift=float(cfg.get("base_shift", 0.5)),
            max_shift=float(cfg.get("max_shift", 1.15)),
            base_image_seq_len=int(cfg.get("base_image_seq_len", 256)),
            max_image_seq_len=int(cfg.get("max_image_seq_len", 4096)),
        )


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """FLUX.1-style resolution-dependent ``mu`` (diffusers ``calculate_shift``)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """FLUX.2's empirical ``mu`` fit over (token count, step count)
    (diffusers ``pipelines.flux2.compute_empirical_mu``)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


@dataclass(frozen=True)
class FlowMatchSchedule:
    """One request's denoise schedule.

    ``sigmas`` has ``num_steps + 1`` entries (terminal 0 appended), ``timesteps``
    has ``num_steps``. Both are float32 CPU tensors, bit-identical to the
    reference scheduler's. Step ``k`` uses ``sigmas[k] -> sigmas[k + 1]`` and
    conditions the model on ``timesteps[k]``.
    """

    sigmas: torch.Tensor
    timesteps: torch.Tensor
    mu: float | None

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.shape[0])

    @classmethod
    def build(cls, config: FlowMatchConfig, num_steps: int, image_seq_len: int) -> "FlowMatchSchedule":
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        # The pipelines' explicit grid, then the scheduler's float32 numpy arithmetic.
        if config.torch_linspace:
            sigmas = torch.linspace(1.0, 1 / num_steps, num_steps, dtype=torch.float32).numpy()
        else:
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps).astype(np.float32)
        mu: float | None = None
        if config.shift_mode is ShiftMode.EMPIRICAL_MU:
            mu = compute_empirical_mu(image_seq_len, num_steps)
        elif config.shift_mode is ShiftMode.DYNAMIC:
            mu = calculate_shift(
                image_seq_len, config.base_image_seq_len, config.max_image_seq_len,
                config.base_shift, config.max_shift,
            )
        if mu is not None:
            # scheduler._time_shift_exponential, in float32 numpy with Python-float mu
            sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** 1.0)
        elif config.shift_mode is ShiftMode.LINEAR:
            sigmas = config.shift * sigmas / (1 + (config.shift - 1) * sigmas)
        sigmas_t = torch.from_numpy(np.ascontiguousarray(sigmas)).to(torch.float32)
        timesteps = sigmas_t * config.num_train_timesteps
        sigmas_t = torch.cat([sigmas_t, torch.zeros(1, dtype=torch.float32)])
        return cls(sigmas=sigmas_t, timesteps=timesteps, mu=mu)


def euler_step(
    sample: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor,
) -> torch.Tensor:
    """One flow-matching Euler update, ``x_{k+1} = x_k + (sigma_{k+1} - sigma_k) * v``.

    ``sigma`` / ``sigma_next`` are float32 tensors: a scalar, or one value per
    request (``[B, 1, 1]`` as the denoise loop stacks them) for a batch at
    different steps; a per-request ``dt`` is reshaped to ``[B, 1, ...]`` at the
    sample's rank, so ``[B, L, C]`` token layouts and ``[B, C, H, W]`` latent
    layouts both broadcast over the batch (right-aligned broadcasting would pair
    the batch with the channel dimension of a 4-D latent). Every operand is a
    tensor so the step captures into a CUDA graph with per-request values staged
    into static buffers.

    Op order and dtypes follow the reference scheduler's ``step`` exactly: the
    sample is upcast to fp32, but ``dt * velocity`` is evaluated in the
    velocity's dtype — the reference's ``dt`` is a 0-dim fp32 tensor, and torch
    promotes a 0-dim operand to the other operand's dtype, so the product is
    bf16 with a bf16-rounded ``dt``. A batched ``[B, 1, 1]`` ``dt`` would instead
    promote the product to fp32, hence the explicit cast; the sum is fp32 and
    the result is cast back to ``velocity.dtype``.
    """
    dt = (sigma_next - sigma).to(velocity.dtype)
    if dt.ndim > 0 and dt.ndim != velocity.ndim:
        dt = dt.reshape(dt.shape[0], *([1] * (velocity.ndim - 1)))
    prev = sample.to(torch.float32) + dt * velocity
    return prev.to(velocity.dtype)
