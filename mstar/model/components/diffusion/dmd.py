"""Few-step DMD sampling for distilled flow models (FastVideo's ``DmdDenoisingStage``).

A DMD-distilled checkpoint (FastWan2.2-TI2V-5B: 3 steps, no CFG) is sampled on a
fixed list of training timesteps rather than a solver schedule. Each step predicts
the clean sample from the flow output and, unless it is the last, noises it back to
the next timestep with fresh Gaussian noise:

    x0      = x_k - sigma_k * v(x_k, t_k)
    x_{k+1} = (1 - sigma_{k+1}) * x0 + sigma_{k+1} * eps_k

Numerics follow FastVideo (``fastvideo/pipelines/basic/wan/stages/dmd.py``, copied
into SGLang's ``stages/denoising_dmd.py``):

* the model is conditioned on the raw integer timestep ``t_k``;
* ``sigma_k`` is looked up in a 1000-entry ``FlowMatchEulerDiscreteScheduler``
  training table built with ``noise_shift`` (FastVideo's DMD_TRAINING_NOISE_SHIFT,
  8.0): the entry whose shifted timestep is nearest ``t_k``, so ``sigma_k`` is close
  to, not equal to, ``t_k / 1000`` (757 -> 0.75675672);
* ``x0`` is computed in float64 and rounded to the flow output's dtype (bf16);
* the re-noise mixes in float32 (``sigma`` is a non-0-dim float32 tensor, which
  promotes the bf16 operands) and rounds to the noise's dtype, the flow's.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from mstar.model.components.diffusion.flow_match import FlowMatchSchedule

# FastVideo's DMD_TRAINING_NOISE_SHIFT (fastvideo/models/wan/definition.py)
DMD_NOISE_SHIFT = 8.0


def dmd_training_table(noise_shift: float, num_train_timesteps: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sigmas, timesteps)`` of the scheduler a DMD stage looks its sigmas up in: the
    constructor's table of ``FlowMatchEulerDiscreteScheduler(shift=noise_shift)``, float32."""
    timesteps = torch.from_numpy(
        np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    )
    sigmas = timesteps / num_train_timesteps
    sigmas = noise_shift * sigmas / (1 + (noise_shift - 1) * sigmas)
    return sigmas, sigmas * num_train_timesteps


def dmd_schedule(
    timesteps: Sequence[int], noise_shift: float = DMD_NOISE_SHIFT, num_train_timesteps: int = 1000,
) -> FlowMatchSchedule:
    """The schedule of a DMD recipe: ``timesteps`` (the model's conditioning) as given, and
    each one's sigma from the training table (terminal 0 appended, as the scaffold reads
    ``sigmas[k + 1]``)."""
    if not timesteps:
        raise ValueError("a DMD schedule needs at least one timestep")
    table_sigmas, table_timesteps = dmd_training_table(noise_shift, num_train_timesteps)
    picked = []
    for t in timesteps:
        # x0 looks t up against the float64 table, the re-noise against the float32
        # one; one entry serves both only while the two agree
        index = int(torch.argmin((table_timesteps.double() - float(t)).abs()))
        if index != int(torch.argmin((table_timesteps - torch.tensor(int(t))).abs())):
            raise ValueError(f"DMD timestep {t} falls between two training-table entries")
        picked.append(table_sigmas[index])
    sigmas = torch.cat([torch.stack(picked), torch.zeros(1, dtype=torch.float32)])
    return FlowMatchSchedule(sigmas=sigmas, timesteps=torch.tensor(list(timesteps), dtype=torch.float32), mu=None)


def dmd_step(
    sample: torch.Tensor, flow: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor,
    noise: torch.Tensor | None,
) -> torch.Tensor:
    """One DMD step: the clean prediction, re-noised to ``sigma_next`` with ``noise``,
    or the clean prediction alone on the last step (``noise`` None).

    ``sigma`` / ``sigma_next`` are float32 tensors with at least one dim (a 0-dim one
    would keep the mix in bf16); the result is in ``flow``'s dtype."""
    x0 = (sample.double() - sigma.double() * flow.double()).to(flow.dtype)
    if noise is None:
        return x0
    return ((1 - sigma_next) * x0.float() + sigma_next * noise.float()).to(noise.dtype)


# --- LightX2V's step distill (WanStepDistillScheduler, lightx2v/models/schedulers/wan/step_distill/scheduler.py) ---


def step_distill_schedule(
    timesteps: Sequence[int], shift: float, num_train_timesteps: int = 1000,
) -> FlowMatchSchedule:
    """The schedule of LightX2V's step-distill recipe: sigma ``shift``-ed from ``t / 1000`` on its
    ``linspace(1, 0, 1001)`` table, and the model conditioned on ``sigma * 1000`` (750 -> 937.5 at
    shift 5), not on ``t``; terminal 0 appended."""
    if not timesteps:
        raise ValueError("a step-distill schedule needs at least one timestep")
    sigmas = torch.linspace(1.0, 0.0, num_train_timesteps + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    picked = sigmas[[num_train_timesteps - int(t) for t in timesteps]]
    return FlowMatchSchedule(
        sigmas=torch.cat([picked, torch.zeros(1, dtype=torch.float32)]),
        timesteps=picked * num_train_timesteps, mu=None,
    )


def step_distill_step(sample: torch.Tensor, flow: torch.Tensor, sigma: float, sigma_next: float | None) -> torch.Tensor:
    """LightX2V's ``step_post``: the clean prediction, moved back out to ``sigma_next`` along the
    predicted flow (no fresh noise; the Euler step on these sigmas), in float32. Python-float sigmas,
    as the reference reads them with ``.item()``; ``sigma_next`` None on the last step."""
    flow = flow.float()
    out = sample.float() - sigma * flow
    if sigma_next is not None:
        out = out + flow * sigma_next
    return out
