# Copyright (c) 2026 Robbyant Team.
# Copyright (c) 2024-2025 The Alibaba Wan Team Authors.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the schedule tables built without
# the solver object, with the same numerics.
#
# Original files (Robbyant/lingbot-world, wan/image2video_fast.py and
# wan/utils/fm_solvers_unipc.py) were released under Apache-2.0, with the full
# license text available at https://github.com/Robbyant/lingbot-world/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

"""The causal-fast sampling schedule, ported exactly from the reference.

Provenance: the step picks and re-noising loop follow LingBot-World v1's
``WanI2VFast.generate`` (Robbyant/lingbot-world, Apache-2.0);
``FlowUniPCMultistepScheduler``'s shifted sigmas are Wan-Video/Wan2.1's
(Apache-2.0). The step indices and shift are the V2 1.3B checkpoint's settings
(``config.py``).

The reference builds a ``FlowUniPCMultistepScheduler`` (shift 1), calls
``set_timesteps(1000, shift=shift)`` and picks four entries by index. It never
calls the solver's ``step``: each step converts the flow prediction to x0 and,
except after the last step, re-noises x0 to the next timestep with
``add_noise``. The tables below are built with the same numpy/torch ops, so the
sigmas and int64 timesteps are bit-identical.

The int64 truncation makes neighbouring timesteps collide (about 0.3 timestep
per index near t=1000 at shift 5), and the two lookups resolve a collision
differently: the x0 conversion takes the FIRST match (``argmin``), ``add_noise``
the SECOND (``index_for_timestep``'s ``pos = 1``). Both are kept as is.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FastSchedule:
    sigmas: torch.Tensor  # [N + 1] float32, CPU, trailing 0
    timesteps: torch.Tensor  # [N] int64, CPU
    step_timesteps: tuple[torch.Tensor, ...]  # the chosen entries, 0-dim int64

    def x0_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """``_convert_flow_pred_to_x0``'s sigma: first-match argmin, float64."""
        sigmas = self.sigmas.double()
        timesteps = self.timesteps.double()
        index = torch.argmin((timesteps - timestep.double()).abs())
        return sigmas[index].reshape(-1, 1, 1, 1)

    def noise_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """``add_noise``'s sigma: the second match when the timestep repeats."""
        indices = (self.timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return self.sigmas[indices[pos].item()]


def make_fast_schedule(
    shift: float, timesteps_index: tuple[int, ...], num_train_timesteps: int = 1000,
) -> FastSchedule:
    # FlowUniPCMultistepScheduler.__init__(shift=1)
    alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
    init_sigmas = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)
    init_sigmas = 1.0 * init_sigmas / (1 + (1.0 - 1) * init_sigmas)
    sigma_max = init_sigmas[0].item()
    sigma_min = init_sigmas[-1].item()
    # set_timesteps(num_train_timesteps, shift=shift), final_sigmas_type="zero"
    sigmas = np.linspace(sigma_max, sigma_min, num_train_timesteps + 1).copy()[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * num_train_timesteps
    sigmas = np.concatenate([sigmas, [0]]).astype(np.float32)
    timesteps_t = torch.from_numpy(timesteps).to(dtype=torch.int64)
    return FastSchedule(
        sigmas=torch.from_numpy(sigmas),
        timesteps=timesteps_t,
        step_timesteps=tuple(timesteps_t[i] for i in timesteps_index),
    )


def flow_to_x0(
    flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, schedule: FastSchedule,
) -> torch.Tensor:
    """x0 = x_t - sigma_t * flow, in float64, cast back to the prediction's dtype."""
    original_dtype = flow_pred.dtype
    sigma_t = schedule.x0_sigma(timestep).to(flow_pred.device)
    x0 = xt.double().to(flow_pred.device) - sigma_t * flow_pred.double()
    return x0.to(original_dtype)


def add_noise(
    x0: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor, schedule: FastSchedule,
) -> torch.Tensor:
    """``(1 - sigma) * x0 + sigma * noise`` at the sample's dtype."""
    sigma = schedule.noise_sigma(timestep).to(device=x0.device, dtype=x0.dtype).flatten()
    while sigma.dim() < x0.dim():
        sigma = sigma.unsqueeze(-1)
    return (1 - sigma) * x0 + sigma * noise
