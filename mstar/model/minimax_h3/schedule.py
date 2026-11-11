"""MiniMax-H3's two rectified-flow schedules (video shift 12, audio 3) and its step, as
``MiniMaxH3Scheduler``: ``linspace(1, 0, N)`` shifted in fp32 and deduplicated, so N
points run N - 1 forwards; the model conditions on ``t = 1 - sigma`` and predicts
``x0 = x_t + (1 - t) v``. ``H3Schedule`` adds the audio schedule and the keyframe rows'
timestep to the scaffold's ``FlowMatchSchedule``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.minimax_h3.config import KEYFRAME_NOISE_AUG


def shifted_sigmas(num_points: int, shift: float) -> torch.Tensor:
    """``MiniMaxH3Scheduler.set_timesteps``'s sigma grid, fp32 on the CPU."""
    if num_points < 2:
        raise ValueError(f"num_inference_steps must be >= 2, got {num_points}")
    if not shift > 0:
        raise ValueError(f"the sigma shift must be positive, got {shift}")
    base = torch.linspace(1.0, 0.0, int(num_points), dtype=torch.float32)
    sigmas = torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))
    if not torch.isfinite(sigmas).all():
        # a shift past fp32's range (or one that underflows) gives inf / inf
        raise ValueError(f"the sigma shift {shift} gives a non-finite grid in fp32")
    return sigmas


@dataclass(frozen=True)
class H3Schedule(FlowMatchSchedule):
    """The video schedule (the base's ``sigmas`` / ``timesteps``), the audio one, and the
    timestep keyframe rows are held at each step, ``max(t_video, 0.999)``."""

    audio_sigmas: torch.Tensor | None = None
    audio_timesteps: torch.Tensor | None = None
    cond_timesteps: torch.Tensor | None = None


def h3_schedule(num_inference_steps: int, video_shift: float, audio_shift: float) -> H3Schedule:
    video = shifted_sigmas(num_inference_steps, video_shift)
    audio = shifted_sigmas(num_inference_steps, audio_shift)
    if video.shape != audio.shape:
        # a shift collapsed grid points in one schedule only; the reference then steps
        # the two out of step, which no released setting does
        raise ValueError(
            f"num_inference_steps={num_inference_steps}: the video and audio grids have "
            f"{video.shape[0]} and {audio.shape[0]} points"
        )
    timesteps = 1.0 - video[:-1]
    return H3Schedule(
        sigmas=video, timesteps=timesteps, mu=None,
        audio_sigmas=audio, audio_timesteps=1.0 - audio[:-1],
        cond_timesteps=torch.clamp(timesteps, min=KEYFRAME_NOISE_AUG),
    )


def h3_step(sample: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor, sigma: torch.Tensor,
            sigma_next: torch.Tensor) -> torch.Tensor:
    """``MiniMaxH3Scheduler.step`` (eta 0), fp32. ``sample`` / ``velocity`` ``[B, L, C]``;
    the scalars ``[B, 1, 1]``. The x0 estimate recovers sigma from ``t`` (the
    reference keeps the two sources apart: ``1 - (1 - sigma)`` is not exact in fp32)."""
    denoised = sample + (1 - timestep) * velocity.float()
    ratio = sigma_next / sigma
    return ratio * sample + (1.0 - ratio) * denoised
