"""MiniMax-H3's two rectified-flow schedules (video shift 12, audio 3) and its step, as
``MiniMaxH3Scheduler``: ``linspace(1, 0, N)`` shifted in fp32 and deduplicated, so N
points run N - 1 forwards; the model conditions on ``t = 1 - sigma`` and predicts
``x0 = x_t + (1 - t) v``. ``H3Schedule`` adds the audio schedule and the keyframe rows'
timestep to the scaffold's ``FlowMatchSchedule``.

A few-step distilled checkpoint (FastVideo's FastH3) is sampled by the same step on its
trained rungs instead of the linspace grid: ``rung / 1000`` and a terminal 0, through each
modality's shift (``rung_sigmas``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.minimax_h3.config import KEYFRAME_NOISE_AUG


def _shifted(base: torch.Tensor, shift: float) -> torch.Tensor:
    """The exponential shift of an fp32 grid in [0, 1]."""
    if not shift > 0:
        raise ValueError(f"the sigma shift must be positive, got {shift}")
    sigmas = shift * base / (1 + (shift - 1) * base)
    if not torch.isfinite(sigmas).all():
        # a shift past fp32's range (or one that underflows) gives inf / inf
        raise ValueError(f"the sigma shift {shift} gives a non-finite grid in fp32")
    return sigmas


def shifted_sigmas(num_points: int, shift: float) -> torch.Tensor:
    """``MiniMaxH3Scheduler.set_timesteps``'s sigma grid, fp32 on the CPU."""
    if num_points < 2:
        raise ValueError(f"num_inference_steps must be >= 2, got {num_points}")
    return torch.unique_consecutive(_shifted(torch.linspace(1.0, 0.0, int(num_points), dtype=torch.float32), shift))


def rung_sigmas(rungs: Sequence[int], shift: float) -> torch.Tensor:
    """A distilled checkpoint's grid: its rungs over 1000 and a terminal 0, shifted in fp32, as
    FastVideo's ``MiniMaxH3DenoisingStage._set_shifted_schedule`` builds it (and SGLang's
    ``minimax_h3_time_shift_sigmas``). One forward per rung."""
    if (not rungs or any(isinstance(r, bool) or not isinstance(r, int) or not 0 < r <= 1000 for r in rungs)
            or any(a <= b for a, b in zip(rungs, rungs[1:], strict=False))):
        raise ValueError(f"the trained rungs must be decreasing integers in (0, 1000], got {list(rungs)}")
    sigmas = _shifted(torch.tensor([r / 1000.0 for r in rungs] + [0.0], dtype=torch.float32), shift)
    if not bool((sigmas[1:] < sigmas[:-1]).all()):
        raise ValueError(f"the rungs {list(rungs)} collide in fp32 at shift {shift}")
    return sigmas


@dataclass(frozen=True)
class H3Schedule(FlowMatchSchedule):
    """The video schedule (the base's ``sigmas`` / ``timesteps``), the audio one, and the
    timestep keyframe rows are held at each step, ``max(t_video, 0.999)``."""

    audio_sigmas: torch.Tensor | None = None
    audio_timesteps: torch.Tensor | None = None
    cond_timesteps: torch.Tensor | None = None


class ScheduleArgs(NamedTuple):
    steps: int
    video_shift: float
    audio_shift: float
    rungs: tuple[int, ...] | None = None  # a distilled checkpoint's trained rungs


def h3_schedule(num_inference_steps: int, video_shift: float, audio_shift: float,
                rungs: Sequence[int] | None = None) -> H3Schedule:
    """``num_inference_steps`` grid points, as the reference counts them; on ``rungs``, one more
    than there are rungs."""
    if rungs is None:
        video = shifted_sigmas(num_inference_steps, video_shift)
        audio = shifted_sigmas(num_inference_steps, audio_shift)
    else:
        if num_inference_steps != len(rungs) + 1:
            raise ValueError(
                f"the checkpoint is distilled for {len(rungs)} steps: num_inference_steps counts sigma points and "
                f"must be {len(rungs) + 1}, got {num_inference_steps}"
            )
        video, audio = rung_sigmas(rungs, video_shift), rung_sigmas(rungs, audio_shift)
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
