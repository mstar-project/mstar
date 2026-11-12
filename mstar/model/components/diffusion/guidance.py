"""Guidance for the DiT scaffold: what makes a row a guidance branch, and how the branches' predictions combine.

A model declares ``GUIDANCE = Guidance(branches=..., combine=...)`` on its
:class:`~mstar.model.components.diffusion.denoise_loop.DenoiseLoopSubmodule`, and the base does the rest:
it runs each branch as a row of the model's forward, splits the branches over the deployment's ``cfg_size``
ranks, gathers the predictions and combines them, with the request's scales (the knobs the combine names in its
``scales``: ``guidance_scale`` for :data:`cfg`) as per-row step scalars. A model writes the prediction
(``model_output``), never the branches.

Whether guidance is on is decided per request, at intake: a request runs its branches at every step when one of its
scale knobs is above 1 (``active``, which intake and the text encoder ask too), and branch 0 alone otherwise.

Branch order is part of the contract. Branch ``b`` is ``branches[b]``, and ``combine`` receives the
predictions in that order (``cfg``: conditional first). Within a step the rows are request-major
(request 0's branches, then request 1's), and ``attention_segments`` lists a request's segments in the
order of ``key.branches``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

import torch

# The scale knob (``step_metadata``) of a single-scale guidance, and the step scalar it becomes.
GUIDANCE_SCALE = "guidance_scale"


class GuidanceCombine(Protocol):
    """``(predictions in branch order, {scale name: [B, 1, ...] at the predictions' rank}) -> prediction``.

    ``scales`` names the request knobs it reads, the one place they are declared: the base stages each per step
    under its name and hands their values here. Two scales (text and image guidance, say) are two names. A combine
    is an object that carries them, as :class:`Cfg` does: one with a single knob of another name subclasses
    :class:`Cfg` with its own ``scales``, and one with more knobs declares its ``scales`` and ``__call__``."""

    scales: tuple[str, ...]

    def __call__(self, preds: Sequence[torch.Tensor], values: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


class Cfg:
    """Classifier-free guidance, ``u + s (c - u)`` with ``s`` its one scale (``guidance_scale``).

    The references multiply the difference by a Python float, which rounds the product once in the
    prediction's dtype. ``s`` is an fp32 tensor here, so the product is taken in fp32 and cast back
    to match, as ``flow_match.euler_step`` does for its ``dt``."""

    scales = (GUIDANCE_SCALE,)

    def __call__(self, preds: Sequence[torch.Tensor], values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        (scale,) = self.scales
        cond, uncond = preds
        return uncond + (values[scale] * (cond - uncond).float()).to(cond.dtype)


class CfgRenorm(Cfg):
    """:class:`Cfg`, rescaled per token to the conditional prediction's norm over the last dim
    (Qwen-Image's ``comb * |cond| / |comb|``)."""

    def __call__(self, preds: Sequence[torch.Tensor], values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        comb = super().__call__(preds, values)
        return comb * (preds[0].norm(dim=-1, keepdim=True) / comb.norm(dim=-1, keepdim=True))


cfg = Cfg()
cfg_renorm = CfgRenorm()


class Guidance(NamedTuple):
    """A DiT's guidance branches and their combine (see the module docstring)."""

    # Per branch: {model_output argument: the request input it takes}. Inputs no branch names are shared
    # by every branch.
    # ``request_inputs`` returns, under its input name, every input a branch of ``key.branches`` takes (it
    # may skip the others), each the same shape across the branches: the base stacks them as rows. A model
    # that shapes a branch's input per step (Qwen-Image left-pads each prompt to the step's longest) finds
    # the input's name here, ``branches[b][argument]``.
    branches: tuple[Mapping[str, str], ...]
    combine: GuidanceCombine

    @property
    def scales(self) -> tuple[str, ...]:
        """The request knobs ``combine`` reads (its ``scales``)."""
        return self.combine.scales

    def active(self, values: Mapping) -> bool:
        """Whether guidance is on at these values (a request's ``step_metadata``): one of its scales above 1. Every
        place that acts on it (the text encoder skipping the negative prompt, intake, the step key) asks here."""
        return any(float(values[name]) > 1.0 for name in self.scales)
