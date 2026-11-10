"""Guidance for the DiT scaffold: what makes a row a guidance branch, and how the branches' predictions combine.

A model declares ``GUIDANCE = Guidance(branches=..., combine=...)`` on its
:class:`~mstar.model.components.diffusion.denoise_loop.DenoiseLoopSubmodule`, and the base does the rest:
it runs each branch as a row of the model's forward, splits the branches over the deployment's ``cfg_size``
ranks, gathers the predictions and combines them, with the request's ``guidance_scale`` as a per-row
step scalar. A model writes the prediction (``model_output``), never the branches.

Branch order is part of the contract. Branch ``b`` is ``branches[b]``, and ``combine`` receives the
predictions in that order (``cfg``: conditional first). Within a step the rows are request-major
(request 0's branches, then request 1's), and ``attention_segments`` lists a request's segments in the
order of ``key.branches``. A model whose branches read KV labels instead of inputs (empty mappings) lists
the sources of a ``combined_labels`` plan in branch order as well, since a combined plan concatenates its
source labels in the order of that tuple.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

import torch

# The request knob every guided model reads (``step_metadata``), and its step scalar's source.
GUIDANCE_SCALE = "guidance_scale"
GUIDANCE_SCALES = "guidance_scales"


class GuidanceCombine(Protocol):
    """``(predictions in branch order, scale [B, 1, ...] at the predictions' rank) -> prediction``."""

    def __call__(self, preds: Sequence[torch.Tensor], scale: torch.Tensor) -> torch.Tensor: ...


def cfg(preds: Sequence[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    """Classifier-free guidance, ``u + s (c - u)``.

    The references multiply the difference by a Python float, which rounds the product once in the
    prediction's dtype. ``scale`` is an fp32 tensor here, so the product is taken in fp32 and cast back
    to match, as ``flow_match.euler_step`` does for its ``dt``."""
    cond, uncond = preds
    return uncond + (scale * (cond - uncond).float()).to(cond.dtype)


def cfg_renorm(preds: Sequence[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    """:func:`cfg`, rescaled per token to the conditional prediction's norm over the last dim
    (Qwen-Image's ``comb * |cond| / |comb|``)."""
    comb = cfg(preds, scale)
    return comb * (preds[0].norm(dim=-1, keepdim=True) / comb.norm(dim=-1, keepdim=True))


class Guidance(NamedTuple):
    """A DiT's guidance branches and their combine (see the module docstring)."""

    # Per branch: {model_output argument: the request input it takes}. Inputs no branch names are shared
    # by every branch. An empty mapping is a branch that differs by what its row reads, not by an input.
    branches: tuple[Mapping[str, str], ...]
    combine: GuidanceCombine

    def active(self, step_metadata: Mapping) -> bool:
        """Whether a request runs its branches: a scale above 1. Every place that acts on it (the
        text encoder skipping the negative prompt, intake, the step key) asks here."""
        return float(step_metadata[GUIDANCE_SCALE]) > 1.0
