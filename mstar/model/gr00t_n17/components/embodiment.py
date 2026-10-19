"""Per-embodiment MLPs: one set of weights per robot, picked by its id.

GR00T serves 32 embodiments from one checkpoint by giving several small MLPs a
leading category axis and indexing it per request. The state encoder is the
first user; the action encoder and decoder are the others.

Class and parameter names are the reference's (``CategorySpecificLinear.W`` /
``.b``, ``layer1`` / ``layer2``), so the checkpoint's keys map straight on.
"""
from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """``[B, T]`` timesteps to ``[B, T, embedding_dim]``, the action encoder's
    own encoding of the denoising time.

    This is **not** the convention the DT's ``timestep_encoder`` uses, and the
    two live in one model: that one is diffusers' ``Timesteps`` with
    ``flip_sin_to_cos=True``, this one is sin then cos with no flip, and the
    frequency spacing here is ``log(10000) / half_dim`` where the usual form
    divides by ``half_dim - 1``. The module has no parameters, so nothing in
    the checkpoint would reveal a wrong convention -- only a comparison
    against the reference's own class does (C06 CHK-1).

    The exponent is built in **fp32 throughout**, as the reference does.
    Computing it in fp64 and casting at the end diverges by ~1e-6 (measured);
    ``math.log`` in place of ``torch.log`` does not -- the two are bit-identical
    at every width here, so that is a style choice and the dtype is not.

    One more reason the arithmetic is spelled out rather than delegated to
    diffusers' ``Timesteps(flip_sin_to_cos=False, downscale_freq_shift=0)``,
    which is the same maths by a different route: at a width whose ``half_dim``
    is a power of two the two agree bit for bit, but at the checkpoint's 1536
    (``half_dim`` 768) they part by ~6e-05 -- too large to call rounding, too
    small for any bar that would still reject a wrong convention.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # The reference takes rank 2 and nothing else (`B, T = timesteps.shape`).
        # Keeping that is the one thing stopping this module from being fed a
        # `[B]` tensor meant for the DT's `TimestepEncoder`: that returns
        # `[B, dim]` too, so a crossed call site would change no shape, raise
        # nothing, and produce plausible magnitudes under a different sinusoid
        # convention (found by C06 CHK-1's QA).
        if timesteps.dim() != 2:
            raise ValueError(
                f"the action encoder's timesteps are [B, T], got "
                f"{list(timesteps.shape)}; the DT's TimestepEncoder is the "
                "one that takes [B]"
            )
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=timesteps.device,
        ) * (torch.log(torch.tensor(10000.0)) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """``x @ W[id] + b[id]``, with ``W`` held per embodiment."""

    def __init__(self, num_categories: int, input_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.empty(num_categories, output_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T, input_dim]`` and ``cat_ids`` is ``[B]``."""
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two per-embodiment linears with a ReLU between."""

    def __init__(
        self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int,
    ):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(x, cat_ids)), cat_ids)


def build_state_encoder(config) -> CategorySpecificMLP:
    """Raw state in, one conditioning token out (``[B, 1, input_embedding_dim]``).

    The node that owns this names it ``state_encoder``, because every
    action-head node strips ``action_head.`` and keeps the reference's own
    attribute names below it (``weight_loader.NODE_PREFIXES``).
    """
    return CategorySpecificMLP(
        num_categories=config.max_num_embodiments,
        input_dim=config.max_state_dim,
        hidden_dim=config.hidden_size,
        output_dim=config.input_embedding_dim,
    )
