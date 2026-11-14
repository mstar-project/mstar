"""Norms and the timestep embedding of the Praxis-1 trunk.

Small pieces, and the ones most likely to be wrong in a way that still runs.
Three conventions here differ from what a reader would assume, and each
produces plausible output when inverted:

* the backbone's norms scale by ``1 + weight``, Gemma style, so a checkpoint
  tensor whose mean is near 0 is storing ``weight - 1`` rather than a weight
  near 1,
* the backbone and the expert disagree. The backbone norms by ``1 + weight``
  at eps ``1e-6``; the expert's has no weight at all and uses eps ``1e-4``,
  and its output norm is a LayerNorm, which subtracts the mean,
* the timestep is not a sinusoidal ladder. It is fixed random Fourier
  features read from two non-trainable checkpoint buffers, and the angle is
  rounded back to the input dtype *before* the cosine.

The reference also keeps two RMS norm classes with different compute
dtypes: one casts to float32 and back, the other calls ``F.rms_norm`` in the
input's dtype. Both are here, and which one a layer takes is the layer's
business, not a detail to unify.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ZeroCenteredRMSNorm(nn.Module):
    """RMS norm scaled by ``1 + weight``, computed in the input's dtype.

    The reference uses this for the text projection's tail and the decoder
    output projection's head. Unlike :class:`Float32RMSNorm` it does not
    cast the input or the weight, so it hands ``F.rms_norm`` whatever dtype
    arrives and leaves the internal accumulation to the kernel.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x, (self.dim,), weight=1 + self.weight, eps=self.eps
        )


class Float32RMSNorm(nn.Module):
    """RMS norm scaled by ``1 + weight``, computed in float32.

    The reference's ``layers.RMSNorm``: input and weight are cast to float32,
    normed, and cast back. This is what the trunk's ``ln_in``, ``ln_out``,
    ``q_norm`` and ``k_norm`` use, and the cast is the difference from
    :class:`ZeroCenteredRMSNorm` rather than an implementation detail.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast first, then add. The reference casts the weight to float32
        # and only then forms 1 + w, so a bfloat16 weight gains the 1 in
        # float32. Doing (1 + w).float() instead rounds the sum in bfloat16
        # before the cast, which is a different number and still runs.
        out = torch.nn.functional.rms_norm(
            x.float(), (self.dim,), weight=1 + self.weight.to(torch.float32), eps=self.eps
        )
        return out.to(x.dtype)


def expert_rms_norm(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """The action expert's norm: no weight, eps ``1e-4``, mixed precision.

    Deliberately not the backbone's, and deliberately not ``F.rms_norm``.
    The expert carries no scale here at all, its epsilon is two orders
    larger, and the precision is split: the **square runs in the input's
    dtype**, while the mean, the reciprocal square root and the product run
    in float32 before the cast back.

    That split is the part a port loses. Calling ``F.rms_norm`` instead would
    square in whatever the kernel chooses, which on bfloat16 input is a
    different sum of a different set of roundings.
    """
    mean_sq = torch.mean(x**2, dim=-1, keepdim=True, dtype=torch.float32)
    scale = torch.rsqrt(mean_sq + eps)
    return (x.to(torch.float32) * scale).to(x.dtype)


class TimestepEmbedding(nn.Module):
    """Fixed random Fourier features of the flow timestep.

    ``freqs`` and ``phases`` are checkpoint buffers rather than learned
    parameters, so this module is meaningless with random init and the
    weight loader has to fill it.

    The rounding is load bearing. The product is formed in float32, cast
    back to the timestep's dtype, and only then taken through the cosine, so
    a bfloat16 timestep quantizes the angle before the cosine sees it.
    Keeping the whole thing in float32 would be more accurate and would not
    match the checkpoint.
    """

    freqs: torch.Tensor
    phases: torch.Tensor

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.zeros(1, dim))
        self.register_buffer("phases", torch.zeros(1, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``(B,)`` timesteps to ``(B, dim)`` features."""
        with torch.no_grad():
            x = t[..., None].float() * self.freqs.float() + self.phases.float()
            x = x.to(t.dtype)
            return (x * 2 * math.pi).cos() * 0.5
