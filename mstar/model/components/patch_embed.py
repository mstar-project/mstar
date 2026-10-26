"""Patch embeddings computed as one GEMM."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LinearPatchEmbed(nn.Module):
    """A patchifying conv (kernel = stride, no padding) run as ``F.linear`` over patches that arrive already cut.

    Each output row is then the dot product of one flattened patch with each flattened filter, so the conv and
    the GEMM compute the same sums. On AMD GPUs MIOpen's first conv call at each new input size can take seconds, and
    a GEMM has no per-size setup.
    """

    def __init__(self, proj: nn.Conv2d | nn.Conv3d):
        super().__init__()
        if (tuple(proj.kernel_size) != tuple(proj.stride) or any(proj.padding) or proj.groups != 1
                or any(d != 1 for d in proj.dilation)):
            raise ValueError(f"{proj} is not a patchifying conv: kernel must equal stride, without padding, "
                             "groups or dilation")
        self.weight = nn.Parameter(proj.weight.detach().flatten(1), requires_grad=False)
        self.bias = None if proj.bias is None else nn.Parameter(proj.bias.detach(), requires_grad=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches`` flattens to one row per patch, channels first, in the conv's input layout; returns
        ``[num_patches, out_channels]``."""
        rows = patches.reshape(-1, self.weight.shape[1]).to(self.weight.dtype)
        return F.linear(rows, self.weight, self.bias)
