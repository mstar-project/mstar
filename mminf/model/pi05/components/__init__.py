"""Shared components for the Pi0.5 model."""

import torch
from torch import nn


class Pi05GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: ``normed * (1 + weight)``.

    Matches HF Gemma's ``GemmaRMSNorm`` and lerobot's ``PiGemmaRMSNorm`` (in
    its non-conditional path). The ``1 +`` shift is the load-bearing
    difference vs Llama's RMSNorm — using the wrong convention silently
    produces incorrect outputs when loading Gemma weights, since the stored
    weight values are centered around zero (not one).

    The variance is computed in float32 to match the reference exactly.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        var = x.square().mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.variance_epsilon)
        normed = normed * (1.0 + self.weight.to(torch.float32))
        return normed.to(orig_dtype)
