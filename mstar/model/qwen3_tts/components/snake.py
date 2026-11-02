"""Fused SnakeBeta activation for the 12 Hz codec decoder.

The reference ``SnakeBeta`` (``x + 1/exp(beta) * sin(x * exp(alpha))**2``) is
the shared Snake kernel (``mstar.model.components.snake``) with the per-channel
``exp`` terms precomputed (the weights are frozen at inference).
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.components.snake import snake


class FusedSnakeBeta(nn.Module):
    """Drop-in for the reference ``SnakeBeta`` built from its loaded parameters."""

    def __init__(self, reference: nn.Module):
        super().__init__()
        with torch.no_grad():
            alpha = torch.exp(reference.alpha.detach().float())
            inv_beta = 1.0 / (torch.exp(reference.beta.detach().float()) + reference.no_div_by_zero)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("inv_beta", inv_beta, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return snake(hidden_states, self.alpha, self.inv_beta)


def fuse_snake_activations(module: nn.Module) -> int:
    """Swap every reference ``SnakeBeta`` under ``module`` for ``FusedSnakeBeta``; returns the count."""
    swapped = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ == "SnakeBeta":
            setattr(module, name, FusedSnakeBeta(child).to(child.alpha.device))
            swapped += 1
        else:
            swapped += fuse_snake_activations(child)
    return swapped
