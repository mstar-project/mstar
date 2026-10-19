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
