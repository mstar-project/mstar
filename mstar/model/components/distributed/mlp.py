"""TP-aware SwiGLU MLPs (parallel counterparts of
``mstar.model.components.GatedMLP``).

``ParallelGatedMLP`` fuses the gate/up projections into a single
``MergedColumnParallelLinear`` (sharded along the intermediate dim) with a
``RowParallelLinear`` down projection that all-reduces the partial sums.
The checkpoint stores ``gate_proj.weight`` and ``up_proj.weight``
separately; the model's weight loader calls
``self.gate_up_proj.weight.weight_loader(loaded_weight,
loaded_shard_id=0)`` for gate and ``loaded_shard_id=1`` for up.

``ParallelGatedMLPUnfused`` keeps gate/up as separate
``ColumnParallelLinear`` projections so ``state_dict()`` keys match a
checkpoint's one-to-one — for loaders that stream weights by name with no
stacked-parameter rules.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.mlp import _resolve_activation

# Public shard IDs for gate / up projections.
GATE_SHARD_ID = 0
UP_SHARD_ID = 1


class ParallelGatedMLP(nn.Module):
    """SwiGLU-style gated MLP partitioned across TP ranks.

    Args:
        comm_group: TP comm group for this MLP's parallel linears.
        hidden_size: model hidden dim (full, not per-partition).
        intermediate_size: SwiGLU intermediate dim (full).
        activation: HF activation name (``silu``, ``gelu``, ``gelu_tanh``).
        bias: whether the linears have a bias term.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        comm_group: CommGroup | None = None,
        activation: str | Callable = "silu",
        bias: bool = False,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act = _resolve_activation(activation)

        self.gate_up_proj = MergedColumnParallelLinear(
            comm_group=comm_group,
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=bias,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            comm_group=comm_group,
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            reduce_results=True,
        )

        self.intermediate_size_per_partition = (
            intermediate_size // comm_group.world_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.intermediate_size_per_partition, dim=-1)
        return self.down_proj(self.act(gate) * up)


class ParallelGatedMLPUnfused(nn.Module):
    """SwiGLU MLP with separate (unfused) gate/up column shards.

    Same math and sharding as ``ParallelGatedMLP``, with ``gate_proj`` and
    ``up_proj`` kept as separate ``ColumnParallelLinear`` projections so the
    parameter names match a checkpoint's ``gate_proj.weight`` /
    ``up_proj.weight`` one-to-one (plain name-matched loading, no
    stacked-parameter loader rules). A trivial comm group (world size 1)
    makes the projections plain linears.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        comm_group: CommGroup | None = None,
        activation: str | Callable = "silu",
        bias: bool = False,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.act = _resolve_activation(activation)
        self.gate_proj = ColumnParallelLinear(comm_group, hidden_size, intermediate_size, bias=bias)
        self.up_proj = ColumnParallelLinear(comm_group, hidden_size, intermediate_size, bias=bias)
        self.down_proj = RowParallelLinear(comm_group, intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
