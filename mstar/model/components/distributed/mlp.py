"""TP-aware SwiGLU MLP.

Mirrors ``mstar.model.components.GatedMLP`` but with the gate/up
projections fused into a single ``MergedColumnParallelLinear`` (sharded
along the intermediate dim) and the down projection as a
``RowParallelLinear`` that all-reduces the partial sums.

The checkpoint stores ``gate_proj.weight`` and ``up_proj.weight``
separately; the model's weight loader calls
``self.gate_up_proj.weight.weight_loader(loaded_weight,
loaded_shard_id=0)`` for gate and ``loaded_shard_id=1`` for up.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.model.components.distributed.linear import (
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
        comm_group: TPCommGroup | None = None,
        activation: str | Callable = "silu",
        bias: bool = False,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
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
