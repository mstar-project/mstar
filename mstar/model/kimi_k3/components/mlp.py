"""TP-aware SiTU-GLU MLP (the dense layer-0 MLP and the fused shared experts).

``gate_up_proj`` is a ``MergedColumnParallelLinear`` over ``[gate_proj | up_proj]`` sharded
on the intermediate dim, so each rank's output is ``[gate_local | up_local]`` and the
SiTU-GLU applies locally; ``down_proj`` all-reduces (or not, when the caller fuses the
reduction with another partial, see ``reduce_results``).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import MergedColumnParallelLinear, RowParallelLinear
from mstar.model.kimi_k3.components.common import SiTUAndMul


class ParallelSiTUMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        comm_group: CommGroup | None = None,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
        reduce_results: bool = True,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.gate_up_proj = MergedColumnParallelLinear(
            comm_group=comm_group, input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size], bias=False, gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            comm_group=comm_group, input_size=intermediate_size, output_size=hidden_size,
            bias=False, input_is_parallel=True, reduce_results=reduce_results,
        )
        self.act = SiTUAndMul(situ_beta, situ_linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_up_proj(x)))
