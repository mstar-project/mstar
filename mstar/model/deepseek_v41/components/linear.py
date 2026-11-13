"""DeepSeek-V4.1's linears as the reference's block-scaled GEMMs compute them.

The checkpoint stores most linears as E4M3 with a power-of-two scale per 32 x 32 block, and the
reference's GEMM for them reads its input rounded to FP8 in 32-wide groups, sums in fp32 and
returns the model's dtype. bf16 holds every dequantized weight exactly, so ``RoundedLinear``
keeps the weight in the model's dtype, rounds its input the same way (``mstar.utils.mx_round``)
and sums in fp32: the same values without an fp8 kernel. A fused W8A8 kernel replaces it where
speed matters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.utils.mx_round import round_fp8

COLUMN = "column"  # this rank's rows of the output
ROW = "row"        # this rank's columns of the input, the ranks' partial outputs summed


def fp32_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` summed in fp32, in ``x``'s dtype."""
    return F.linear(x.float(), weight.float()).to(x.dtype)


class RoundedLinear(nn.Module):
    """A bias-free linear whose input is rounded to FP8 first (``rounded``; off for the
    linears the checkpoint stores unquantized). ``shard`` splits it across the TP group."""

    def __init__(
        self, in_features: int, out_features: int, comm_group: CommGroup | None = None,
        shard: str | None = None, *, rounded: bool = True, block: int = 32,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.shard = shard if self.comm_group.world_size > 1 else None
        tp = self.comm_group.world_size if self.shard else 1
        self.in_features = divide(in_features, tp) if self.shard == ROW else in_features
        self.out_features = divide(out_features, tp) if self.shard == COLUMN else out_features
        self.rounded, self.block = rounded, block
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=dtype))

    def local(self, full: torch.Tensor) -> torch.Tensor:
        """This rank's part of the checkpoint's ``[out, in]`` weight."""
        rank = self.comm_group.rank
        if self.shard == COLUMN:
            return full.narrow(0, rank * self.out_features, self.out_features)
        if self.shard == ROW:
            return full.narrow(1, rank * self.in_features, self.in_features)
        return full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rounded:
            x = round_fp8(x, self.block)
        out = fp32_linear(x, self.weight)
        if self.shard == ROW:
            out = self.comm_group.all_reduce(out.float()).to(x.dtype)
        return out
