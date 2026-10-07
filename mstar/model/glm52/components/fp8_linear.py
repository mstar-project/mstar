"""GLM-5.2 linears that keep the checkpoint's fp8 weights (``dense_fp8``).

Each holds this rank's shard of an e4m3 weight (uint8 container, like the routed experts,
so ``module.to(bf16)`` leaves it alone) and its fp32 ``weight_scale_inv`` blocks. Column
parallel shards the output rows, row parallel the input columns; the scales shard with
them, so every per-rank size must be a whole number of blocks.
"""
from __future__ import annotations

from functools import partial

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.quantization import FP8_DTYPE, dequantize_fp8_block_weight

REPLICATED, COLUMN, ROW = None, 0, 1


def dense_fp8_block(config: Glm52ModelConfig) -> tuple[int, int] | None:
    """The scale block when the non-expert linears stay fp8, else None."""
    if not config.dense_fp8:
        return None
    if config.quantization_config is None:
        raise ValueError(
            "dense_fp8 needs the fp8 checkpoint's quantization_config (block scales)")
    assert config.hidden_act == "silu", "the fp8 gate/up kernel applies SwiGLU"
    return tuple(config.quantization_config.weight_block_size)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def reference(x, weight, scale, block_size, glu=False):
    """The flag-off numerics: the weight dequantized to bf16 as the loader does, then the
    plain linear in x.dtype."""
    w = dequantize_fp8_block_weight(weight, scale, block_size=block_size).to(x.dtype)
    y = torch.nn.functional.linear(x, w)
    if glu:
        g, u = y.chunk(2, dim=-1)
        y = torch.nn.functional.silu(g) * u
    return y


def fp8_block_linear(x, weight, scale, block_size, glu=False):
    """``x @ (weight * scale).T`` (its SwiGLU with ``glu``): the dense_fp8 kernels on CUDA,
    ``reference`` on host tensors (tests, no triton needed)."""
    if not x.is_cuda:
        return reference(x, weight, scale, block_size, glu)
    from mstar.model.glm52 import dense_fp8

    return dense_fp8.linear(x, weight, scale, block_size, glu)


class Fp8Linear(nn.Module):
    """``y = x @ (weight * weight_scale_inv).T`` over one rank's shard.

    ``shard``: REPLICATED, COLUMN (output rows) or ROW (input columns). ``merged`` > 1 stacks
    that many column shards (gate and up), each loaded by its shard id.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int],
        comm_group: CommGroup | None = None,
        shard: int | None = REPLICATED,
        merged: int = 1,
        reduce_results: bool = False,
    ) -> None:
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.comm_group = comm_group
        self.tp_rank, self.tp_size = comm_group.rank, comm_group.world_size
        self.block_size = tuple(block_size)
        self.shard, self.merged = shard, merged
        self.reduce_results = reduce_results
        bo, bi = self.block_size
        tp = self.tp_size
        self.out_features = divide(out_features, tp) if shard == COLUMN else out_features
        self.in_features = divide(in_features, tp) if shard == ROW else in_features
        if shard == COLUMN and tp > 1:
            assert (self.out_features // merged) % bo == 0, (
                f"per-rank rows {self.out_features // merged} split a {bo}-row scale block")
        if shard == ROW and tp > 1:
            assert self.in_features % bi == 0, (
                f"per-rank columns {self.in_features} split a {bi}-column scale block")
        if merged > 1:
            assert (self.out_features // merged) % bo == 0
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=torch.uint8),
            requires_grad=False)
        self.weight_scale_inv = nn.Parameter(
            torch.empty(_ceil_div(self.out_features, bo), _ceil_div(self.in_features, bi),
                        dtype=torch.float32),
            requires_grad=False)
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        bo, bi = self.block_size
        self.weight.weight_loader = partial(self._load, 1, 1)
        self.weight_scale_inv.weight_loader = partial(self._load, bo, bi)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def _load(self, row_unit: int, col_unit: int, param: nn.Parameter,
              loaded: torch.Tensor, shard_id: int | None = None) -> None:
        """Copy this rank's slice of a full checkpoint tensor (fp8 weight or its scales)."""
        if param is self.weight:
            if loaded.dtype not in (FP8_DTYPE, torch.uint8):
                raise TypeError(
                    f"dense_fp8 expects an fp8 checkpoint weight, got {loaded.dtype}")
            loaded = loaded.view(torch.uint8)
        dst = param.data
        if self.merged > 1:
            assert shard_id is not None, "merged fp8 linear needs a shard id"
            rows = dst.shape[0] // self.merged
            dst = dst.narrow(0, int(shard_id) * rows, rows)
        else:
            assert shard_id is None
        if self.shard == COLUMN:
            rows = dst.shape[0]
            loaded = loaded.narrow(0, self.tp_rank * rows, rows)
        elif self.shard == ROW:
            cols = dst.shape[1]
            loaded = loaded.narrow(1, self.tp_rank * cols, cols)
        assert dst.shape == loaded.shape, (
            f"fp8 linear shard {tuple(loaded.shape)} does not fit {tuple(dst.shape)}")
        dst.copy_(loaded)

    def forward(self, x: torch.Tensor, glu: bool = False) -> torch.Tensor:
        y = fp8_block_linear(x, self.weight, self.weight_scale_inv, self.block_size, glu)
        if self.reduce_results and self.tp_size > 1:
            y = self.comm_group.all_reduce(y)
        return y

    def process_weights_after_loading(self, device) -> None:
        if torch.device(device).type == "cuda":
            from mstar.model.glm52 import dense_fp8  # registers the op before any compile

            dense_fp8.reserve(device)

    def release(self) -> None:
        """Drop the storage (the absorbed MLA fuses its sources); keep the Parameters."""
        for p in (self.weight, self.weight_scale_inv):
            p.data = p.data.new_empty(0)


class Fp8ParallelGatedMLP(nn.Module):
    """``ParallelGatedMLP`` over fp8 weights: gate/up stacked column-parallel with SwiGLU in
    the same launch, down row-parallel."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        block_size: tuple[int, int],
        comm_group: CommGroup | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = Fp8Linear(
            hidden_size, 2 * intermediate_size, block_size, comm_group, shard=COLUMN, merged=2)
        self.down_proj = Fp8Linear(
            intermediate_size, hidden_size, block_size, comm_group, shard=ROW,
            reduce_results=reduce_results)
        self.intermediate_size_per_partition = self.gate_up_proj.out_features // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_up_proj(x, glu=True))
