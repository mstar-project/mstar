"""Non-distributed fused linear projections.

``FusedColumnLinear`` concatenates several output shards into a single
weight (and optional bias) so a transformer's QKV or gate/up projections
run as one GEMM instead of several. Each shard is loaded independently
via a ``weight_loader`` so HF checkpoints that keep the shards as
separate tensors (``q``/``k``/``v``, ``gate``/``up``) load straight into
the fused parameter.

This mirrors the role of the TP-aware ``QKVParallelLinear`` /
``MergedColumnParallelLinear`` in ``model.components.distributed`` but
without any tensor-parallel sharding — use it for models that fuse
projections but don't need TP.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FusedColumnLinear(nn.Module):
    """Linear whose output is the concatenation of several shards along
    dim 0, fused into a single weight (and optional bias).

    Args:
        input_size: input feature dim.
        shard_sizes: maps a shard id to its output size. Shards are laid
            out along dim 0 in iteration order, so the order of this dict
            defines the layout (and the split used at forward time).
        bias: whether to include a (fused) bias.

    The fused ``weight`` / ``bias`` carry a ``weight_loader(param, tensor,
    shard_id)`` method that copies one checkpoint shard into its slice of
    the fused parameter, dispatched by ``shard_id`` (a key of
    ``shard_sizes``). Wire the per-shard checkpoint keys to it with the
    loader's stacked-param rules.
    """

    def __init__(
        self,
        input_size: int,
        shard_sizes: dict[str | int, int],
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._shard_sizes = dict(shard_sizes)
        self._shard_offsets: dict[str | int, int] = {}
        offset = 0
        for shard_id, size in self._shard_sizes.items():
            self._shard_offsets[shard_id] = offset
            offset += size
        self.output_size = offset
        self.weight = nn.Parameter(torch.empty(offset, input_size, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(offset, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # ``to_empty`` / ``.to(...)`` reallocate the Parameters and drop the
        # attached ``weight_loader``; re-attach to the new objects.
        self._attach_weight_loaders()
        return result

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | int | None = None,
    ) -> None:
        assert loaded_shard_id in self._shard_sizes, (
            f"FusedColumnLinear got unknown shard id {loaded_shard_id!r}; "
            f"expected one of {list(self._shard_sizes)}"
        )
        offset = self._shard_offsets[loaded_shard_id]
        size = self._shard_sizes[loaded_shard_id]
        dst = param.data.narrow(0, offset, size)
        assert dst.shape == loaded_weight.shape, (
            f"shard {loaded_shard_id!r} shape mismatch: dst {tuple(dst.shape)} "
            f"vs weight {tuple(loaded_weight.shape)}"
        )
        dst.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
