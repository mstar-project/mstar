"""One GEMM for several projections of the same input.

Decode steps are weight-bandwidth bound, and a projection under ~25 MB streams its weights
far below the memory roofline because each launch pays a fixed ramp (a 2 MB projection takes
as long as a 20 MB one on an H100). Layers that apply several linears to one activation
(attention gates, low-rank factors, a router next to a down-projection) get most of that back
by concatenating the weights along the output dim and running one GEMM.

``MergedParallelLinear`` holds such a concatenation for a tensor-parallel comm group, where each
segment is either **column-parallel** (each rank holds ``size / world`` of its rows, like
``ColumnParallelLinear``) or **replicated** (every rank holds all of its rows). The weight rows
are ``[segment 0 local | segment 1 local | ...]``; the checkpoint's separate tensors load through
``weight_loader(param, tensor, segment_name)`` (a stacked-shard rule per segment), and
``split(out)`` returns the per-segment views of the GEMM output.

Views of a multi-row output are column slices and therefore not contiguous; callers that hand a
segment to a kernel needing contiguous memory call ``.contiguous()`` on it (a no-op for one
row, one small copy otherwise), which is still far cheaper than the GEMM launches saved.
Segment offsets and the total width are padded to multiples of ``ALIGN`` elements (16 bytes in
bf16), so every view is aligned like a tensor of its own: cuBLAS otherwise falls back to slow
"align2" kernels for a strided GEMM input (measured 18 µs instead of 2.4 for a 0.4 MB
projection). The padding rows of the weight are never loaded or read.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide

COLUMN = "column"
REPLICATED = "replicated"
ALIGN = 8  # elements; segment offsets and the row width are multiples of it


def _aligned(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


@dataclass(frozen=True)
class Segment:
    name: str
    output_size: int  # full (unsharded) size
    sharding: str = COLUMN

    def local_size(self, world_size: int) -> int:
        if self.sharding == REPLICATED:
            return self.output_size
        if self.sharding == COLUMN:
            return divide(self.output_size, world_size)
        raise ValueError(f"segment {self.name!r}: unknown sharding {self.sharding!r}")


class MergedParallelLinear(nn.Module):
    def __init__(
        self,
        comm_group: CommGroup | None,
        input_size: int,
        segments: list[Segment | tuple],
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if bias:
            raise NotImplementedError("MergedParallelLinear: bias")
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.comm_group = comm_group
        self.tp_rank, self.tp_size = comm_group.rank, comm_group.world_size
        self.segments = tuple(s if isinstance(s, Segment) else Segment(*s) for s in segments)
        names = [s.name for s in self.segments]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate segment names {names}")
        self.input_size = input_size
        self.local_sizes = {s.name: s.local_size(self.tp_size) for s in self.segments}
        self.offsets = {}
        off = 0
        for s in self.segments:
            self.offsets[s.name] = off
            off += _aligned(self.local_sizes[s.name])
        self.output_size_local = off  # includes the alignment padding
        self.weight = nn.Parameter(torch.empty(off, input_size, dtype=dtype))
        self.register_parameter("bias", None)
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        self.weight.weight_loader = self.weight_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def segment(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(name)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None):
        """Copy this rank's rows of one segment's full checkpoint tensor ``[output_size, input]``
        into the merged parameter; ``loaded_shard_id`` is the segment name."""
        if loaded_shard_id is None:
            raise ValueError("MergedParallelLinear.weight_loader needs the segment name as loaded_shard_id")
        seg = self.segment(loaded_shard_id)
        n = self.local_sizes[seg.name]
        if seg.sharding == COLUMN:
            src = loaded_weight.narrow(0, self.tp_rank * n, n)
        else:
            src = loaded_weight
        dst = param.data.narrow(0, self.offsets[seg.name], n)
        assert dst.shape == src.shape, (seg.name, tuple(dst.shape), tuple(src.shape))
        dst.copy_(src)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., input] -> [..., output_size_local]``: every segment's local columns in one GEMM
        (plus the alignment padding columns, which hold no meaningful values)."""
        return torch.nn.functional.linear(x, self.weight)

    def split(self, out: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-segment views ``[..., local_size]`` of a :meth:`forward` result."""
        return {s.name: out.narrow(-1, self.offsets[s.name], self.local_sizes[s.name]) for s in self.segments}

    def project(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.split(self.forward(x))

    def extra_repr(self) -> str:
        segs = ", ".join(f"{s.name}={self.local_sizes[s.name]}{'r' if s.sharding == REPLICATED else ''}" for s in self.segments)
        return f"in={self.input_size}, out_local={self.output_size_local} [{segs}], tp={self.tp_size}"


__all__ = ["ALIGN", "COLUMN", "REPLICATED", "MergedParallelLinear", "Segment"]
