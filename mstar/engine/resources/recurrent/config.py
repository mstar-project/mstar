"""What a model declares about constant-size per-request recurrent state: its parts,
its spec, its step.

A linear-attention / SSM layer keeps a fixed-size state per request per layer (for
Kimi Delta Attention: a ``[H, 128, 128]`` fp32 recurrent matrix plus a ``[3*H*128, W]``
short-conv window). Unlike a paged KV cache the state does not grow with the sequence,
so the resource hands out *slots*: one row per request per layer in a preallocated
``[num_layers, max_num_slots + 1, *shape]`` tensor per part. Slot 0 is scratch: padding
rows in a captured replay point at it and rows without a resident state gather zeros
instead of reading it.

Kept free of the manager so a submodule can declare a step without importing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class StatePart:
    """One tensor of the per-request state.

    ``shape`` is the *unsharded* per-request, per-layer shape. ``shard_dim`` names the
    dimension a tensor-parallel deployment divides by the instance world size (the head
    dimension for KDA: heads are split across TP ranks), or ``None`` for replicated.
    """
    shape: tuple[int, ...]
    dtype: torch.dtype
    shard_dim: int | None = None


@dataclass
class RecurrentStateConfig:
    num_layers: int
    parts: dict[str, StatePart]
    max_num_slots: int = 256
    # pinned host slots kept for offload; 0 disables it
    cpu_offload_slots: int = 0

    def __post_init__(self):
        self._unsharded_parts = {k: StatePart(p.shape, p.dtype, p.shard_dim) for k, p in self.parts.items()}

    def shard(self, num_shards: int) -> None:
        """Narrow every sharded part to one rank's slice. Idempotent."""
        from mstar.distributed.utils import divide

        for name, full in self._unsharded_parts.items():
            if full.shard_dim is None:
                continue
            shape = list(full.shape)
            shape[full.shard_dim] = divide(shape[full.shard_dim], num_shards)
            self.parts[name] = StatePart(tuple(shape), full.dtype, full.shard_dim)

    def slot_nbytes(self) -> int:
        """Bytes one request's state takes across all layers (per rank)."""
        total = 0
        for p in self.parts.values():
            n = 1
            for s in p.shape:
                n *= s
            total += n * torch.empty(0, dtype=p.dtype).element_size()
        return total * self.num_layers


class CommitMode(Enum):
    """How a step's writes become the request's resident state.

    * ``IN_PLACE``: the kernels update the slot directly during the forward (the
      default for prefill and plain decode).
    * ``CHECKPOINT``: as ``IN_PLACE``, and the prefill additionally snapshots the state
      at a chunk-aligned offset into a checkpoint slot (prefix caching; FlashKDA
      ``checkpoint_state``).
    * ``DEFERRED``: the forward must not write the resident slot; ``commit`` receives
      the accepted length afterwards and replays only the accepted prefix (speculative
      decoding, RecoverSSM style).
    """
    IN_PLACE = "in_place"
    CHECKPOINT = "checkpoint"
    DEFERRED = "deferred"


@dataclass
class RecurrentStateSpec(NodeResourceSpec):
    config: RecurrentStateConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.recurrent.manager import RecurrentStateManager

        return RecurrentStateManager

    def apply_yaml_overrides(
        self, max_num_slots: int | None = None, cpu_offload_slots: int | None = None,
    ):
        """How many requests can hold state at once, and how many can be parked on the host."""
        if max_num_slots is not None:
            self.config.max_num_slots = max_num_slots
        if cpu_offload_slots is not None:
            self.config.cpu_offload_slots = cpu_offload_slots


@dataclass(frozen=True)
class RecurrentStateStep(ResourceStep):
    commit_mode: CommitMode = CommitMode.IN_PLACE


@dataclass
class RecurrentPlanOutput:
    """What a layer reads to address the state for one step.

    ``slot_ids[i]`` / ``has_state[i]`` describe the i-th segment (request row) in
    declaration order; under a CUDA-graph lease both live in static buffers padded to
    the capture batch size, so the captured kernels read stable addresses.
    """
    slot_ids: torch.Tensor  # int32 [rows] on device
    has_state: torch.Tensor  # bool [rows] on device
    cu_seqlens: torch.Tensor  # int32 [rows + 1] on device: token boundaries of the rows
    slot_ids_cpu: list[int] = field(default_factory=list)
    has_state_cpu: list[bool] = field(default_factory=list)
    cu_seqlens_cpu: list[int] = field(default_factory=list)
    is_decode: bool = False  # every row appends exactly one token
    num_rows: int = 0  # real (unpadded) rows
    num_tokens: int = 0
