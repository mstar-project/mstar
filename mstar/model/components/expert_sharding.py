"""How a MoE layer's experts are split over a communication group.

Two axes, one collective. ``ep_size`` groups of ranks each own a disjoint, contiguous subset of
the experts (expert parallelism); inside a group the ``tp_size = world_size / ep_size`` ranks
shard every owned expert along its intermediate dimension (tensor parallelism, the layout
``ParallelSparseMoeBlock`` and ``KimiLatentMoE`` have always used). A rank's result is the
partial sum of the token-expert products it holds, whatever the split, so the layer combines
partials with the same all-reduce in every configuration: ``ep_size=1`` is plain TP,
``ep_size=world_size`` is pure EP, anything in between is the hybrid.

The object is pure bookkeeping (shapes, ownership, checkpoint routing, the local expert
map); it owns no tensors and issues no collectives. Kernels see the *local* expert id of every
token-expert assignment, or ``invalid_id`` (``= local_experts``) for assignments that belong to
another expert-parallel group; the alignment step (``moe_align_block_size``) drops those.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertSharding:
    num_experts: int  # global
    intermediate_size: int  # global (per expert)
    world_size: int
    rank: int
    ep_size: int = 1

    def __post_init__(self):
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside a group of {self.world_size}")
        if self.ep_size < 1 or self.world_size % self.ep_size:
            raise ValueError(f"ep_size {self.ep_size} must divide the group size {self.world_size}")
        if self.num_experts % self.ep_size:
            raise ValueError(f"{self.num_experts} experts cannot be split over {self.ep_size} expert-parallel groups")
        if self.intermediate_size % self.tp_size:
            raise ValueError(f"intermediate size {self.intermediate_size} not divisible by tp_size {self.tp_size}")

    @classmethod
    def from_group(cls, comm_group, num_experts: int, intermediate_size: int, ep_size: int = 1) -> "ExpertSharding":
        """For one rank of ``comm_group`` (``None`` = a single rank)."""
        if comm_group is None:
            return cls(num_experts, intermediate_size, 1, 0, 1)
        return cls(num_experts, intermediate_size, comm_group.world_size, comm_group.rank, ep_size)

    # ------------------------------------------------------------------ geometry
    @property
    def tp_size(self) -> int:
        return self.world_size // self.ep_size

    @property
    def ep_rank(self) -> int:
        """Expert-parallel groups are contiguous rank blocks: ranks ``[ep_rank * tp_size, (ep_rank + 1) * tp_size)``."""
        return self.rank // self.tp_size

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def expert_offset(self) -> int:
        """First global expert id this rank holds."""
        return self.ep_rank * self.local_experts

    @property
    def inter_local(self) -> int:
        return self.intermediate_size // self.tp_size

    @property
    def inter_offset(self) -> int:
        """First intermediate channel this rank holds of every owned expert."""
        return self.tp_rank * self.inter_local

    @property
    def invalid_id(self) -> int:
        """Local expert id standing for 'held by another expert-parallel group'. It is
        ``local_experts``, i.e. one past the last local expert, which the alignment kernels
        skip (they ignore ids ``>= num_experts``)."""
        return self.local_experts

    @property
    def is_partial(self) -> bool:
        """Whether this rank skips some token-expert assignments (expert parallelism on)."""
        return self.ep_size > 1

    def owns(self, expert: int) -> bool:
        return self.expert_offset <= expert < self.expert_offset + self.local_experts

    def local_expert(self, expert: int) -> int:
        if not self.owns(expert):
            raise ValueError(f"expert {expert} is not held by rank {self.rank} (ep rank {self.ep_rank})")
        return expert - self.expert_offset

    # ------------------------------------------------------------------ routing
    def expert_map(self, device: torch.device | str) -> torch.Tensor:
        """``[num_experts]`` int32: global expert id -> local id, ``invalid_id`` elsewhere."""
        m = torch.full((self.num_experts,), self.invalid_id, dtype=torch.int32)
        m[self.expert_offset:self.expert_offset + self.local_experts] = torch.arange(self.local_experts, dtype=torch.int32)
        return m.to(device)

    def localize(self, topk_idx: torch.Tensor) -> torch.Tensor:
        """Global top-k expert ids ``[T, k]`` -> local ids with ``invalid_id`` for assignments
        of other expert-parallel groups. One gather (the map is cached per device); the
        identity, without a launch, when every rank holds every expert."""
        if not self.is_partial:
            return topk_idx
        key = (str(topk_idx.device),)
        cache = _EXPERT_MAPS.setdefault(self, {})
        m = cache.get(key)
        if m is None:
            m = cache[key] = self.expert_map(topk_idx.device)
        return m[topk_idx]

    # ------------------------------------------------------------------ checkpoint routing
    def load_gate_up(self, param: torch.Tensor, loaded: torch.Tensor, kind: str, expert: int) -> None:
        """Route one expert's ``gate`` (w1) or ``up`` (w3) tensor ``[inter, cols]`` into the fused
        ``[local_experts, 2 * inter_local, cols]`` parameter (gate rows first). The row dim is the
        intermediate dim in every storage format (bf16 weights, packed nibbles, group scales)."""
        if not self.owns(expert):
            return
        assert kind in ("gate", "up"), kind
        src = loaded.narrow(0, self.inter_offset, self.inter_local)
        row0 = 0 if kind == "gate" else self.inter_local
        dst = param.data[self.local_expert(expert)].narrow(0, row0, self.inter_local)
        assert dst.shape == src.shape, (tuple(dst.shape), tuple(src.shape))
        dst.copy_(src)

    def load_down(self, param: torch.Tensor, loaded: torch.Tensor, expert: int, per_col: int = 1) -> None:
        """Route one expert's ``down`` (w2) tensor ``[rows, inter / per_col]`` into the fused
        ``[local_experts, rows, inter_local / per_col]`` parameter; ``per_col`` is the number of
        intermediate channels per stored column (1 for weights, 2 for packed nibbles, 32 for
        MXFP4 group scales)."""
        if not self.owns(expert):
            return
        cols = self.inter_local // per_col
        src = loaded.narrow(1, self.inter_offset // per_col, cols)
        dst = param.data[self.local_expert(expert)]
        assert dst.shape == src.shape, (tuple(dst.shape), tuple(src.shape))
        dst.copy_(src)

    def describe(self) -> str:
        return (f"{self.num_experts} experts x {self.intermediate_size} over {self.world_size} ranks: "
                f"ep {self.ep_size} x tp {self.tp_size}; this rank holds experts "
                f"[{self.expert_offset}, {self.expert_offset + self.local_experts}) at {self.inter_local} channels")


# per-sharding, per-device expert maps (tiny int32 tensors); keyed by the frozen dataclass
_EXPERT_MAPS: dict[ExpertSharding, dict[tuple, torch.Tensor]] = {}


__all__ = ["ExpertSharding"]
