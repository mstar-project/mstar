"""What a model declares about a fixed-size per-request state: its tensors,
its spec, its step.

Linear-attention / SSM families (KDA, GDN, Mamba) keep a fixed-size
recurrent state per request — a matrix memory and a short-conv tail per
layer — instead of a paged KV cache. This is the first resource on the pool
engine that is not paged: one slot per request, allocated at first admit,
freed at removal, and addressed by a per-row slot index the forward gathers
and scatters through. Kept free of the manager so a submodule can declare
the step without pulling anything in behind it.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


# Reserved slot every pool holds in addition to `max_slots`: padding rows of
# a captured replay and rows of rids that hold no slot address it, so a
# graph-shaped step never reads out of bounds and never touches a live
# request's state. Its contents are garbage by design and never committed.
SINK_SLOT = 0


@dataclass
class SlotTensorSpec:
    """One named per-slot tensor of the pool.

    ``shape`` is the per-request shape WITHOUT the slot axis; the pool inserts
    an axis of ``max_slots + 1`` at ``slot_dim`` (layer-major state keeps the
    layer axis outermost so a per-layer gather touches one contiguous plane).
    ``shard_dim`` names the axis of ``shape`` that is divided across the
    joint (tp * sp) world size — a head axis, once the model shards the
    layer — or None for state that is replicated on every rank.
    """
    shape: tuple[int, ...]
    dtype: torch.dtype
    slot_dim: int = 0
    shard_dim: int | None = None

    def __post_init__(self):
        self.shape = tuple(int(d) for d in self.shape)
        if not 0 <= self.slot_dim <= len(self.shape):
            raise ValueError(
                f"slot_dim={self.slot_dim} out of range for shape {self.shape}"
            )
        if self.shard_dim is not None and not 0 <= self.shard_dim < len(self.shape):
            raise ValueError(
                f"shard_dim={self.shard_dim} out of range for shape {self.shape}"
            )

    def pool_shape(self, max_slots: int) -> tuple[int, ...]:
        shape = list(self.shape)
        shape.insert(self.slot_dim, max_slots + 1)
        return tuple(shape)


@dataclass
class SlotStateConfig:
    tensors: dict[str, SlotTensorSpec]
    # real slots; the pool holds one more (SINK_SLOT)
    max_slots: int = 32

    def __post_init__(self):
        if self.max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {self.max_slots}")
        self._unsharded = {
            name: tuple(spec.shape) for name, spec in self.tensors.items()
        }

    def shard(self, num_shards: int) -> None:
        """Narrow every ``shard_dim`` axis to one rank's slice. Idempotent —
        computed from the declared shapes, so calling it twice is harmless."""
        from mstar.distributed.utils import divide

        for name, spec in self.tensors.items():
            if spec.shard_dim is None:
                continue
            full = list(self._unsharded[name])
            full[spec.shard_dim] = divide(full[spec.shard_dim], num_shards)
            spec.shape = tuple(full)


@dataclass
class SlotStateSpec(NodeResourceSpec):
    config: SlotStateConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.slot_state.manager import SlotStateManager

        return SlotStateManager

    def apply_yaml_overrides(self, max_slots: int | None = None):
        """How many requests may hold state at once — the deployment's
        concurrency cap for this node, sized against the box."""
        if max_slots is not None:
            self.config.max_slots = int(max_slots)


@dataclass(frozen=True)
class SlotStateStep(ResourceStep):
    """``mode`` says how the forward walks the state:

    - ``"step"``: one new token per row; the forward gathers every row's
      slot by the planned ``slot_index``, runs the recurrence once, scatters
      back — graph-shaped, no host work.
    - ``"chunk"``: a span of tokens per row; the forward loops the planned
      ``spans`` on the host against in-place slot views (prefill, chunked
      continue).
    """
    mode: str = "step"
    # per-request state a step reads but must not commit (speculative draft)
    commit: bool = True

    def __post_init__(self):
        if self.mode not in ("step", "chunk"):
            raise ValueError(f"SlotStateStep.mode must be 'step' or 'chunk', got {self.mode!r}")


@dataclass
class SlotSpan:
    """One row of a planned step: where its tokens sit in the packed batch
    and which slot holds its state. ``ctx_start`` is the request's committed
    length before this step."""
    request_id: str
    slot: int
    q_start: int
    q_len: int
    ctx_start: int
    # False for a padding row / a rid that holds no slot: reads the sink,
    # never committed
    real: bool = True


@dataclass
class SlotStatePlan:
    mode: str
    # device int64 [rows], packed-batch order; padding rows -> SINK_SLOT
    slot_index: torch.Tensor
    spans: list[SlotSpan] = field(default_factory=list)

    @property
    def request_ids(self) -> list[str]:
        return [span.request_id for span in self.spans]
