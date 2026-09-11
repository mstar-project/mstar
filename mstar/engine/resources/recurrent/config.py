"""What a model declares about a pool of recurrent state.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling a backend in behind it.

The pool is deliberately ignorant of what the state means. A slot is a fixed
number of bytes per layer, held for as long as a request needs it; whether
those bytes are a delta-net [HV, V, K] matrix, a Mamba SSM block, or something
else is the calling resource's business. Contrast the KV cache, whose geometry
(pages, tokens, heads) is baked into its own contract.

The consequence that shapes everything here: this state does not grow with the
sequence. Capacity is a slot count, not a byte budget that scales with length,
and a fork is a fixed-size copy rather than a page-count-dependent one.
"""

from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class RecurrentBlockConfig:
    """One per-slot, per-layer tensor block.

    ``shape`` is opaque to the pool. ``shard_dims`` names the axes divided
    across ranks — shape arithmetic, not semantics: the pool never learns that
    axis 0 happens to be a head count.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    shard_dims: tuple[int, ...] = ()

    def __post_init__(self):
        self.shape = tuple(self.shape)
        self._unsharded_shape = self.shape

    def shard(self, num_shards: int) -> None:
        from mstar.distributed.utils import divide

        shape = list(self._unsharded_shape)
        for dim in self.shard_dims:
            shape[dim] = divide(self._unsharded_shape[dim], num_shards)
        self.shape = tuple(shape)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()


def delta_net_conv_dim(
    num_k_heads: int, head_k_dim: int, num_v_heads: int, head_v_dim: int,
) -> int:
    """The depthwise conv runs over [q | k | v] concatenated."""
    return 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim


def delta_net_blocks(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    state_dtype: torch.dtype = torch.float32,
    conv_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, RecurrentBlockConfig]:
    """Blocks for the delta-net family: gated delta rule (Qwen3.5, Qwen3-Next)
    and Kimi delta attention (Kimi Linear, GLM-5.3).

    Both carry the same two: a K-last [HV, V, K] state matrix — the layout
    FlashInfer's pool paths want, and what lets one pool serve either — and a
    short conv window holding every tap but the current token's.

    Head counts are pre-sharding; ``shard_dims`` narrows them at build, as a
    ``KVConfig``'s head counts are.
    """
    conv_dim = delta_net_conv_dim(
        num_k_heads, head_k_dim, num_v_heads, head_v_dim
    )
    return {
        "state": RecurrentBlockConfig(
            shape=(num_v_heads, head_v_dim, head_k_dim),
            dtype=state_dtype,
            shard_dims=(0,),
        ),
        "conv": RecurrentBlockConfig(
            shape=(conv_dim, conv_kernel_size - 1),
            dtype=conv_dtype,
            shard_dims=(0,),
        ),
    }


@dataclass
class RecurrentStateConfig:
    # The total number of recurrent layers, not total transformer layers
    num_layers: int
    # Named blocks, e.g. what `delta_net_blocks` returns. A backend declares
    # what it needs; the pool allocates one tensor per block and hands back
    # per-layer views.
    blocks: dict[str, RecurrentBlockConfig] = field(default_factory=dict)

    # Slots the pool can hand out at once. A request holds one per label, so
    # this bounds concurrent requests times their labels, not requests alone.
    # The sink, when there is one, comes out of this the way SINK_PAGE comes
    # out of a KV cache's `max_num_pages`.
    max_slots: int = 256

    # Whether padding rows address a real sink slot or a negative sentinel.
    #
    # Not the model author's call: it turns on what the backend's kernels do
    # with an unaddressed row, and they disagree. FlashInfer's fp32 GDN decode
    # skips a -1 row entirely; its bf16 fast path redirects -1 onto slot 0 and
    # writes there anyway. A sink is correct under both, so it is the default
    # and the sentinel is opt-in.
    #
    # TODO: derive this from (backend, dtype, ...) once there is more than one
    # backend to ask, and drop the knob.
    disable_sink_slot: bool = False

    def __post_init__(self):
        if not self.blocks:
            raise ValueError("a recurrent state pool must declare a block")
        if not self.disable_sink_slot and self.max_slots < 2:
            raise ValueError(
                f"max_slots={self.max_slots} leaves nothing to hand out: the "
                "sink takes one. Raise it or set disable_sink_slot."
            )

    @property
    def usable_slots(self) -> int:
        """Slots requests can hold; the sink is not one of them."""
        return self.max_slots - (0 if self.disable_sink_slot else 1)

    def shard(self, num_shards: int) -> None:
        """Narrow every block's sharded axes; see ``KVConfig.shard``.

        Idempotent, so one config shared by the pool and the resource planning
        against it can be sharded by both on construction.
        """
        for block in self.blocks.values():
            block.shard(num_shards)

    @property
    def slot_bytes(self) -> int:
        return self.num_layers * sum(b.nbytes for b in self.blocks.values())

    @property
    def total_bytes(self) -> int:
        return self.slot_bytes * self.max_slots


@dataclass
class RecurrentStateSpec(NodeResourceSpec):
    config: RecurrentStateConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.recurrent.pool import RecurrentStatePool

        return RecurrentStatePool

    def apply_yaml_overrides(self, max_slots: int | None = None):
        """How many slots this deployment gets.

        Block shapes are not tunable: they are the model's, and a pool sized
        for shapes the backend does not produce is a crash, not a slow run.
        """
        if max_slots is not None:
            self.config.max_slots = max_slots


@dataclass(frozen=True)
class RecurrentStep(ResourceStep):
    """One step's work against the pool.

    There is no ``commit`` flag, unlike ``KVStep``. A backend writes the pool
    in place, so by the time commit ran the bytes would already be gone.
    A consumer that needs it would have to have two labels: reading one label
    and writing an other.

    Forks mirror ``KVStep``'s: ``(from_label, to_label)`` pairs, reserved at
    admit and copied at plan (pre) or commit (post).
    """

    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()
