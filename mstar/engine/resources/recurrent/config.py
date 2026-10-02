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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
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


class RecurrentGeometry(ABC):
    @abstractmethod
    def to_blocks(self, *args, **kwargs) -> dict[str, RecurrentBlockConfig]:
        pass

    @classmethod
    @abstractmethod
    def from_blocks(
        cls, blocks: dict[str, RecurrentBlockConfig],
    ) -> "RecurrentGeometry":
        pass


@dataclass(frozen=True)
class DeltaNetGeometry(RecurrentGeometry):
    """Head geometry of the delta-net family: gated delta rule (Qwen3.5,
    Qwen3-Next) and Kimi delta attention (Kimi Linear, GLM-5.3).

    Both carry the same two blocks: a K-last [HV, V, K] state matrix — the
    layout FlashInfer's pool paths want, and what lets one pool serve either —
    and a short conv window holding every tap but the current token's.

    ``to_blocks`` and ``from_blocks`` are inverses, so a backend reads its
    geometry off the pool it was pointed at rather than the model declaring it
    twice and the two drifting.
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int

    @property
    def conv_dim(self) -> int:
        """The depthwise conv runs over [q | k | v] concatenated."""
        return (
            2 * self.num_k_heads * self.head_k_dim
            + self.num_v_heads * self.head_v_dim
        )

    def to_blocks(
        self,
        state_dtype: torch.dtype = torch.float32,
        conv_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, RecurrentBlockConfig]:
        """Pool blocks for this geometry.

        Head counts are pre-sharding; ``shard_dims`` narrows them at build, as
        a ``KVConfig``'s head counts are.
        """
        return {
            "state": RecurrentBlockConfig(
                shape=(self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=state_dtype,
                shard_dims=(0,),
            ),
            "conv": RecurrentBlockConfig(
                shape=(self.conv_dim, self.conv_kernel_size - 1),
                dtype=conv_dtype,
                shard_dims=(0,),
            ),
        }

    @classmethod
    def from_blocks(
        cls, blocks: dict[str, RecurrentBlockConfig],
    ) -> "DeltaNetGeometry":
        """Recover head counts from block shapes. Works on sharded shapes,
        since every axis involved shards.

        Raises if the shapes are not a delta-net's — the check that a pool and
        the resource planning against it were built for the same model.
        """
        for name in ("state", "conv"):
            if name not in blocks:
                raise ValueError(
                    f"not a delta-net state pool: no {name!r} block, got "
                    f"{sorted(blocks)}"
                )
        state, conv = blocks["state"].shape, blocks["conv"].shape
        if len(state) != 3:
            raise ValueError(
                f"delta-net 'state' block must be [HV, V, K], got {state}"
            )
        if len(conv) != 2:
            raise ValueError(
                f"delta-net 'conv' block must be [conv_dim, width], got {conv}"
            )
        num_v_heads, head_v_dim, head_k_dim = state
        conv_dim, width = conv

        # conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        key_span = conv_dim - num_v_heads * head_v_dim
        if key_span <= 0 or key_span % (2 * head_k_dim):
            raise ValueError(
                f"conv block {conv} does not match state block {state}: "
                f"[q|k|v] over {num_v_heads}x{head_v_dim} values leaves "
                f"{key_span} for 2 x num_k_heads x {head_k_dim}"
            )
        return cls(
            num_k_heads=key_span // (2 * head_k_dim),
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=width + 1,
        )


@dataclass(frozen=True)
class Mamba2Geometry(RecurrentGeometry):
    """Head geometry of a Mamba-2 (SSD) layer: Nemotron-H, NVIDIA's Nemotron
    Nano/VoiceChat backbones, Mamba-Codestral.

    Two blocks, like the delta-net family: an ``[H, P, N]`` SSM state per head
    (fp32 by default: it is an exponentially decayed running sum over the whole
    session, and the reference keeps it in fp32) and the conv window over the
    ``[x | B | C]`` projection, every tap but the current token's.

    ``n_groups`` is the number of ``B``/``C`` groups shared by ``H`` heads. It
    is a shape fact (``conv_dim = H * P + 2 * n_groups * N``), so it is
    recoverable from the blocks and ``from_blocks`` inverts ``to_blocks``.
    """

    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel_size: int

    @property
    def d_inner(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def conv_dim(self) -> int:
        """The depthwise conv runs over ``[x | B | C]`` concatenated."""
        return self.d_inner + 2 * self.n_groups * self.state_size

    def to_blocks(
        self,
        state_dtype: torch.dtype = torch.float32,
        conv_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, RecurrentBlockConfig]:
        """Pool blocks for this geometry; head counts pre-sharding (heads and
        the conv channels both shard on their leading axis)."""
        return {
            "ssm": RecurrentBlockConfig(
                shape=(self.num_heads, self.head_dim, self.state_size),
                dtype=state_dtype,
                shard_dims=(0,),
            ),
            "conv": RecurrentBlockConfig(
                shape=(self.conv_dim, self.conv_kernel_size - 1),
                dtype=conv_dtype,
                shard_dims=(0,),
            ),
        }

    @classmethod
    def from_blocks(
        cls, blocks: dict[str, RecurrentBlockConfig],
    ) -> "Mamba2Geometry":
        """Recover the geometry from block shapes (sharded ones too, since
        every axis involved shards). Raises on shapes that are not a Mamba-2's,
        which is the check that a pool and the resource planning against it
        were built for the same model."""
        for name in ("ssm", "conv"):
            if name not in blocks:
                raise ValueError(
                    f"not a Mamba-2 state pool: no {name!r} block, got {sorted(blocks)}"
                )
        ssm, conv = blocks["ssm"].shape, blocks["conv"].shape
        if len(ssm) != 3:
            raise ValueError(f"Mamba-2 'ssm' block must be [H, P, N], got {ssm}")
        if len(conv) != 2:
            raise ValueError(f"Mamba-2 'conv' block must be [conv_dim, width], got {conv}")
        num_heads, head_dim, state_size = ssm
        conv_dim, width = conv
        bc_span = conv_dim - num_heads * head_dim
        if bc_span <= 0 or bc_span % (2 * state_size):
            raise ValueError(
                f"conv block {conv} does not match ssm block {ssm}: [x|B|C] over "
                f"{num_heads}x{head_dim} leaves {bc_span} for 2 x n_groups x {state_size}"
            )
        return cls(
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            n_groups=bc_span // (2 * state_size),
            conv_kernel_size=width + 1,
        )


@dataclass
class RecurrentStateConfig:
    # The total number of recurrent layers, not total transformer layers
    num_layers: int
    # Named blocks, e.g. what `DeltaNetGeometry.to_blocks` returns. A backend declares
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
    # writes there anyway; SM90 prefill has no say at all, since it gathers the
    # state in torch rather than addressing the pool, and masks the sentinel
    # itself (`GDNPrefillWrapper.run`). A sink is correct under all three, so
    # it is the default.
    #
    # TODO: derive this from (backend, dtype, ...) automatically.
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

    def apply_yaml_overrides(
        self, max_slots: int | None = None, state_dtype: str | None = None,
    ):
        """How many slots this deployment gets, and how precise they are.

        Block *shapes* are not tunable: they are the model's, and a pool sized
        for shapes the backend does not produce is a crash, not a slow run.
        The state's dtype is a deployment call — it trades precision that
        accumulates over a whole generation against half the bandwidth on a
        tensor read and written every step, and it decides which kernels the
        backend can reach. The model's default stands unless this is set.
        """
        if max_slots is not None:
            self.config.max_slots = max_slots
        if state_dtype is not None:
            try:
                dtype = getattr(torch, state_dtype)
            except AttributeError:
                dtype = None
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"state_dtype {state_dtype!r} is not a torch dtype"
                )
            block = self.config.blocks.get("state")
            if block is None:
                raise ValueError(
                    "state_dtype was set but this pool has no 'state' block; "
                    f"it has {sorted(self.config.blocks)}"
                )
            self.config.blocks["state"] = replace(block, dtype=dtype)


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
