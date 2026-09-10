"""What a model declares about a KV cache: its shape, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class KVLayout(Enum):
    NHD = "NHD"
    # TODO: can add more, like HND, MLA


@dataclass(kw_only=True)
class KVConfig(ABC):
    """The model geometry every KV storage strategy needs, and nothing else.

    What a cache *holds* (layers, heads, head dim) is a checkpoint fact and
    lives here; how it is *stored* — paged or ringed, is storage policy and
    lives in a subclass. ``KVSpec`` dispatches on which one it was handed.

    ``kw_only`` is load-bearing, not style: ``num_qo_heads`` carries a default
    here while both subclasses add required fields, which is an invalid
    positional dataclass field order.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int | None = None

    def __post_init__(self):
        if self.num_qo_heads is None:
            self.num_qo_heads = self.num_kv_heads
        self._unsharded_kv_heads = self.num_kv_heads
        self._unsharded_qo_heads = self.num_qo_heads

    def shard(self, num_shards: int) -> None:
        """Narrow the head counts to one rank's slice.

        Idempotent because one KVConfig is shared by the KV resource and the
        attention resources planned against it, and each shards on construction.
        ``num_shards`` is the instance world size (tp * sp): Ulysses SP
        all-to-alls heads, so attention runs at head-degree tp*sp.
        """
        from mstar.distributed.utils import divide

        if num_shards >= self._unsharded_kv_heads:
            # fewer KV heads than ranks — every rank holds a replicated head
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = divide(self._unsharded_kv_heads, num_shards)
        self.num_qo_heads = divide(self._unsharded_qo_heads, num_shards)

    @abstractmethod
    def apply_yaml_overrides(self, **kwargs) -> None:
        """Patch this deployment's tunables; see ``NodeResourceSpec``.

        Abstract so each storage policy names exactly what it accepts and lets
        the rest raise — a paged key against a ring config is a typo, and the
        spec's contract is that a typo is loud.
        """


@dataclass(kw_only=True)
class PagedKVConfig(KVConfig):
    """Fixed-size pages, appended to as a sequence grows. The default."""

    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    layout: KVLayout = KVLayout.NHD
    # pages of pinned host memory to keep for offloading; 0 disables it
    cpu_offload_pages: int = 0

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        max_seq_len: int | None = None,
        cpu_offload_pages: int | None = None,
    ) -> None:
        """How much cache this deployment gets, and how it is cut up."""
        for name, value in (
            ("max_num_pages", max_num_pages),
            ("page_size", page_size),
            ("max_seq_len", max_seq_len),
            ("cpu_offload_pages", cpu_offload_pages),
        ):
            if value is not None:
                setattr(self, name, value)


@dataclass(frozen=True)
class RingKVLayerConfig:
    """One layer's ring geometry. Layers can hold frames at different strides."""

    ring_frames: int
    ring_buckets: int
    pinned_dilation: int


@dataclass(kw_only=True)
class RingKVConfig(KVConfig):
    """A fixed horizon of frame slots per layer, overwritten in place.

    Every layer's storage is allocated once and reused for the life of the process,
    and a write to an occupied slot is the intended behaviour.
    """

    tokens_per_frame: int
    layers: tuple[RingKVLayerConfig, ...]
    # How many worlds are resident at once. NOT a batch.
    num_worlds: int = 1

    def __post_init__(self):
        super().__post_init__()
        if len(self.layers) != self.num_layers:
            raise ValueError(
                f"ring geometry has {len(self.layers)} layers but num_layers is "
                f"{self.num_layers}; each layer's ring is declared separately."
            )
        if self.num_worlds < 1:
            raise ValueError(
                f"num_worlds must be >= 1; got {self.num_worlds}. A node serving "
                "zero worlds refuses every request at admit."
            )

    def apply_yaml_overrides(self, num_worlds: int | None = None, **kwargs) -> None:
        """``num_worlds`` only. Nothing else here is a deployment knob."""
        if kwargs:
            raise TypeError(
                "ring KV geometry is a checkpoint fact, not a deployment tunable; "
                f"got {sorted(kwargs)}"
            )
        if num_worlds is not None:
            if int(num_worlds) < 1:
                raise ValueError(
                    f"num_worlds must be >= 1; got {num_worlds}. A node serving "
                    "zero worlds refuses every request at admit."
                )
            self.num_worlds = int(num_worlds)


@dataclass
class KVReqConfig(ResourceReqConfig):
    # NOTE: this may need to be refined
    needed_labels: list[str] | None = None
    needed_labels_per_node: dict[str, list[str]] = field(default_factory=dict)
    needed_labels_per_node_walk: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    def get_labels(self, node: str, walk: str):
        if (node, walk) in self.needed_labels_per_node_walk:
            return self.needed_labels_per_node_walk[(node, walk)]
        if node in self.needed_labels_per_node:
            return self.needed_labels_per_node[node]
        if self.needed_labels is not None:
            return self.needed_labels
        return ["main"]


@dataclass
class KVSpec(NodeResourceSpec):
    config: KVConfig

    @property
    def resource_class(self) -> "type[Resource]":
        if isinstance(self.config, RingKVConfig):
            from mstar.engine.resources.kv.ring.manager import RingKVManager

            return RingKVManager

        from mstar.engine.resources.kv.manager import KVManager

        return KVManager

    def apply_yaml_overrides(self, **kwargs):
        """Forwarded to the config: which keys are legal is a property of the
        storage strategy, so the config answers for them."""
        self.config.apply_yaml_overrides(**kwargs)


@dataclass(frozen=True)
class KVStep(ResourceStep):
    # write: bool # @nsagan: opting to remove this for now bc it's dead code
    commit: bool = True

    # e.g., for batched CFG
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict)
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, kw_only=True)
class RingKVStep(ResourceStep):
    """One ring clock per request in the step's batch."""

    frames: tuple[tuple[str, int], ...]
