"""What a model declares about a KV cache: its shape, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

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


@dataclass
class KVConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    num_qo_heads: int = None
    layout: KVLayout = KVLayout.NHD
    # pages of pinned host memory to keep for offloading; 0 disables it
    cpu_offload_pages: int = 0
    # folded into the root, not checked at match time
    prefix_cache_salt: str = ""
    prefix_cache: bool = True

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


@dataclass
class KVReqConfig(ResourceReqConfig):
    # NOTE: this may need to be refined
    needed_labels: list[str] | None = None
    needed_labels_per_node: dict[str, list[str]] = field(default_factory=dict)
    needed_labels_per_node_walk: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # None preserves the generic behavior of publishing every stream. A
    # mapping lets a model export only labels that another instance may read;
    # an absent key in an explicit mapping means this step publishes no KV.
    publish_labels_per_node_walk: (
        dict[tuple[str, str], list[str]] | None
    ) = None
    # Stop-time labels are exported once, after the final loop iteration has
    # committed. None and an absent key both mean no stop-time publication.
    final_publish_labels_per_node_walk: (
        dict[tuple[str, str], list[str]] | None
    ) = None
    # label -> one key per page, from the preprocess worker
    prefix_keys: dict[str, list[bytes]] | None = None
    # label -> prompt tokens past the last whole page, keyed once generation fills it
    prefix_tail: dict[str, list[int]] | None = None
    # label -> the output tensor its sampled ids arrive in, if the stream keys generation
    prefix_decode: dict[str, str] | None = None
    prefix_cache: bool = True

    def apply_conductor_config(
        self,
        prefix_keys: dict[str, list[bytes]] | None=None,
        prefix_tail: dict[str, list[int]] | None=None,
        prefix_decode: dict[str, str] | None=None,
        prefix_cache: bool | None=None,
        **kwargs,
    ):
        if prefix_keys is not None:
            self.prefix_keys = prefix_keys
        if prefix_tail is not None:
            self.prefix_tail = prefix_tail
        if prefix_decode is not None:
            self.prefix_decode = prefix_decode
        if prefix_cache is not None:
            self.prefix_cache = prefix_cache

    def get_labels(self, node: str, walk: str):
        if (node, walk) in self.needed_labels_per_node_walk:
            return self.needed_labels_per_node_walk[(node, walk)]
        if node in self.needed_labels_per_node:
            return self.needed_labels_per_node[node]
        if self.needed_labels is not None:
            return self.needed_labels
        return ["main"]

    def get_publish_labels(
        self,
        node: str | None,
        walk: str | None,
        available: list[str],
        *,
        final: bool = False,
    ) -> list[str]:
        mapping = (
            self.final_publish_labels_per_node_walk
            if final else self.publish_labels_per_node_walk
        )
        if final and mapping is None:
            return []
        if mapping is None:
            return available
        if node is None or walk is None:
            return []
        return mapping.get((node, walk), [])


@dataclass
class KVSpec(NodeResourceSpec):
    config: KVConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.kv.manager import KVManager

        return KVManager

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        max_seq_len: int | None = None,
        cpu_offload_pages: int | None = None,
        prefix_cache_salt: str | None = None,
        prefix_cache: bool | None = None,
    ):
        """How much cache this deployment gets, and how it is cut up."""
        for name, value in (
            ("max_num_pages", max_num_pages),
            ("page_size", page_size),
            ("max_seq_len", max_seq_len),
            ("cpu_offload_pages", cpu_offload_pages),
            ("prefix_cache_salt", prefix_cache_salt),
            ("prefix_cache", prefix_cache),
        ):
            if value is not None:
                setattr(self.config, name, value)


@dataclass(frozen=True)
class KVStep(ResourceStep):
    # write: bool # @nsagan: opting to remove this for now bc it's dead code
    commit: bool = True

    # e.g., for batched CFG
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict)
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()
