"""What a model declares about a KV cache: its shape, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, RoutingContext
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.distributed.communication import JointGroups
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.context_parallel import ContextParallel


class KVLayout(Enum):
    NHD = "NHD"
    # Multi-head Latent Attention: one compressed latent per token per layer,
    # ``[num_layers, max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim]``.
    # No K/V axis and a single (replicated) KV head; only query heads shard.
    MLA = "MLA"


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
        self._head_shards: int | None = None

    def shard(self, num_shards: int) -> None:
        """Narrow the head counts to one rank's slice of ``num_shards`` (``EngineResourceInfo.head_shards``).

        Idempotent because one KVConfig is shared by the KV resource and the
        attention resources planned against it, and each shards on construction;
        a second count is an error, since the cache was sized by the first.
        """
        from mstar.distributed.utils import divide, kv_head_replicas

        if self._head_shards not in (None, num_shards):
            raise ValueError(
                f"KV heads already sharded {self._head_shards} ways, and a resource over the pool "
                f"shards them {num_shards} ways: its builders disagree on the node's head shards"
            )
        kv, qo = self._unsharded_kv_heads, self._unsharded_qo_heads
        if (num_shards % kv if num_shards >= kv else kv % num_shards) or qo % num_shards:
            raise ValueError(
                f"{kv} KV and {qo} query heads don't split {num_shards} ways: the node's head shards "
                "must divide the query heads, and divide or be a multiple of the KV heads"
            )
        self._head_shards = num_shards
        self.num_kv_heads = kv_head_replicas(kv, num_shards)[0]
        self.num_qo_heads = divide(self._unsharded_qo_heads, num_shards)

    @abstractmethod
    def apply_yaml_overrides(self, **kwargs) -> None:
        """Patch this deployment's tunables; see ``NodeResourceSpec``.

        Abstract so each storage policy names exactly what it accepts and lets
        the rest raise — a paged key against a ring config is a typo, and the
        spec's contract is that a typo is loud.
        """


DEFAULT_PAGE_SIZE = 128
# The fewest tokens a page of either group holds when the page size is derived: a window
# keeps up to a page past it, and a smaller page adds page-table entries and plan work a token.
MIN_GROUP_PAGE = 64


@dataclass(kw_only=True)
class PagedKVConfig(KVConfig):
    """Fixed-size pages, appended to as a sequence grows. The default."""

    max_seq_len: int
    # Pages of ``page_size`` tokens of every layer, or of DEFAULT_PAGE_SIZE tokens where
    # ``page_size`` is None: the pool's bytes, whatever page a window derives (`num_units`)
    max_num_pages: int = 2048
    # As declared; None derives the page (`tokens_per_page`)
    page_size: int | None = None
    layout: KVLayout = KVLayout.NHD
    # pages of pinned host memory to keep for offloading, counted as ``max_num_pages`` counts;
    # 0 disables it
    cpu_offload_pages: int = 0
    # folded into the root, not checked at match time
    prefix_cache_salt: str = ""
    prefix_cache: bool = True
    prefix_cache_tp: bool = False
    # place token positions over the ranks that would otherwise hold the same KV heads; see `place`
    context_parallel: bool = False
    # MLA only: the latent is ``[ckv (kv_lora_rank) | kpe (qk_rope_head_dim)]``
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    # MLA: whether num_qo_heads was given. Unset it defaults to the one latent KV head, which
    # the attention backend would plan; a cache only stored (index keys) needs no query heads
    qo_heads_given: bool = field(default=False, init=False, repr=False, compare=False)
    # whether the serving config's ``resources:`` block turned the prefix cache on itself
    prefix_cache_given: bool = field(default=False, init=False, repr=False, compare=False)
    # Derived, never declared: the layers cut by page size and retention, the layers no window
    # is kept for first (`keep_window`, which `derive_kv_groups` calls as the specs resolve,
    # before anything builds or sizes from the pool). One group of every layer until then.
    groups: tuple["KVGroup", ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.layout == KVLayout.MLA:
            if self.kv_lora_rank is None or self.qk_rope_head_dim is None:
                raise ValueError("KVLayout.MLA needs kv_lora_rank and qk_rope_head_dim")
            self.qo_heads_given = self.num_qo_heads is not None
            # the cache stores one latent per token: a single KV "head" of the
            # latent width, whatever the model's head counts are
            self.num_kv_heads = 1
            self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        super().__post_init__()
        self.keep_window(None, ())

    def keep_window(self, window: int | None, layers: tuple[int, ...]) -> None:
        """Keep about ``window`` tokens per request of ``layers``, some of the pool's, and every
        token of the rest; ``layers`` empty keeps every token of every layer.

        A page of either group is the same bytes, so both draw from one arena: with F full and
        S sliding layers, a page holds ``tokens_per_page`` tokens of the full layers or
        ``tokens_per_page·F/S`` of the sliding ones. Derived, it is (S/g)·k, g = gcd(F, S), for
        the smallest k that puts at least MIN_GROUP_PAGE tokens in a page of either group."""
        full = tuple(i for i in range(self.num_layers) if i not in layers)
        if layers and not full:
            raise ValueError("every layer keeps a window; a pool keeps one only beside layers that keep every token")
        if self.page_size is not None:
            tokens = self.page_size
        elif layers:
            g = math.gcd(len(full), len(layers))
            tokens = len(layers) // g * -(-MIN_GROUP_PAGE * g // min(len(full), len(layers)))
        else:
            tokens = DEFAULT_PAGE_SIZE
        groups = (KVGroup(full, tokens),)
        if layers:
            step = len(layers) // math.gcd(len(full), len(layers))
            if tokens % step:
                raise ValueError(
                    f"page_size {self.page_size} with {len(full)} full layers gives the "
                    f"{len(layers)} sliding ones {self.page_size * len(full) / len(layers):g}-token "
                    f"pages; use a multiple of {step}"
                )
            page = tokens * len(full) // len(layers)
            # the window in whole pages: a stream releases its front pages past it at commit
            groups += (KVGroup(tuple(layers), page, RetentionPolicy(context_budget=-(-window // page) * page)),)
        self.groups = groups

    @property
    def tokens_per_page(self) -> int:
        """Tokens a page of the first group's layers holds: ``page_size``, or where it is None
        DEFAULT_PAGE_SIZE, or with a window the smallest page whole for both groups
        (`keep_window`)."""
        return self.groups[0].page_size

    @property
    def num_units(self) -> int:
        """Arena pages, each a page of one group's layers: the bytes of ``max_num_pages``."""
        return self.units(self.max_num_pages)

    def units(self, pages: int) -> int:
        """Arena pages in the bytes of ``pages`` pages as ``max_num_pages`` counts them."""
        return pages * self._counted_page // self._arena_page

    def pages_of(self, units: int, round_up: bool = False) -> int:
        """``units`` arena pages in pages as ``max_num_pages`` counts them, rounded down (what
        they hold), or with ``round_up`` up (what holds them)."""
        if round_up:
            return -(-units * self._arena_page // self._counted_page)
        return units * self._arena_page // self._counted_page

    @property
    def _counted_page(self) -> int:
        # tokens of every layer in a page as max_num_pages counts it
        return (DEFAULT_PAGE_SIZE if self.page_size is None else self.page_size) * self.num_layers

    @property
    def _arena_page(self) -> int:
        # tokens of every layer in an arena page, a page of the first group's layers
        return self.tokens_per_page * len(self.groups[0].layers)

    def keeps_prefix_cache(self, world_size: int, checkpoint_named: bool) -> bool:
        """`NodeResourceSpec.keeps_prefix_cache` for a pool of this config.
        Above one rank only with ``prefix_cache_tp``, and a model that names no
        checkpoint needs a salt, or builds that differ only in their weights
        could not be told apart."""
        return (
            self.prefix_cache
            and (world_size == 1 or self.prefix_cache_tp)
            and (checkpoint_named or bool(self.prefix_cache_salt))
        )

    def place(self, joint: "JointGroups | None", pool: str) -> "ContextParallel | None":
        """This rank's placement of the pool, resolved from ``context_parallel``; None when it is off.

        The KV cache holds it (``KVCache.placement``), not this config, which every rank's
        resources share. The partners are the node's SP ranks when it has more than one, and
        otherwise the block of TP ranks holding this rank's KV head (`kv_head_replicas`).
        """
        from mstar.engine.resources.context_parallel import ContextParallel

        if not self.context_parallel:
            return None
        if len(self.groups) > 1:
            # a page of a group's layers holds another page size, which the placed storage doesn't stripe
            raise ValueError(f"KV {pool!r}: context_parallel is unsupported over a pool that keeps a sliding window")
        tp_size, sp_size = (1, 1) if joint is None else (joint.tp_group.world_size, joint.sp_group.world_size)
        # raises on one rank alone, so past here there is a joint group
        size = self.context_parallel_size(pool, tp_size, sp_size)
        if sp_size > 1:
            sp = joint.sp_group
            return ContextParallel(size=size, index=sp.rank, group=sp, over_sp=True, partners=tuple(range(size)))
        tp = joint.tp_group
        first = tp.rank // size * size
        return ContextParallel(
            size=size, index=tp.rank % size, group=tp, over_sp=False, partners=tuple(range(first, first + size)),
        )

    def context_parallel_size(self, pool: str, tp: int, sp: int) -> int:
        """How many ranks ``context_parallel`` stripes each page over on a node of ``tp`` x ``sp``
        ranks: its SP ranks when it has more than one, else the TP ranks sharing a KV head.
        Raises ``ValueError`` when they can't; ``validate_deployment`` checks it from
        ``node_groups`` before any worker starts."""
        from mstar.distributed.utils import kv_head_replicas

        if sp > 1:
            size = sp
        else:
            heads = self._unsharded_kv_heads
            if heads >= tp or tp % heads:
                raise ValueError(
                    f"KV {pool!r}: context_parallel places positions over the TP ranks that share a "
                    f"KV head, and {heads} KV head(s) over {tp} TP rank(s) "
                    f"{'leave none sharing one' if heads >= tp else 'do not divide evenly'}. Make TP a "
                    "larger multiple of the KV heads, give the node SP ranks, or turn "
                    "context_parallel off."
                )
            size = kv_head_replicas(heads, tp)[1]
        if self.tokens_per_page % size:
            raise ValueError(
                f"KV {pool!r}: context_parallel stripes each page over {size} ranks, "
                f"which doesn't divide page_size {self.tokens_per_page}"
            )
        return size

    @property
    def request_pages(self) -> int:
        """Arena pages requests can hold: all but the sink page the KV manager takes at build."""
        return self.num_units - 1

    def stream_pages(self, layout: "StreamLayout", caps: Mapping[str, int]) -> int:
        """The most arena pages a stream laid out as ``layout`` holds at ``caps``: its pages in
        every group, each window group's under that group's retention."""
        first, *later = self.groups
        return layout.pages(caps, first.page_size) + sum(
            layout.pages(caps, group.page_size, group.retention) for group in later
        )

    @property
    def latent_dim(self) -> int:
        """Per-token cache width under MLA."""
        assert self.layout == KVLayout.MLA
        return self.kv_lora_rank + self.qk_rope_head_dim

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        max_seq_len: int | None = None,
        cpu_offload_pages: int | None = None,
        prefix_cache_salt: str | None = None,
        prefix_cache: bool | None = None,
        prefix_cache_tp: bool | None = None,
        context_parallel: bool | None = None,
    ) -> None:
        """How much cache this deployment gets, and how it is cut up."""
        for name, value in (
            ("max_num_pages", max_num_pages),
            ("page_size", page_size),
            ("max_seq_len", max_seq_len),
            ("cpu_offload_pages", cpu_offload_pages),
            ("prefix_cache_salt", prefix_cache_salt),
            ("prefix_cache", prefix_cache),
            ("prefix_cache_tp", prefix_cache_tp),
            ("context_parallel", context_parallel),
        ):
            if value is not None:
                setattr(self, name, value)
        if prefix_cache is not None:
            self.prefix_cache_given = prefix_cache
        if page_size is not None:
            # every token kept at this page size until the specs derive the window again
            self.keep_window(None, ())


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
    # How many sessions are resident at once. NOT a batch.
    num_sessions: int = 1

    @property
    def total_sessions(self) -> int:
        """Resident sessions plus one shared scratch session for padding rows.

        A replay padded to its capture bucket parks the dummy tail on this session
        (see ``RingKVManager.plan``); it is never handed to a request, so
        resident capacity stays ``num_sessions`` and the deployment knob keeps its
        meaning. The ring buffer and the flex mask both size on this count so the
        padding session is a real, addressable span.
        """
        return self.num_sessions + 1

    def __post_init__(self):
        super().__post_init__()
        if len(self.layers) != self.num_layers:
            raise ValueError(
                f"ring geometry has {len(self.layers)} layers but num_layers is "
                f"{self.num_layers}; each layer's ring is declared separately."
            )
        if (
            not isinstance(self.num_sessions, int)
            or isinstance(self.num_sessions, bool)
            or self.num_sessions < 1
        ):
            raise ValueError(
                f"num_sessions must be a positive int; got {self.num_sessions!r}. A node serving "
                "zero sessions refuses every request at admit."
            )

    def apply_yaml_overrides(self, num_sessions: int | None = None, **kwargs) -> None:
        """``num_sessions`` only. Nothing else here is a deployment knob."""
        if kwargs:
            raise TypeError(
                "ring KV geometry is a checkpoint fact, not a deployment tunable; "
                f"got {sorted(kwargs)}"
            )
        if num_sessions is not None:
            if (
                not isinstance(num_sessions, int)
                or isinstance(num_sessions, bool)
                or num_sessions < 1
            ):
                raise ValueError(
                    f"num_sessions must be a positive int; got {num_sessions!r}. A node serving "
                    "zero sessions refuses every request at admit."
                )
            self.num_sessions = num_sessions


@dataclass(frozen=True)
class PrefixSpan:
    """One write to a keyed stream, as the preprocess worker laid it out.

    ``length`` is how many slots the write fills and ``advance`` is how far it
    moves the position counter. ``digest`` is an item's digest, and None for
    ids. ``end_key`` is set for an item that ends inside a page, and keys that
    page up to the item's end.
    """
    length: int
    advance: int
    walk: str
    scratch: int = 0
    digest: bytes | None = None
    end_key: bytes | None = None


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
    kv_layout: dict[str, "StreamLayout"] | None = None
    loop_caps: dict[str, int] | None = None

    def apply_conductor_config(
        self,
        prefix_keys: dict[str, list[bytes]] | None=None,
        prefix_tail: dict[str, list[int]] | None=None,
        prefix_decode: dict[str, str] | None=None,
        prefix_cache: bool | None=None,
        kv_layout: dict[str, list] | None=None,
        loop_caps: Mapping[str, int] | None=None,
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
        if kv_layout is not None:
            # a model lays out every label the request could open on any deployment, so keep only the ones it opens
            opened = self.opened_labels()
            self.kv_layout = {
                label: StreamLayout(
                    spans=tuple(PrefixSpan(*row) for row in spans),
                    growth=tuple((loop, slots) for loop, slots in growth),
                    scratch=scratch,
                    retention=None if retention is None else RetentionPolicy(*retention),
                    keyed_spans=keyed_spans,
                )
                for label, (spans, growth, scratch, retention, keyed_spans) in kv_layout.items()
                if label in opened
            }
        if loop_caps is not None and self.kv_layout is not None:
            self.loop_caps = {
                loop: loop_caps[loop]
                for layout in self.kv_layout.values() for loop, _ in layout.growth
            }

    def loop_room(self, spec: "KVSpec | None" = None) -> dict[str, int]:
        if spec is None or not isinstance(spec.config, PagedKVConfig):
            return {}
        caps, room = self.loop_caps or {}, {}
        for layout in (self.kv_layout or {}).values():
            if layout.retention is not None:
                continue
            for loop, slots in layout.growth:
                n = (spec.config.max_seq_len - layout.positions(caps) + slots * caps[loop]) // slots
                room[loop] = min(n, room.get(loop, n))
        pool = spec.config.request_pages
        if self.pages(spec.nodes, caps, spec.config) > pool:
            # each loop's room in pages, searched with the other loops at their caps
            for loop, cap in caps.items():
                low, high = 0, cap
                while low < high:
                    mid = (low + high + 1) // 2
                    if self.pages(spec.nodes, {**caps, loop: mid}, spec.config) <= pool:
                        low = mid
                    else:
                        high = mid - 1
                room[loop] = min(low, room.get(loop, low))
        return room

    def validate(self, spec: "KVSpec | None" = None) -> None:
        """``ValueError`` (a 400) when a label would hold more positions at the request's caps
        than its pool's ``max_seq_len``. A label under retention is left out, since Cosmos3,
        the one model that declares one, computes its positions in each forward.

        Also when the labels ``spec.nodes`` open would hold more pages at the caps than the pool
        has for requests. The pool admits on the prompt alone, so such a request would be admitted
        and then could not finish even alone in an empty pool."""
        if spec is None or not isinstance(spec.config, PagedKVConfig):
            return
        for label, layout in (self.kv_layout or {}).items():
            positions = layout.positions(self.loop_caps or {})
            if layout.retention is None and positions > spec.config.max_seq_len:
                raise ValueError(
                    f"this request needs {positions} positions of {label!r}, "
                    f"past the model's context of {spec.config.max_seq_len}"
                )
        pages = self.pages(spec.nodes, self.loop_caps or {}, spec.config)
        if pages > spec.config.request_pages:
            # counted in arena pages; the message counts pages of every layer, as max_num_pages does
            raise ValueError(
                f"this request needs {spec.config.pages_of(pages, round_up=True)} pages of "
                f"{spec.resource_key!r} for labels {sorted(self.opened_labels(spec.nodes))}, and the pool "
                f"has {spec.config.pages_of(spec.config.request_pages)}"
            )

    def get_labels(self, node: str, walk: str):
        if (node, walk) in self.needed_labels_per_node_walk:
            return self.needed_labels_per_node_walk[(node, walk)]
        if node in self.needed_labels_per_node:
            return self.needed_labels_per_node[node]
        if self.needed_labels is not None:
            return self.needed_labels
        return ["main"]

    def opened_labels(self, nodes: Collection[str] | None = None) -> set[str]:
        """The labels the request opens on ``nodes``, or on every node when ``nodes`` is None."""
        labels = set(self.needed_labels or ["main"])
        for node, named in self.needed_labels_per_node.items():
            if nodes is None or node in nodes:
                labels.update(named)
        for (node, _), named in self.needed_labels_per_node_walk.items():
            if nodes is None or node in nodes:
                labels.update(named)
        return labels

    def pages(self, nodes: Collection[str] | None, caps: Mapping[str, int], pool: PagedKVConfig) -> int:
        """The most arena pages the request holds in ``pool``, shared by ``nodes``, at ``caps``."""
        layouts = self.kv_layout or {}
        return sum(pool.stream_pages(layouts[label], caps) for label in self.opened_labels(nodes) if label in layouts)

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
        # Final publication is a new, opt-in hook. Ordinary publication keeps
        # the legacy "all available labels" behavior when its map is absent.
        if final and mapping is None:
            return []
        if mapping is None:
            return available
        if node is None or walk is None:
            return []
        return mapping.get((node, walk), [])



def refuse_undeclared_labels(
    model: str, declared: Mapping[str, tuple[str, ...]], kv_layout: Mapping[str, Mapping],
    configs: Mapping[str, ResourceReqConfig],
) -> None:
    """The model is at fault and not the client, so this raises RuntimeError, which fails the request with a 500."""
    for key in declared.keys() | kv_layout.keys():
        laid_out = set(kv_layout.get(key, ()))
        if extra := sorted(laid_out - set(declared.get(key, ()))):
            raise RuntimeError(f"{model} returned a kv_layout for {key!r}/{extra[0]!r}, which it does not declare")
        if missing := sorted(configs[key].opened_labels() - laid_out):
            raise RuntimeError(f"{model} returned no kv_layout for {key!r}/{missing[0]!r}, which the request opens")


def without_layouts(configs: Mapping[str, ResourceReqConfig]) -> dict[str, ResourceReqConfig]:
    """Only the KV manager's ingest reads the layouts and caps, so they are dropped
    before the configs go out with every frame between workers."""
    return {
        key: replace(config, kv_layout=None, loop_caps=None) if isinstance(config, KVReqConfig) else config
        for key, config in configs.items()
    }


def instance_size(spec: NodeResourceSpec, instance: Mapping[str, int]) -> int:
    """What `opening_prefix_caches` sizes ``spec`` at: the smallest lockstep
    instance of its nodes in ``instance``, 1 where it has none."""
    return min((instance[node] for node in spec.nodes if node in instance), default=1)


def unrestorable(specs: Mapping[str, NodeResourceSpec], opened: Collection[str]) -> list[NodeResourceSpec]:
    """The resources that can't restore what a hit skips while ``opened`` open:
    those that set `NodeResourceSpec.restores_prefix` and keep no open cache of
    their own, unless they are over open caches that restore. Each keeps every
    cache on its nodes shut."""
    return [
        spec for spec in specs.values()
        if spec.restores_prefix and spec.resource_key not in opened and not (
            spec.depends_on() and all(dep in opened and specs[dep].restores_prefix for dep in spec.depends_on())
        )
    ]


def opening_prefix_caches(
    specs: Iterable[NodeResourceSpec], instance: Mapping[str, int], checkpoint_named: bool,
) -> frozenset[str]:
    """The prefix caches that open: those each spec keeps of its own
    (`NodeResourceSpec.keeps_prefix_cache`), less those a resource on their
    nodes keeps from serving a hit. Every input is fixed at startup, so the
    engines, the conductor's routing and the data worker agree.

    ``instance`` is the lockstep instance of each node the caller places
    (`instance_size`). A cache over other caches, such as a recurrent pool's
    snapshots over its KV cache's pages, keeps only where they open on all of
    its nodes. A node carrying a resource that can't restore what a hit skips
    (`unrestorable`) keeps every cache on it shut: a hit there would skip
    tokens its state never saw. Closing one cache can close others, so the rule
    runs until nothing changes.
    """
    by_key = {spec.resource_key: spec for spec in specs}
    opened = {
        key for key, spec in by_key.items() if spec.keeps_prefix_cache(instance_size(spec, instance), checkpoint_named)
    }
    while True:
        kept = {
            key for key in opened
            if all(dep in opened and by_key[key].nodes <= by_key[dep].nodes for dep in by_key[key].depends_on())
        }
        blocked = {node for spec in unrestorable(by_key, kept) for node in spec.nodes}
        now = {key for key in kept if by_key[key].nodes.isdisjoint(blocked)}
        if now == opened:
            return frozenset(now)
        opened = now


def prefix_caches_by_rank(
    specs: Iterable[NodeResourceSpec],
    groups: Iterable[tuple[Sequence[int], int, Collection[str]]],
    checkpoint_named: bool,
) -> dict[int, frozenset[str]]:
    """The caches each rank's engine opens: `opening_prefix_caches` over the
    nodes on the rank, each at its lockstep instance size, as `Engine.load_model`
    evaluates it, less the caches with no node there, which that engine counts at
    one rank for its item encoders but never builds. ``groups`` gives each node
    group's ranks, instance size and nodes."""
    specs = list(specs)
    local: dict[int, dict[str, int]] = {}
    for ranks, size, nodes in groups:
        for rank in ranks:
            local.setdefault(rank, {}).update(dict.fromkeys(nodes, size))
    return {
        rank: opening_prefix_caches(specs, nodes, checkpoint_named)
        & {spec.resource_key for spec in specs if not spec.nodes.isdisjoint(nodes)}
        for rank, nodes in local.items()
    }


@dataclass
class KVSpec(NodeResourceSpec):
    config: KVConfig
    # Under CFG parallel, the node whose pool admits each request. The other nodes' pools hold no
    # more pages per request than the leader's, as long as the leader offloads nothing.
    leader: str | None = None

    @property
    def resource_class(self) -> "type[Resource]":
        if isinstance(self.config, RingKVConfig):
            from mstar.engine.resources.kv.ring.manager import RingKVManager

            return RingKVManager

        from mstar.engine.resources.kv.manager import KVManager

        return KVManager

    @property
    def places_positions(self) -> bool | None:
        # a paged pool's (`PagedKVConfig.context_parallel`); a ring keeps every position on each rank
        return self.config.context_parallel if isinstance(self.config, PagedKVConfig) else None

    def apply_yaml_overrides(self, **kwargs):
        """Forwarded to the config: which keys are legal is a property of the
        storage strategy, so the config answers for them."""
        self.config.apply_yaml_overrides(**kwargs)

    def keeps_prefix_cache(self, world_size: int, checkpoint_named: bool) -> bool:
        return isinstance(self.config, PagedKVConfig) and self.config.keeps_prefix_cache(world_size, checkpoint_named)

    def routing_mirror(self, ctx: RoutingContext):
        """A mirror of the prefix cache where it opens and a walk that keys it runs."""
        labels = [label for label, walks in ctx.keyed.items() if ctx.walks.intersection(walks)]
        if not labels or not ctx.prefix_cache_opens:
            return None
        from mstar.engine.resources.kv.routing import KVRoutingMirror

        return KVRoutingMirror(ctx.num_replicas, self.config.num_units, self.config.tokens_per_page, labels)

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        # the attentions over a pool decide what it keeps, and their module imports this one
        from mstar.engine.resources.attn.config import derive_kv_groups

        return derive_kv_groups(self.resource_key, self.config, specs)


@dataclass(frozen=True)
class RetentionPolicy:
    """FIFO retention for a stream that keeps committing (windowed / rolling
    generation): once the committed tokens behind ``protected_prefix`` exceed
    ``context_budget``, the oldest unprotected pages are released.

    A committing step declares it for its stream on ``KVStep.retention`` and
    ``KVManager.commit`` applies it, so the release happens between steps as
    far as every planner is concerned: a pre-plan of the next step gates on
    this commit and sees the compacted stream. Whole pages only (the page
    straddling the prefix boundary and a partial tail page stay), so the
    realized context can run over the budget by up to a page; the excess is
    re-offered at the next commit. ``protected_prefix`` tokens at the head (a
    text prompt, say) are never released.
    """
    context_budget: int
    # front tokens the window never releases; only pages within them are indexed
    protected_prefix: int = 0

    def __post_init__(self):
        if self.context_budget <= 0:
            raise ValueError(f"context_budget must be > 0, got {self.context_budget}")
        if self.protected_prefix < 0:
            raise ValueError(f"protected_prefix must be >= 0, got {self.protected_prefix}")

    def max_pages(self, page_size: int, step: int = 0) -> int:
        """The most pages of ``page_size`` a stream under this policy holds, with a step of
        ``step`` tokens past its prefix. `KVManager._apply_retention` leaves less than a page over
        the budget, so the window peaks at the protected prefix plus the budget, or at the prefix
        rounded up to a page if that is larger, plus a page of overrun and the step."""
        prefix = self.protected_prefix
        window = max(prefix + self.context_budget, -(-prefix // page_size) * page_size)
        return -(-(window + page_size - 1 + step) // page_size)


@dataclass(frozen=True)
class KVGroup:
    """Layers of a paged pool that share a page size and a retention (`PagedKVConfig.groups`)."""
    layers: tuple[int, ...]
    page_size: int
    # applied at each commit: what the group's attention can no longer read goes back to the pool
    retention: RetentionPolicy | None = None


@dataclass(frozen=True)
class StreamLayout:
    """`LabelLayout` as the KV manager holds it."""
    spans: tuple[PrefixSpan, ...] = ()
    growth: tuple[tuple[str, int], ...] = ()
    scratch: int = 0
    retention: RetentionPolicy | None = None
    keyed_spans: int = 0

    def slots(self, caps: Mapping[str, int]) -> int:
        return sum(span.length for span in self.spans) + sum(
            slots * caps[loop] for loop, slots in self.growth
        )

    def positions(self, caps: Mapping[str, int]) -> int:
        return sum(span.advance for span in self.spans) + sum(
            slots * caps[loop] for loop, slots in self.growth
        )

    def pages(
        self, caps: Mapping[str, int], page_size: int, retention: "RetentionPolicy | None" = None,
    ) -> int:
        """The most pages the stream can hold, where ``caps`` gives each growth loop's iterations;
        under ``retention`` (the layout's own where None), its window with the largest step past
        the prefix."""
        if retention is None:
            retention = self.retention
        peak, end = self.slots(caps) + self.scratch, 0
        for span in self.spans:
            peak = max(peak, end + span.scratch)
            end += span.length
        pages = -(-peak // page_size)
        if retention is None:
            return pages
        prefix = retention.protected_prefix
        step, end = self.scratch, 0
        for span in self.spans:
            end += span.length
            if max(end, end - span.length + span.scratch) > prefix:
                step = max(step, span.length, span.scratch)
        step = max([step, *(slots for _, slots in self.growth)])
        return min(pages, retention.max_pages(page_size, step))


@dataclass(frozen=True)
class KVStep(ResourceStep):
    # write: bool # @nsagan: opting to remove this for now bc it's dead code
    commit: bool = True
    # The step writes this cache's planned tokens eagerly, outside any captured
    # graph (e.g. in ``preprocess``, before the replay). Its addressing stays an
    # ordinary plan state rather than the slot's static buffers, which are sized
    # for the bucket's in-graph tokens: a decoder prefill whose encoder context
    # is far longer than its prompt can then replay a graph sized by the prompt.
    eager_write: bool = False

    # e.g., for batched CFG
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict)
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()
    # the rows the forks apply to; None is every row. A chunked prompt forks
    # on some of its chunks only, e.g. a pre-fork on its first
    fork_rids: frozenset | None = None
    # (request_id, label) pairs that run against SINK_PAGE and hold no pages
    sink: frozenset[tuple[str, str]] = frozenset()
    # (request_id, label) -> the retention a committing stream declares for
    # itself this step, applied at commit. It rides with the step rather than
    # living on the stream, so an offload has nothing to lose.
    retention: Mapping[tuple[str, str], RetentionPolicy] = field(default_factory=dict)
    # Set by the manager when the step commits, while copies to the host are on: its count of commits so far.
    # The submodule doesn't declare it, as it doesn't declare `SubmoduleStep`'s context.
    commits: int = field(default=0, init=False, compare=False, repr=False)


@dataclass(frozen=True, kw_only=True)
class RingKVStep(ResourceStep):
    """One ring clock per request in the step's batch."""

    frames: tuple[tuple[str, int], ...]
