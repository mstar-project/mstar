"""What a model declares about a KV cache: its shape, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, RoutingContext
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


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

    def shard(self, num_shards: int) -> None:
        """Narrow the head counts to one rank's slice.

        Idempotent because one KVConfig is shared by the KV resource and the
        attention resources planned against it, and each shards on construction.
        ``num_shards`` is the instance session size (tp * sp): Ulysses SP
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
    # folded into the root, not checked at match time
    prefix_cache_salt: str = ""
    prefix_cache: bool = True
    prefix_cache_tp: bool = False
    # MLA only: the latent is ``[ckv (kv_lora_rank) | kpe (qk_rope_head_dim)]``
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    # MLA: whether num_qo_heads was given. Unset it defaults to the one latent KV head, which
    # the attention backend would plan; a cache only stored (index keys) needs no query heads
    qo_heads_given: bool = field(default=False, init=False, repr=False, compare=False)

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

    def prefix_cache_opens(self, world_size: int, checkpoint_named: bool) -> bool:
        """Whether a pool of this config opens its prefix cache. Every input is
        fixed at startup: ``world_size`` is the pool's lockstep instance, and a
        model that names no checkpoint needs a salt, or builds that differ only
        in their weights could not be told apart. The conductor's routing reads
        the same rule the workers open their caches by."""
        return (
            self.prefix_cache
            and (world_size == 1 or self.prefix_cache_tp)
            and (checkpoint_named or bool(self.prefix_cache_salt))
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
        return room

    def validate(self, spec: "KVSpec | None" = None) -> None:
        """``ValueError`` (a 400) when a label would hold more positions at the request's caps
        than its pool's ``max_seq_len``. A label under retention is left out, since Cosmos3,
        the one model that declares one, computes its positions in each forward."""
        if spec is None or not isinstance(spec.config, PagedKVConfig):
            return
        for label, layout in (self.kv_layout or {}).items():
            positions = layout.positions(self.loop_caps or {})
            if layout.retention is None and positions > spec.config.max_seq_len:
                raise ValueError(
                    f"this request needs {positions} positions of {label!r}, "
                    f"past the model's context of {spec.config.max_seq_len}"
                )

    def get_labels(self, node: str, walk: str):
        if (node, walk) in self.needed_labels_per_node_walk:
            return self.needed_labels_per_node_walk[(node, walk)]
        if node in self.needed_labels_per_node:
            return self.needed_labels_per_node[node]
        if self.needed_labels is not None:
            return self.needed_labels
        return ["main"]

    def opened_labels(self) -> set[str]:
        labels = set(self.needed_labels or ["main"])
        for named in (*self.needed_labels_per_node.values(), *self.needed_labels_per_node_walk.values()):
            labels.update(named)
        return labels

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

    def routing_mirror(self, ctx: RoutingContext):
        if not isinstance(self.config, PagedKVConfig) or not self.config.prefix_cache_opens(
            ctx.instance_size, ctx.checkpoint_named,
        ):
            return None
        from mstar.engine.resources.kv.routing import KVRoutingMirror

        return KVRoutingMirror(
            ctx.num_replicas,
            # without SINK_PAGE, which the pool never hands out
            capacity=self.config.max_num_pages - 1, page_size=self.config.page_size,
        )


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

    def pages(self, caps: Mapping[str, int], page_size: int) -> int:
        """The most pages the stream can hold, where ``caps`` gives each growth loop's iterations.

        Under ``retention`` the stream holds a window. `KVManager._apply_retention` leaves
        less than a page over the budget, so the window peaks at the protected prefix plus
        the budget, or at the prefix rounded up to a page if that is larger. The bound adds
        a page of overrun and the largest step past the prefix to that peak.
        """
        peak, end = self.slots(caps) + self.scratch, 0
        for span in self.spans:
            peak = max(peak, end + span.scratch)
            end += span.length
        pages = -(-peak // page_size)
        if self.retention is None:
            return pages
        prefix = self.retention.protected_prefix
        step, end = self.scratch, 0
        for span in self.spans:
            end += span.length
            if max(end, end - span.length + span.scratch) > prefix:
                step = max(step, span.length, span.scratch)
        step = max([step, *(slots for _, slots in self.growth)])
        window = max(prefix + self.retention.context_budget, -(-prefix // page_size) * page_size)
        return min(pages, -(-(window + page_size - 1 + step) // page_size))


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
    # the rows the forks apply to; None is every row (a chunked prompt
    # pre-forks on its first chunk and post-forks on its last)
    fork_rids: frozenset | None = None
    # (request_id, label) pairs that run against SINK_PAGE and hold no pages
    sink: frozenset[tuple[str, str]] = frozenset()
    # (request_id, label) -> the retention a committing stream declares for
    # itself this step, applied at commit. It rides with the step rather than
    # living on the stream, so an offload has nothing to lose.
    retention: Mapping[tuple[str, str], RetentionPolicy] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class RingKVStep(ResourceStep):
    """One ring clock per request in the step's batch."""

    frames: tuple[tuple[str, int], ...]
