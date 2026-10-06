"""What a model declares about a KV cache: its shape, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
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


_ADMISSION_CHOICES = {
    "admission_fit": ("sum", "peak"),
    "admission_order": ("fifo", "backfill"),
    "backfill_protect": ("easy",),
}
# the environment variable that overrides each key, for flipping a deployment's
# mode without editing its YAML
_ADMISSION_ENV = {
    "admission_fit": "MSTAR_KV_ADMISSION_FIT",
    "admission_order": "MSTAR_KV_ADMISSION_ORDER",
}
_BACKFILL_WINDOW_ENV = "MSTAR_KV_BACKFILL_WINDOW"


@dataclass(kw_only=True)
class PagedKVConfig(KVConfig):
    """Fixed-size pages, appended to as a sequence grows. The default.

    How a request is admitted against the pool (the defaults are the summed
    reservations in strict arrival order, the pool's original behavior):

    ``admission_fit``: ``sum`` admits a request only if everything admitted
    could take all it reserved at once; ``peak`` admits it if the set's
    projected footprint, round by round, stays within the pool, and guards each
    page grant so the admitted requests can always finish.

    ``admission_order``: ``fifo`` lets nothing pass the request that has waited
    longest; ``backfill`` lets a request behind it be admitted if it fits and
    does not delay it, by ``backfill_protect`` (``easy``: it must be gone
    before the head could start, or fit in the room the head leaves spare).

    ``backfill_window``: under ``backfill``, only the first this-many requests
    waiting (the head included, in the order they first asked) are looked at
    for a place; the rest wait without being asked about, so a long queue costs
    no more per scheduling pass than a short one. A request behind the window
    is looked at once enough ahead of it have been admitted.

    Each of the first two, and the window, is overridden by an environment
    variable, which wins over the YAML: ``MSTAR_KV_ADMISSION_FIT``,
    ``MSTAR_KV_ADMISSION_ORDER`` and ``MSTAR_KV_BACKFILL_WINDOW``. The
    environment is read where the cache is built, so it reaches every model's
    paged pool, in each worker process that inherits it.
    """

    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    layout: KVLayout = KVLayout.NHD
    # pages of pinned host memory to keep for offloading; 0 disables it
    cpu_offload_pages: int = 0
    # folded into the root, not checked at match time
    prefix_cache_salt: str = ""
    prefix_cache: bool = True
    admission_fit: str = "sum"
    admission_order: str = "fifo"
    backfill_protect: str = "easy"
    backfill_window: int = 128

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        max_seq_len: int | None = None,
        cpu_offload_pages: int | None = None,
        prefix_cache_salt: str | None = None,
        prefix_cache: bool | None = None,
        admission_fit: str | None = None,
        admission_order: str | None = None,
        backfill_protect: str | None = None,
        backfill_window: int | None = None,
    ) -> None:
        """How much cache this deployment gets, how it is cut up, and how it admits."""
        for name, value in (
            ("max_num_pages", max_num_pages),
            ("page_size", page_size),
            ("max_seq_len", max_seq_len),
            ("cpu_offload_pages", cpu_offload_pages),
            ("prefix_cache_salt", prefix_cache_salt),
            ("prefix_cache", prefix_cache),
            ("admission_fit", admission_fit),
            ("admission_order", admission_order),
            ("backfill_protect", backfill_protect),
            ("backfill_window", backfill_window),
        ):
            if value is not None:
                setattr(self, name, value)
        self.resolved_admission()
        self.resolved_backfill_window()

    def resolved_admission(self) -> tuple[str, str]:
        """``(fit, order)`` as the pool is to run them: the environment, then the config.

        A value that names no mode is an error, as a typo in a YAML key is:
        it would otherwise leave the pool on a mode its operator did not pick.
        """
        for name, choices in _ADMISSION_CHOICES.items():
            if getattr(self, name) not in choices:
                raise ValueError(
                    f"{name} must be one of {choices}; got {getattr(self, name)!r}"
                )
        resolved = []
        for name, env in _ADMISSION_ENV.items():
            value = os.environ.get(env) or getattr(self, name)
            if value not in _ADMISSION_CHOICES[name]:
                raise ValueError(
                    f"{env} must be one of {_ADMISSION_CHOICES[name]}; got {value!r}"
                )
            resolved.append(value)
        return resolved[0], resolved[1]

    def resolved_backfill_window(self) -> int:
        """How many waiting requests backfill looks at: the environment, then the config.

        Not a positive whole number is an error, as a mode that names none is.
        """
        raw = os.environ.get(_BACKFILL_WINDOW_ENV)
        source, value = (_BACKFILL_WINDOW_ENV, raw) if raw else ("backfill_window", self.backfill_window)
        try:
            window = int(value)
        except (TypeError, ValueError):
            window = 0
        if isinstance(value, bool) or window < 1 or str(window) != str(value).strip():
            raise ValueError(f"{source} must be a positive whole number; got {value!r}")
        return window


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
    # the most tokens the request may generate; None on a row no request owns
    max_tokens: int | None = None
    # label -> tokens it holds over the request's life, decode aside, as the model counts them
    prompt_slots: dict[str, int] | None = None
    # labels decode grows, by up to max_tokens; read only beside prompt_slots
    decode_labels: list[str] | None = None

    def apply_conductor_config(
        self,
        prefix_keys: dict[str, list[bytes]] | None=None,
        prefix_tail: dict[str, list[int]] | None=None,
        prefix_decode: dict[str, str] | None=None,
        prefix_cache: bool | None=None,
        max_tokens: int | None=None,
        prompt_slots: dict[str, int] | None=None,
        decode_labels: list[str] | None=None,
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
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if prompt_slots is not None:
            self.prompt_slots = prompt_slots
        if decode_labels is not None:
            self.decode_labels = decode_labels

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
        # Final publication is a new, opt-in hook. Ordinary publication keeps
        # the legacy "all available labels" behavior when its map is absent.
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
    # of nodes on several workers (CFG parallel), the one whose cache admits a
    # request for all of them; the others hold at most what it does per request
    leader: str | None = None

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
class KVStep(ResourceStep):
    # write: bool # @nsagan: opting to remove this for now bc it's dead code
    commit: bool = True

    # e.g., for batched CFG
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict)
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()
    # (request_id, label) -> the retention a committing stream declares for
    # itself this step, applied at commit. It rides with the step rather than
    # living on the stream, so an offload has nothing to lose.
    retention: Mapping[tuple[str, str], RetentionPolicy] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class RingKVStep(ResourceStep):
    """One ring clock per request in the step's batch."""

    frames: tuple[tuple[str, int], ...]
