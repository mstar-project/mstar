"""Resource declarations models hand to the engine.

The engine builds each node's resources once, at load time, from these
specs. A spec names what to build and its parameters; the model declares,
the engine constructs. The default declaration wraps a model's KV cache
configs unchanged, so a model only overrides it to add resources beyond
what those configs already describe.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, Protocol

if TYPE_CHECKING:
    from mstar.engine.resources.attn.config import SpExchange
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.sequence_shard import SeqShardSpec

logger = logging.getLogger(__name__)

# how a resource stores its per-request state over a node's ranks (`NodeResourceSpec.storage`)
Storage = Literal["by_head", "by_position", "copied"]


@dataclass(frozen=True)
class CombinedSplit:
    """A combined walk run on two kernels: decode for one-token rows, prefill
    for the rows of ``prefill_walks``, captured for as many as
    ``max_rows_in_combined`` lets in (an uncapped walk: the bucket's rows).

    The engine orders a step of ``combined_walk`` with the rows of
    ``prefill_walks`` last, so both kernels read the step's tokens in place:
    the decode kernel the leading run of one-token rows, the prefill kernel
    the rows after it (under capture every token, the run as one leading
    sequence). Every walk of ``combined_walk`` outside ``prefill_walks`` must
    be one token per row: its rows go first, and a multi-token one there
    ends the run, so it and every decode row after it count as prompt rows,
    and a captured step's plan raises if its prompt rows then exceed
    ``max_prefill_rows``."""
    combined_walk: str
    prefill_walks: tuple[str, ...]

    def max_prefill_rows(
        self, max_rows_in_combined: Mapping[str, "int | None"], bucket_rows: int,
    ) -> int:
        total = 0
        for walk in self.prefill_walks:
            cap = max_rows_in_combined.get(walk)
            total += bucket_rows if cap is None else cap
        return min(total, bucket_rows)


class RoutingContext(NamedTuple):
    """The data-parallel replicas a routing mirror is built for, and what a
    mirror reads of the model's KV declarations, which ``ReplicaRouter`` fills."""
    num_replicas: int
    # the walks of the worker graphs that hold the resource's nodes on these replicas
    walks: frozenset[str]
    # the resource's keyed labels, each with the walks whose writes it keys
    keyed: Mapping[str, Collection[str]]
    # whether every replica's engine opens the resource's prefix cache (`prefix_caches_by_rank`),
    # derived by the router rather than declared
    prefix_cache_opens: bool


class RoutingFacts(NamedTuple):
    """What a mirror predicts for a new request on one replica, by name, for
    the conductor's ``routing_cost`` to price. Every field is a count whose
    identity is 0: a mirror leaves the fields it does not report at 0, and the
    router adds the rows of every mirror on the replica field by field before
    pricing them once."""
    # the positions the request's prompt writes, every KV label's
    work: int = 0
    # of those, the positions a KV prefix cache is predicted to hold
    kv_hit: int = 0


class RoutingMirror(Protocol):
    """A conductor-side prediction of one resource's state on each data-parallel
    replica, for choosing a new request's replica. It holds bookkeeping only,
    never the resource's data, follows the resource's own rules on what the
    conductor sees, and a wrong prediction makes a request slower, never wrong.

    ``config`` is the request's config for the resource, as the conductor
    opens it with: built and validated before the pick.
    """

    def facts(self, config: "ResourceReqConfig") -> list[RoutingFacts]:
        """The request's facts on each replica."""

    def add_request(self, config: "ResourceReqConfig", replica: int) -> Any:
        """Record the request as placed on ``replica``; the handle comes back to ``remove_request``."""

    def remove_request(self, handle: Any, ran: bool) -> None:
        """Release what ``add_request`` recorded, once the request is gone.
        ``ran`` says whether a pass came back for one of the request's
        partitions with a walk on the resource's node groups (not necessarily a
        walk that uses the resource), so the resource may hold what it left."""


@dataclass
class NodeResourceSpec(ABC):
    resource_key: str
    nodes: set[str]
    # how a split step reaches this resource where it has more than one way (``sp_exchanges``): a
    # deployment's choice, its ``resources:`` block's ``sp_exchange`` (`apply_yaml_overrides`); None
    # takes the first wherever its node splits (`own_exchange`)
    sp_exchange: str | None = field(default=None, init=False, repr=False)

    # the ways a split step may reach this resource that a deployment picks among, the default first;
    # none for one whose way follows from its storage or its model (an attention's, a ragged one's)
    sp_exchanges: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        if not isinstance(self.nodes, set):
            self.nodes = set(self.nodes)

    def depends_on(self) -> set[str]:
        """Keys whose specs this one builds against; the engine resolves them
        into ``EngineResourceInfo.dependencies``."""
        return set()

    def combined_splits(self) -> "tuple[CombinedSplit, ...]":
        """The combined walks this resource runs on split kernels."""
        return ()

    @property
    def writes_prefix_snapshots(self) -> bool:
        """Fills the snapshot slots of the recurrent pool it depends on, so a hit can restore them."""
        return False

    def keeps_prefix_cache(self, world_size: int, checkpoint_named: bool) -> bool:
        """Whether the resource keeps a prefix cache of its own on a lockstep
        instance of ``world_size`` ranks, for a model that names its checkpoint
        or not. False (the default) keeps nothing across requests.
        `opening_prefix_caches` weighs it against the resources on its nodes."""
        return False

    @property
    def restores_prefix(self) -> bool:
        """A hit needs state the resource holds from the skipped tokens, so it
        serves one only from a prefix cache of its own, or, keeping none, from
        those of the resources it depends on that restore."""
        return False

    @property
    def storage(self) -> "Storage | None":
        """How this resource stores its per-request state over a node's ranks: ``by_head`` (each
        rank a slice of the heads, over TP x SP), ``by_position`` (each a share of every page's
        token positions over ranks that would otherwise hold the same heads, heads over TP:
        ``mstar.engine.resources.context_parallel``), or ``copied`` (heads over TP, every SP rank the
        whole state: a recurrent pool, MLA's one latent). None for one holding none. A resource
        over pools that are all ``copied`` shards its heads over TP (``EngineResourceInfo.head_shards``),
        and a node whose pools store by position runs copied unless its shard's exchange splits it.
        From raw fields, as the shard's hooks."""
        return None

    def resolve_exchange(self, specs: Mapping[str, "NodeResourceSpec"]) -> "SpExchange":
        """How a split step's tokens reach this attention (``sequence_shard``): ``none`` (the
        default) for a resource that takes none. From raw fields, so the sequence shard's
        ``derive`` can ask in any order; a node's steps split where any of its attentions answers
        otherwise (``SeqShardSpec.splits``).

        Every hook the sequence shard reads is computed from the spec's own raw fields and other
        specs' hooks, never from a field ``derive`` sets: specs derive in key order, so such a
        field may still hold its default when the shard derives."""
        return "none"

    def own_exchange(self, specs: Mapping[str, "NodeResourceSpec"]) -> str:
        """For a resource with ``sp_exchanges``: ``none`` where its node's sequence shard never
        splits (``shard_exchange``), which ``sp_exchange`` doesn't change, and otherwise
        ``sp_exchange`` or the first of them. From raw fields, as ``resolve_exchange``."""
        from mstar.engine.resources.sequence_shard import shard_exchange

        if shard_exchange(self, specs) == "none":
            return "none"
        return self.sp_exchange or self.sp_exchanges[0]

    def captures_split_steps(self, specs: Mapping[str, "NodeResourceSpec"]) -> bool:
        """Whether this resource runs a split step inside an accelerator graph. False for one whose
        split exchange plans per rank or syncs the host; a node with any such resource plans every
        leased step copied (``SeqShardSpec.eager_splits``). From raw fields, as ``resolve_exchange``."""
        return True

    def split_step_refusals(
        self, node: str, shard: "SeqShardSpec", tp: int, sp: int, specs: Mapping[str, "NodeResourceSpec"],
    ) -> list[str]:
        """What this resource can't run on ``node``, whose steps ``shard`` splits over ``sp`` SP
        ranks (``tp`` TP ranks each), among ``specs`` (every spec, by key); each a sentence."""
        return []

    @property
    @abstractmethod
    def resource_class(self) -> "type[Resource]":
        """What builds this spec. Imported inside the property, so declaring a
        resource stays free of the manager and its kernels.

        The builder, not necessarily the class built: an attention spec names
        `AttentionManager`, whose `build` picks a backend subclass.
        """

    def routing_mirror(self, ctx: RoutingContext) -> RoutingMirror | None:
        """A mirror of this resource for the replicas ``ctx`` describes; None
        (the default) leaves the resource out of the replica pick there."""
        return None

    def derive(self, specs: Mapping[str, "NodeResourceSpec"]) -> list[str]:
        """Set what this spec takes from the specs that read it (a KV pool keeps the window its
        attentions declare), after a deployment's overrides; return what is wrong with it.
        Idempotent: it runs wherever the specs resolve (`resolve_spec_dependencies`)."""
        return []

    def apply_yaml_overrides(self, **kwargs):
        """Patch declared parameters from this resource's YAML block.

        The model declares shapes that suit the model; a deployment tunes what
        suits the box it runs on. The block is scoped to this spec's
        ``resource_key`` (see ``apply_yaml_overrides`` below), so an
        unrecognized key here is a typo, not another resource's setting —
        subclasses name exactly what they accept and let the rest raise.

        TODO: generalize. Matching YAML keys against the spec's (and its
        config's) dataclass fields would remove these per-resource
        implementations, at the cost of silently accepting anything named
        alike.
        """
        if kwargs:
            raise TypeError(
                f"resource {self.resource_key!r} takes no YAML overrides; got "
                f"{sorted(kwargs)}"
            )


class ResourceReqConfig:
    """Per-request parameters for one resource, carried on the request and
    handed to that resource at ingest. Keyed by resource key, so it needs no
    tag of its own — a marker base, with no contract beyond the hooks below."""

    def apply_conductor_config(self, **kwargs):
        return

    def loop_room(self, spec: "NodeResourceSpec | None" = None) -> dict[str, int]:
        """The most iterations each loop can run before the request outgrows the
        resource built from ``spec``. A loop the resource doesn't bound is left out."""
        return {}

    def validate(self, spec: "NodeResourceSpec | None" = None) -> None:
        """Raise ``ValueError`` if a client-supplied value is out of range, or
        one the resource built from ``spec`` cannot honour.

        The conductor calls it at admission, so a bad request is a 400 before
        any worker sees it.
        """
        return


def spec_dependency_problems(specs: Sequence[NodeResourceSpec]) -> list[str]:
    """Every duplicate resource key and undeclared ``depends_on`` in ``specs``."""
    problems: list[str] = []
    keys: set[str] = set()
    for spec in specs:
        if spec.resource_key in keys:
            problems.append(
                f"two resources declared under the key {spec.resource_key!r}"
            )
        keys.add(spec.resource_key)
    for spec in specs:
        missing = sorted(spec.depends_on() - keys)
        if missing:
            problems.append(
                f"resource {spec.resource_key!r} depends on {missing}, which "
                f"this model does not declare; available: {sorted(keys)}"
            )
    return problems


def resolve_spec_dependencies(
    specs: Sequence[NodeResourceSpec],
) -> dict[str, NodeResourceSpec]:
    """Index specs by resource key, checking uniqueness and ``depends_on``, and derive what each
    takes from the others (`NodeResourceSpec.derive`)."""
    by_key = _indexed(specs)
    _derive(by_key)
    return by_key


def _indexed(specs: Sequence[NodeResourceSpec]) -> dict[str, NodeResourceSpec]:
    problems = spec_dependency_problems(specs)
    if problems:
        raise ValueError(problems[0])
    return {spec.resource_key: spec for spec in specs}


def derive_specs(by_key: Mapping[str, NodeResourceSpec]) -> list[tuple[str, str]]:
    """Run every spec's `NodeResourceSpec.derive`; each problem as (resource key, message)."""
    return [(key, message) for key, spec in by_key.items() for message in spec.derive(by_key)]


def _derive(by_key: Mapping[str, NodeResourceSpec]) -> None:
    problems = derive_specs(by_key)
    if problems:
        key, message = problems[0]
        raise ValueError(f"resource {key!r}: {message}")


TOP_LEVEL_KV_CACHE = (
    "top-level `kv_cache:` in the serving config is no longer read; "
    "move it under `resources:` keyed by resource name, e.g.\n"
    "resources:\n  kv_cache:\n    max_num_pages: 1024"
)


def unknown_overrides_message(unknown: list[str], declared: Collection[str]) -> str:
    return (
        f"serving config overrides unknown resource(s) {unknown}; this "
        f"model declares {sorted(declared)}"
    )


def apply_yaml_overrides(
    specs: Sequence[NodeResourceSpec], model_config: Mapping[str, Any],
) -> None:
    """Apply a deployment's ``resources:`` block to the specs it names.

    ``resources: {<resource_key>: {...}}``, one block per resource, so a model
    with two pools of the same kind (whisper's decoder cache and its encoder
    context) can have each tuned on its own. An unknown key is a ValueError: it
    would otherwise be a silently ineffective setting.

    Then derives what the specs derive (`resolve_spec_dependencies`), with or without
    overrides and after them, so a deployment's page size or backend is what the derivation
    checks, as validation checks it.
    """
    if "kv_cache" in model_config:
        raise ValueError(TOP_LEVEL_KV_CACHE)
    by_key = _indexed(specs)
    overrides = model_config.get("resources") or {}
    unknown = sorted(overrides.keys() - by_key.keys())
    if unknown:
        raise ValueError(unknown_overrides_message(unknown, by_key))
    for key, kwargs in overrides.items():
        spec = by_key[key]
        if isinstance(kwargs, Mapping) and "sp_exchange" in kwargs:
            kwargs = dict(kwargs)
            choice = kwargs.pop("sp_exchange")
            if not spec.sp_exchanges:
                raise ValueError(f"resources.{key}: resource {key!r} takes no sp_exchange; got {choice!r}")
            if choice not in spec.sp_exchanges:
                raise ValueError(
                    f"resources.{key}: sp_exchange must be one of {', '.join(spec.sp_exchanges)}; got {choice!r}"
                )
            spec.sp_exchange = choice
        try:
            spec.apply_yaml_overrides(**kwargs)
        except TypeError as e:
            # a key the resource doesn't take: a bad config, as every other problem here is
            raise ValueError(f"resources.{key}: {e}") from e
    _derive(by_key)
    if overrides:
        logger.info("Resource specs after YAML overrides: %s", specs)
