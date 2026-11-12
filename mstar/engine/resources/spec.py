"""Resource declarations models hand to the engine.

The engine builds each node's resources once, at load time, from these
specs. A spec names what to build and its parameters; the model declares,
the engine constructs. The default declaration wraps a model's KV cache
configs unchanged, so a model only overrides it to add resources beyond
what those configs already describe.
"""

import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
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
    # Built only when the serving config's ``resources:`` block names this spec's key
    # (``mstar.model.base.serving_resources``): a resource a deployment turns on, such as
    # per-request LoRA adapters, declared beside the model's other resources.
    opt_in: ClassVar[bool] = False
    # Built as one instance per node (``for_node``) rather than one shared by its nodes:
    # for a resource whose nodes share no state and may sit in different lockstep
    # instances (a vision tower beside a tensor-parallel LLM). ``serving_resources``
    # splits it, so every process that builds or validates requests (the conductor and
    # workers) sees the instances, keyed ``<resource_key>@<node>`` (``instance_key``); the
    # serving config's block still names the declared key.
    per_node: ClassVar[bool] = False

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
        whole state: a recurrent pool). None for one holding none. A resource
        over pools that are all ``copied`` shards its heads over TP (``EngineResourceInfo.head_shards``),
        and a node whose pools store by position runs copied unless its shard's exchange splits it.
        From raw fields, as the shard's hooks."""
        return None

    def resolve_exchange(self, specs: Mapping[str, "NodeResourceSpec"]) -> "SpExchange":
        """How a split step's tokens reach this resource (``sequence_shard``): ``none`` (the
        default) for a resource that takes none. From raw fields, so the sequence shard's
        ``derive`` can ask in any order; a node's steps split where its shard's ``node_exchange``
        isn't ``none`` and any of its resources answers otherwise (``SeqShardSpec.splits``).

        Every hook the sequence shard reads is computed from the spec's own raw fields and other
        specs' hooks, never from a field ``derive`` sets: specs derive in key order, so such a
        field may still hold its default when the shard derives."""
        return "none"

    def splits_eagerly(self, specs: Mapping[str, "NodeResourceSpec"]) -> bool:
        """Whether this resource's split steps run eagerly only, their plans differing by rank
        where a graph holds one: a node with one captures no step that may split
        (``SeqShardSpec.eager``). False (the default) for one whose split steps replay in a graph.
        From raw fields, as ``resolve_exchange``."""
        return False

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
    def for_node(self, node: str) -> "NodeResourceSpec | None":
        """A ``per_node`` spec's instance for ``node`` alone, or None when ``node`` needs
        none."""
        return dataclasses.replace(self, resource_key=instance_key(self.resource_key, node), nodes={node})

    def request_config(
        self, model_kwargs: dict | None, current: "ResourceReqConfig | None" = None,
    ) -> "ResourceReqConfig | None":
        """The config a request is opened with on this resource, built from its kwargs, or
        None to keep ``current``, the config the model built for it (``get_request_resource_configs``)
        or None. The conductor puts it in place of ``current``, so a resource needs no model
        code to take request settings, even where the model builds the config."""
        return None

    def capacity_problem(
        self, requests: int, request: "ResourceReqConfig | None", axis_sizes: Sequence[Mapping[str, int]] = (),
    ) -> str | None:
        """Why this resource can't hold ``requests`` requests at once, each opened with ``request``, or None.
        ``axis_sizes`` holds the size on every mesh axis of each node group that runs its nodes. None by default: a
        resource that requests don't size holds any number."""
        return None

    def prefix_variant(self, model_kwargs: dict) -> bytes | None:
        """What sets a request's KV on this resource apart from another's for the same
        tokens (a LongRoPE table picked per request), from the kwargs ``request_config``
        reads; None (the default) for nothing. The API server's data worker folds it into
        the request's prefix-cache root, so the two never share pages. Called per request
        on the data worker's threads, so it reads only ``model_kwargs`` and the spec."""
        return None

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


def instance_key(resource_key: str, node: str) -> str:
    """The key of a ``per_node`` spec's instance for ``node``: the same on every rank,
    since it names the node rather than the worker."""
    return f"{resource_key}@{node}"


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


def check_sizes(sizes: Mapping[str, Any], smallest: int = 1) -> None:
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < smallest:
            raise ValueError(f"{name} must be an integer >= {smallest}; got {value!r}")


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
        try:
            by_key[key].apply_yaml_overrides(**kwargs)
        except TypeError as e:
            # a key the resource doesn't take: a bad config, as every other problem here is
            raise ValueError(f"resources.{key}: {e}") from e
    _derive(by_key)
    if overrides:
        logger.info("Resource specs after YAML overrides: %s", specs)
