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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CombinedSplit:
    """A combined walk run on two kernels: decode for one-token rows, prefill
    for the rows of ``prefill_walks``, captured for as many as
    ``max_rows_in_combined`` lets in (an uncapped walk: the bucket's rows)."""
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
    # whether these replicas' workers open the resource's prefix cache: `opening_prefix_caches`
    # at their instance size, derived by the router rather than declared
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

    def __post_init__(self):
        if not isinstance(self.nodes, set):
            self.nodes = set(self.nodes)

    def depends_on(self) -> set[str]:
        """Keys whose specs this one builds against; the engine resolves them
        into ``EngineResourceInfo.dependencies``."""
        return set()

    @property
    def writes_prefix_snapshots(self) -> bool:
        """Fills the snapshot slots of the recurrent pool it depends on, so a hit can restore them."""
        return False

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
    """Index specs by resource key, checking uniqueness and ``depends_on``."""
    problems = spec_dependency_problems(specs)
    if problems:
        raise ValueError(problems[0])
    return {spec.resource_key: spec for spec in specs}


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
    context) can have each tuned on its own. An unknown key is an error: it
    would otherwise be a silently ineffective setting.
    """
    if "kv_cache" in model_config:
        raise ValueError(TOP_LEVEL_KV_CACHE)
    overrides = model_config.get("resources") or {}
    if not overrides:
        return
    by_key = resolve_spec_dependencies(specs)
    unknown = sorted(overrides.keys() - by_key.keys())
    if unknown:
        raise ValueError(unknown_overrides_message(unknown, by_key))
    for key, kwargs in overrides.items():
        by_key[key].apply_yaml_overrides(**kwargs)
    logger.info("Resource specs after YAML overrides: %s", specs)
