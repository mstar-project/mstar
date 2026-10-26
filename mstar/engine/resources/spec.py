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
from typing import TYPE_CHECKING, Any, ClassVar

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
    # splits it, so every process sees the instances, keyed ``<resource_key>@<node>``
    # (``instance_key``); the serving config's block still names the declared key.
    per_node: ClassVar[bool] = False

    def __post_init__(self):
        if not isinstance(self.nodes, set):
            self.nodes = set(self.nodes)

    def depends_on(self) -> set[str]:
        """Keys whose specs this one builds against; the engine resolves them
        into ``EngineResourceInfo.dependencies``."""
        return set()

    @property
    @abstractmethod
    def resource_class(self) -> "type[Resource]":
        """What builds this spec. Imported inside the property, so declaring a
        resource stays free of the manager and its kernels.

        The builder, not necessarily the class built: an attention spec names
        `AttentionManager`, whose `build` picks a backend subclass.
        """

    def for_node(self, node: str) -> "NodeResourceSpec | None":
        """A ``per_node`` spec's instance for ``node`` alone, or None when ``node`` needs
        none."""
        return dataclasses.replace(self, resource_key=instance_key(self.resource_key, node), nodes={node})

    def request_config(self, model_kwargs: dict | None) -> "ResourceReqConfig | None":
        """The config a request is opened with on this resource, built from its kwargs, or
        None. The conductor adds it to the model's own (``get_request_resource_configs``),
        so a resource the deployment declares needs no model code to take request
        settings."""
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
