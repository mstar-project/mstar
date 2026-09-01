"""Resource declarations models hand to the engine.

The engine builds each node's resources once, at load time, from these
specs. A spec names what to build and its parameters; the model declares,
the engine constructs. The default declaration wraps a model's KV cache
configs unchanged, so a model only overrides it to add resources beyond
what those configs already describe.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

logger = logging.getLogger(__name__)


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
    @abstractmethod
    def resource_class(self) -> "type[Resource]":
        """What builds this spec. Imported inside the property, so declaring a
        resource stays free of the manager and its kernels.

        The builder, not necessarily the class built: an attention spec names
        `AttentionManager`, whose `build` picks a backend subclass.
        """

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
    tag of its own — a marker base, with no contract beyond the hook below."""

    def apply_conductor_config(self, **kwargs):
        return


def resolve_spec_dependencies(
    specs: Sequence[NodeResourceSpec],
) -> dict[str, NodeResourceSpec]:
    """Index specs by resource key, checking uniqueness and ``depends_on``."""
    by_key: dict[str, NodeResourceSpec] = {}
    for spec in specs:
        if spec.resource_key in by_key:
            raise ValueError(
                f"two resources declared under the key {spec.resource_key!r}"
            )
        by_key[spec.resource_key] = spec
    for spec in specs:
        missing = sorted(spec.depends_on() - by_key.keys())
        if missing:
            raise ValueError(
                f"resource {spec.resource_key!r} depends on {missing}, which "
                f"this model does not declare; available: {sorted(by_key)}"
            )
    return by_key


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
        raise ValueError(
            "top-level `kv_cache:` in the serving config is no longer read; "
            "move it under `resources:` keyed by resource name, e.g.\n"
            "resources:\n  kv_cache:\n    max_num_pages: 1024"
        )
    overrides = model_config.get("resources") or {}
    if not overrides:
        return
    by_key = resolve_spec_dependencies(specs)
    unknown = sorted(overrides.keys() - by_key.keys())
    if unknown:
        raise ValueError(
            f"serving config overrides unknown resource(s) {unknown}; this "
            f"model declares {sorted(by_key)}"
        )
    for key, kwargs in overrides.items():
        by_key[key].apply_yaml_overrides(**kwargs)
    logger.info("Resource specs after YAML overrides: %s", specs)
