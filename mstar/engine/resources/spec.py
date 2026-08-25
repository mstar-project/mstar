"""Resource declarations models hand to the engine.

The engine builds each node's resources once, at load time, from these
specs. A spec names what to build and its parameters; the model declares,
the engine constructs. The default declaration wraps a model's KV cache
configs unchanged, so a model only overrides it to add resources beyond
what those configs already describe.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class NodeResourceSpec(ABC):
    resource_key: str
    nodes: set[str]

    def __post_init__(self):
        if not isinstance(self.nodes, set):
            self.nodes = set(self.nodes)

    @property
    @abstractmethod
    def resource_class(self) -> "type[Resource]":
        """What builds this spec. Imported inside the property, so declaring a
        resource stays free of the manager and its kernels.

        The builder, not necessarily the class built: an attention spec names
        `AttentionManager`, whose `build` picks a backend subclass.
        """

    def apply_yaml_overrides(self, **kwargs):
        """Patch declared parameters from a deployment's YAML.

        The model declares shapes that suit the model; a deployment tunes what
        suits the box it runs on. Each spec takes the keys it recognizes and
        ignores the rest, since one YAML block reaches every resource.

        TODO: generalize. Matching YAML keys against the spec's (and its
        config's) dataclass fields would remove these per-resource
        implementations, at the cost of silently accepting anything named
        alike.
        """
        return


class ResourceReqConfig:
    """Per-request parameters for one resource, carried on the request and
    handed to that resource at ingest. Keyed by resource key, so it needs no
    tag of its own — a marker base, with no contract beyond the hook below."""

    def apply_conductor_config(self, **kwargs):
        return
