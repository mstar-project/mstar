"""Resource declarations models hand to the engine.

The engine builds each node's resources once, at load time, from these
specs. A spec names what to build and its parameters; the model declares,
the engine constructs. The default declaration wraps a model's KV cache
configs unchanged, so a model only overrides it to add resources beyond
what those configs already describe.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum


class ResourceType(IntEnum):
    KV_CACHE = 0
    SAMPLER = 1
    ATTENTION = 2
    CROSS_ATTENTION = 3
    POSITIONS = 4


@dataclass
class NodeResourceSpec(ABC):
    label: str
    nodes: set[str]

    def __post_init__(self):
        if not isinstance(self.nodes, set):
            self.nodes = set(self.nodes)

    @property
    @abstractmethod
    def resource_type(self) -> ResourceType:
        pass

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



class ResourceReqConfig(ABC):
    @property
    @abstractmethod
    def resource_type(self) -> ResourceType:
        pass

    def apply_conductor_config(self, **kwargs):
        return

