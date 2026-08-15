"""Resource declarations models hand to the engine.

The engine builds each node's resources once, at load time, from these
specs. A spec names what to build and its parameters; the model declares,
the engine constructs. The default declaration wraps a model's KV cache
configs unchanged, so a model only overrides it to add resources beyond
what those configs already describe.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

import torch


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



class ResourceReqConfig(ABC):
    @property
    @abstractmethod
    def resource_type(self) -> ResourceType:
        pass

    @abstractmethod
    def apply_conductor_config(self, **kwargs):
        pass

