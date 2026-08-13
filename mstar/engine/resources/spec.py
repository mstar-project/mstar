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
    SCRATCH_KV = 1
    SAMPLER = 2
    ATTENTION = 3


@dataclass(frozen=True)
class ScratchKVSpec:
    """A fixed-shape scratch cache: overwritten every step, slot-indexed
    by batch position, no per-request lifetime. A ``dtype`` of None means
    the engine's KV cache dtype."""
    shape: tuple[int, ...]
    dtype: "torch.dtype | None" = None


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


@dataclass
class ScratchKVSpec(NodeResourceSpec):
    config: ScratchKVSpec

    @property
    def resource_type(self):
        return ResourceType.SCRATCH_KV



class ResourceReqConfig(ABC):
    @property
    @abstractmethod
    def resource_type(self) -> ResourceType:
        pass

    @abstractmethod
    def apply_conductor_config(self, **kwargs):
        pass



@dataclass
class KVReqConfig(ResourceReqConfig):
    # NOTE: this may need to be refined
    needed_labels: list[str] | None = None
    needed_labels_per_node: dict[str, list[str]] = field(default_factory=dict)
    needed_labels_per_node_walk: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    @property
    def resource_type(self):
        return ResourceType.KV_CACHE

    def get_labels(self, node: str, walk: str):
        if (node, walk) in self.needed_labels_per_node_walk:
            return self.needed_labels_per_node_walk[(node, walk)]
        if node in self.needed_labels_per_node:
            return self.needed_labels_per_node[node]
        if self.needed_labels is not None:
            return self.needed_labels
        return ["main"]

