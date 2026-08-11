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

from mstar.engine.kv_store import KVCacheConfig
from mstar.utils.sampling import SamplingConfig


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


@dataclass
class SamplerSpec(NodeResourceSpec):
    config: SamplingConfig

    @property
    def resource_type(self):
        return ResourceType.SAMPLER



class ResourceReqConfig(ABC):
    @property
    @abstractmethod
    def resource_type(self) -> ResourceType:
        pass

    @abstractmethod
    def apply_conductor_config(self, **kwargs):
        pass


@dataclass
class SamplingReqConfig(ResourceReqConfig):
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 1
    ignore_eos: bool = False # used for benchmark parity
    repetition_penalty: float = 1
    _seed: int = 0 # set by the conductor

    @property
    def resource_type(self):
        return ResourceType.SAMPLER

    def apply_conductor_config(
        self, seed: int=0,
        **kwargs
    ):
        self._seed = seed

    @property
    def seed(self):
        return self._seed

