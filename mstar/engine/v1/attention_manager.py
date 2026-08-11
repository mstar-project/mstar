

from dataclasses import dataclass
from enum import Enum

import torch

from mstar.engine.resources.base import Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceType
from mstar.engine.v1.kv_manager import KVManager


class AttnBackend(Enum):
    FLASHINFER = "flashinfer"

@dataclass
class AttentionConfig:
    backend: AttnBackend=AttnBackend.FLASHINFER
    kv_cache: str # name of the KV cache

@dataclass
class AttentionSpec(NodeResourceSpec):
    config: AttentionConfig

    @property
    def resource_type(self):
        return ResourceType.ATTENTION


class AttentionManager(Resource):
    # Remains abstract except for build; will build based
    # on the attention backend

    @classmethod
    def build(
        cls, spec: AttentionSpec,
        device: torch.device,
        # I don't think I love giving the attention manager a reference
        # to the kv manager; just trying this out and it can be refined
        kv_managers: dict[str, KVManager],
        **engine_kwargs
    ):
        if spec.config.backend == AttnBackend.FLASHINFER:
            return FlashInferManager(
                kv_manager=kv_managers[spec.config.kv_cache]
            )
        


class FlashInferManager(AttentionManager):
    def __init__(
        self,
        kv_manager: KVManager
    ):
        self.kv_manager = kv_manager
        # TODO: something to think about is that we need an eager and
        # cuda graphed version of some of these resources, and sometimes
        # the cudagraphed version interacts with the eager version (e.g.,
        # the cuda graphed sampler uses the eager version for storage of
        # the repetition penalty buffer). I think we need an abstraction
        # for this; maybe a resource can have the option to instantiate
        # a cuda graphable version of itself.
        