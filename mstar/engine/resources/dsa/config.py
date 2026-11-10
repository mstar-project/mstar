"""What a model declares for DeepSeek sparse attention (DSA): its config, spec and step.

Kept free of the manager so a submodule can declare a step without importing it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# the MLA cache's aux planes (``PagedKVConfig.aux_planes``) that hold the index keys
INDEX_PLANES = "dsa_index"


@dataclass
class DSAConfig:
    # the paged MLA cache: a step's rows come from its plan, the attended latents from its pages
    kv_cache: str
    # the keys a query attends once it sees more: its top ``topk`` by the indexer's score
    topk: int
    # tokens per pooled key (GLM-5.3's k-pool indexer: paged.py); 1 scores every token's key
    kpool: int = 1
    # the absorbed MLA's softmax scale, for the sparse-attention plans the resource makes
    sm_scale: float | None = None
    # kpool > 1: the indexer's key width; an INDEX_PLANES row is three of them wide
    index_head_dim: int | None = None
    # MiB each FlashInfer workspace of its sparse plans starts at (one per walk and capture
    # slot, one for eager steps); a plan that needs more grows it until captures are sealed
    workspace_mb: int = 128


@dataclass
class DSASpec(NodeResourceSpec):
    config: DSAConfig

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    @property
    def resource_class(self) -> "type[Resource]":
        if self.config.kpool > 1:
            from mstar.engine.resources.dsa.manager import KPoolDSA

            return KPoolDSA
        from mstar.engine.resources.dsa.manager import DSAManager

        return DSAManager


@dataclass(frozen=True)
class DSAStep(ResourceStep):
    pass
