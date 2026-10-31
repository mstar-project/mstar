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
    # the paged MLA cache: the step's rows come from its plan; its pages hold the latents and,
    # in aux_planes[INDEX_PLANES], the index keys
    kv_cache: str
    # the keys a query attends once it sees more: its top ``topk`` by the indexer's score
    topk: int
    # tokens per pooled key (GLM-5.3's k-pool indexer); 1 scores every token's key
    kpool: int = 1
    # the absorbed MLA's softmax scale, for the sparse-attention plans the resource makes
    sm_scale: float | None = None
    # attend densely while no row sees more than topk keys (there, the same computation): such
    # a step plans no sparse attention
    dense_below_topk: bool = False


@dataclass
class DSASpec(NodeResourceSpec):
    config: DSAConfig

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.dsa.manager import DSAManager

        return DSAManager


@dataclass(frozen=True)
class DSAStep(ResourceStep):
    pass
