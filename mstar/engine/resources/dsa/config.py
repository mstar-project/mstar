"""What a model declares for DeepSeek sparse attention (DSA): its config, spec and step.

Kept free of the manager so a submodule can declare a step without importing it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class DSAConfig:
    # the paged MLA cache: a step's rows come from its plan, the attended latents from its pages
    kv_cache: str
    # the keys a query attends once it sees more: its top ``topk`` by the indexer's score
    topk: int


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
