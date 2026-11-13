"""What a model declares for compressed sparse attention (CSA): its config, spec and step.

Kept free of the manager so a submodule can declare a step without importing it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# the latent cache's aux planes (``PagedKVConfig.aux_planes``) that hold the index keys
INDEX_PLANES = "csa_index"


@dataclass
class CSAConfig:
    # the paged latent cache: a plane per compressed source, its index keys in
    # aux_planes[INDEX_PLANES]. A source that pools ``ratio`` tokens keeps a group's latent and
    # key in the row of the group's last token.
    kv_cache: str
    # the compressed rows a query attends: its top ``topk`` by the indexer's score
    topk: int
    # the softmax scale every attention runs at
    sm_scale: float
    # the indexer's key width, one INDEX_PLANES row
    index_head_dim: int
    # bytes of fp32 scores one selection launch holds, and of gathered rows one attention launch
    score_bytes: int = 256 << 20
    attend_bytes: int = 256 << 20


@dataclass
class CSASpec(NodeResourceSpec):
    config: CSAConfig

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.csa.manager import CSAManager

        return CSAManager

    def apply_yaml_overrides(self, score_bytes: int | None = None, attend_bytes: int | None = None):
        """What the box decides, not the model: the memory a launch's scores and rows may take."""
        for name, value in (("score_bytes", score_bytes), ("attend_bytes", attend_bytes)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            setattr(self.config, name, value)


@dataclass(frozen=True)
class CSAStep(ResourceStep):
    pass
