"""What a model declares for DeepSeek sparse attention (DSA): its config, spec and step.

Kept free of the manager so a submodule can declare a step without importing it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from mstar.engine.resources.sizing import PoolConfig
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class DSAConfig(PoolConfig):
    """The index keys grow with each request rather than drawing from a pool, so
    sizing counts what a request holds and allocates nothing for them."""

    pool_unit: ClassVar[str] = "token"
    size_key: ClassVar[str] = ""

    # the paged MLA cache: a step's rows come from its plan, the attended latents from its pages
    kv_cache: str
    # the keys a query attends once it sees more: its top ``topk`` by the indexer's score
    topk: int
    # the indexer layers that store keys (FULL ones), a key's width and dtype (None: the
    # engine's KV dtype), so each token's keys can be counted; 0 layers leaves them out
    key_layers: int = 0
    key_dim: int = 0
    key_dtype: torch.dtype | None = None

    @property
    def pool_units(self) -> None:
        return None

    def unit_bytes(self, kv_dtype: torch.dtype, num_shards: int = 1) -> int:
        """One token's keys in every FULL layer, replicated on every rank."""
        del num_shards
        return self.key_layers * self.key_dim * (self.key_dtype or kv_dtype).itemsize

    def units_per_request(self, max_seq_len: int | None) -> int:
        return max_seq_len or 0


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
