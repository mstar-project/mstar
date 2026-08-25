"""What a model declares about attention: its backend, its spec, its step.

Kept free of the managers and their kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class AttnBackend(Enum):
    FLASHINFER = "flashinfer"
    DENSE = "dense"


@dataclass
class AttentionConfig:
    kv_cache: str # name of the KV cache
    backend: AttnBackend = AttnBackend.FLASHINFER
    flashinfer_backend: str = "auto"


@dataclass
class AttentionSpec(NodeResourceSpec):
    config: AttentionConfig
    kv_config: KVConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.base import AttentionManager

        return AttentionManager

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        **kwargs,
    ):
        """Track the cache's geometry: the wrappers are planned against it, so
        a deployment that resizes the cache resizes these too."""
        del kwargs  # keys meant for other resources
        if max_num_pages is not None:
            self.kv_config.max_num_pages = max_num_pages
        if page_size is not None:
            self.kv_config.page_size = page_size


@dataclass
class CrossAttentionConfig:
    """Cross-attention against a context written once and never extended.

    ``kv_cache`` names the KV resource holding the encoder context;
    ``query_kv_cache`` names the decoder's KV resource, whose plan defines
    this step's query packing. They may be the same resource when the
    context shares the decoder's head config — the context then lives in it
    under its own ``context_label``. They differ when it does not, which is
    the usual case (an encoder's head count rarely matches the decoder's).

    ``query_kv_cache=None`` covers the query side having no KV cache at all
    (nothing is cached across steps on it): the packing then comes off the
    cross-attention step's own segments, one qo entry per segment in
    declared order.
    """
    kv_cache: str  # name of the KV cache holding the context
    query_kv_cache: str | None = None  # KV cache driving the queries, if any
    context_label: str = "context"
    backend: AttnBackend = AttnBackend.FLASHINFER
    flashinfer_backend: str = "auto"


@dataclass
class CrossAttentionSpec(NodeResourceSpec):
    config: CrossAttentionConfig
    # head config of the *context* cache, which need not match the decoder's
    kv_config: KVConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.cross import CrossAttentionManager

        return CrossAttentionManager

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        **kwargs,
    ):
        """Track the context cache's geometry; see ``AttentionSpec``."""
        del kwargs  # keys meant for other resources
        if max_num_pages is not None:
            self.kv_config.max_num_pages = max_num_pages
        if page_size is not None:
            self.kv_config.page_size = page_size


@dataclass(frozen=True)
class AttentionStep(ResourceStep):
    causal: bool = True
