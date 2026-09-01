"""What a model declares about attention: its backend, its spec, its step.

Kept free of the managers and their kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

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

    def depends_on(self) -> set[str]:
        # the wrappers are planned against the cache's geometry, and it has to
        # be the same KVConfig the KV resource holds
        return {self.config.kv_cache}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.base import AttentionManager

        return AttentionManager

    def apply_yaml_overrides(
        self,
        backend: str | AttnBackend | None = None,
        flashinfer_backend: str | None = None,
    ):
        """Which kernel to run is the deployment's call as much as the
        model's — an image that cannot build FA3 pins FA2 here.

        Cache geometry is not repeated here: it belongs to the KV resource
        this spec depends on, and is tuned under that resource's own block.
        """
        if backend is not None:
            self.config.backend = AttnBackend(backend)
        if flashinfer_backend is not None:
            self.config.flashinfer_backend = flashinfer_backend


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

    def depends_on(self) -> set[str]:
        # the context cache, whose head config need not match the decoder's
        keys = {self.config.kv_cache}
        if self.config.query_kv_cache is not None:
            keys.add(self.config.query_kv_cache)
        return keys

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.cross import CrossAttentionManager

        return CrossAttentionManager

    def apply_yaml_overrides(
        self,
        backend: str | AttnBackend | None = None,
        flashinfer_backend: str | None = None,
    ):
        """Which kernel to run against the context cache; see ``AttentionSpec``."""
        if backend is not None:
            self.config.backend = AttnBackend(backend)
        if flashinfer_backend is not None:
            self.config.flashinfer_backend = flashinfer_backend


@dataclass(frozen=True)
class AttentionStep(ResourceStep):
    causal: bool = True
