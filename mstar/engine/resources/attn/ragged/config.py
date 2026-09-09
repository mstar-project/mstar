"""What a model declares about cacheless (ragged) attention.

Kept free of the manager and its kernels, like the other resources' configs, so
a submodule can declare a step without pulling FlashInfer in behind it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class RaggedAttentionConfig:
    """Varlen self-attention over segments packed into one forward, with no KV
    cache: the whole layout is this step's, and nothing carries to the next.

    Head counts are **pre-sharding**; the engine narrows them to the rank's
    slice at build, as it does for a ``KVConfig``.
    """

    num_qo_heads: int
    num_kv_heads: int
    head_dim: int

    # Defaults to head_dim ** -0.5 on the TRUE head dim; FlashInfer's own
    # default would derive it from the padded one (see `padded_head_dim`).
    sm_scale: float | None = None

    # Per-request ceiling sizing a CUDA-graph bucket: a capture at batch size
    # `bs` gets `bs` times this. A "segment" is an independently-attending
    # span, not a request — a request carrying several images contributes
    # several.
    max_segments_per_request: int = 1
    # Only for a runner that buckets by batch size alone; elsewhere the
    # bucket's own token count is the ceiling and this stays None.
    max_tokens_per_request: int | None = None

    flashinfer_backend: str = "auto"

    def __post_init__(self):
        if self.sm_scale is None:
            self.sm_scale = self.head_dim ** -0.5
        self._unsharded_qo_heads = self.num_qo_heads
        self._unsharded_kv_heads = self.num_kv_heads

    def shard(self, num_shards: int) -> None:
        """Narrow the head counts to one rank's slice; see ``KVConfig.shard``.

        Idempotent, so a rebuild (or a second manager over one config) is free.
        """
        from mstar.distributed.utils import divide

        if num_shards >= self._unsharded_kv_heads:
            # fewer KV heads than ranks — every rank replicates one
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = divide(self._unsharded_kv_heads, num_shards)
        self.num_qo_heads = divide(self._unsharded_qo_heads, num_shards)

    def max_segments_for(self, bs: int) -> int:
        return bs * self.max_segments_per_request

    def max_tokens_for(self, bs: int) -> int | None:
        if self.max_tokens_per_request is None:
            return None
        return bs * self.max_tokens_per_request


@dataclass
class RaggedAttentionSpec(NodeResourceSpec):
    config: RaggedAttentionConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.ragged.base import RaggedAttnManager

        return RaggedAttnManager

    def apply_yaml_overrides(
        self,
        flashinfer_backend: str | None = None,
        max_segments_per_request: int | None = None,
        max_tokens_per_request: int | None = None,
    ):
        """Which kernel to run is the deployment's call as much as the model's —
        an image that cannot build FA3 pins FA2 here.

        The two ceilings are here rather than on the model because they size
        CUDA-graph buckets, which is a deployment's memory/coverage trade.
        """
        if flashinfer_backend is not None:
            self.config.flashinfer_backend = flashinfer_backend
        if max_segments_per_request is not None:
            self.config.max_segments_per_request = max_segments_per_request
        if max_tokens_per_request is not None:
            self.config.max_tokens_per_request = max_tokens_per_request
