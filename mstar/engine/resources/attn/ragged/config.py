"""What a model declares about cacheless (ragged) attention.

Kept free of the manager and its kernels, like the other resources' configs, so
a submodule can declare a step without pulling FlashInfer in behind it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


def cross_label(q_label: str, kv_label: str) -> str:
    """The plan a cross-attention pair runs under: ``q_label``'s spans attending
    ``kv_label``'s."""
    return f"{q_label}<-{kv_label}"


@dataclass(frozen=True)
class RaggedCrossAttentionStep(ResourceStep):
    """A ragged cross-attention step: which spans attend which.

    Each label in ``segments`` is one span per request (video tokens, text tokens,
    ...). Each ``(q_label, kv_label)`` in ``pairs`` is one attention: every request's
    ``q_label`` span attends that request's ``kv_label`` span, so both labels must be
    declared for the same requests in the same order. Never causal.
    """

    pairs: tuple[tuple[str, str], ...] = ()


@dataclass
class RaggedAttentionConfig:
    """Varlen attention over segments packed into one forward, with no KV
    cache: the whole layout is this step's, and nothing carries to the next.
    The config of both ``RaggedAttentionSpec`` (self-attention within each span)
    and ``RaggedCrossAttentionSpec`` (one span attending another).

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

    # Activation dtype of the q / k / v the node hands the kernel (and of its
    # output). A cacheless attention doesn't need to adhere to the dtype of its
    # KV cache, so it is free to declare a separate dtype here; when None, the
    # engine's KV dtype (which is actually a misnamed autocast dtype) is used.
    dtype: torch.dtype | None = None

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
class _RaggedSpecBase(NodeResourceSpec):
    """The config and deployment overrides both ragged specs share."""

    config: RaggedAttentionConfig

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


@dataclass
class RaggedAttentionSpec(_RaggedSpecBase):
    """Cacheless self-attention within each declared span (``AttentionStep``)."""

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.ragged.base import RaggedAttnManager

        return RaggedAttnManager


@dataclass
class RaggedCrossAttentionSpec(_RaggedSpecBase):
    """Cacheless cross-attention between two spans of each request: a forward's
    ``(q_label, kv_label)`` pairs (``RaggedCrossAttentionStep``), each run through
    ``RaggedAttentionCallable(resource, cross_label(q_label, kv_label))``.

    Separate from ``RaggedAttentionSpec`` so a resource is one kind of attention:
    a model with both declares one of each per head geometry.
    """

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.ragged.base import RaggedCrossAttnManager

        return RaggedCrossAttnManager


@dataclass
class RaggedBlockCausalAttentionSpec(_RaggedSpecBase):
    """Cacheless block-causal self-attention within each declared span: a token in
    block ``b`` of its span (``block_size`` tokens a block, from the span's start)
    attends every key of blocks ``0..b``. Declared with ``AttentionStep(causal=False)``,
    since the block structure is the whole mask.

    Its own kind, not a flag on ``RaggedAttentionSpec``: it runs a different kernel
    (paged prefill over one-token pages, see ``attn.ragged.block_causal``).
    """

    block_size: int

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.ragged.base import RaggedBlockCausalAttnManager

        return RaggedBlockCausalAttnManager
