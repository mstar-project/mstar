"""What a model declares about cacheless (ragged) attention.

Kept free of the manager and its kernels, like the other resources' configs, so
a submodule can declare a step without pulling FlashInfer in behind it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.sequence_shard import SeqShardSpec


def cross_label(q_label: str, kv_label: str) -> str:
    """The plan a cross-attention pair runs under: ``q_label``'s spans attending
    ``kv_label``'s."""
    return f"{q_label}<-{kv_label}"


class RaggedAttnBackend(Enum):
    # the platform's default, decided when the resource is built
    AUTO = "auto"
    FLASHINFER = "flashinfer"
    # FA3 (AITER on ROCm) varlen; YAML name matches AttnBackend.DENSE
    DENSE = "dense"


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
    # For a resource whose captured region replays a whole step as one row
    # (`one_row_per_step`): every segment of the step, every request's, that
    # one replay's plan holds, in place of the per-request ceiling.
    max_segments_per_step: int | None = None
    # Only for a runner that buckets by batch size alone; elsewhere the
    # bucket's own token count is the ceiling and this stays None.
    max_tokens_per_request: int | None = None

    backend: RaggedAttnBackend = RaggedAttnBackend.AUTO
    flashinfer_backend: str = "auto"

    # Whether a compiled caller keeps the attention in its graph, as the
    # ``mstar::ragged_attention`` op, rather than breaking there per layer. For
    # a compiled, uncaptured region whose fusions around attention pay; off by
    # default, since tracing a whole transformer as one graph per capture
    # bucket costs startup time and capture memory (Z-Image: +2.6 GiB).
    compile_op: bool = False

    # Activation dtype of the q / k / v the node hands the kernel (and of its
    # output). A cacheless attention doesn't need to adhere to the dtype of its
    # KV cache, so it is free to declare a separate dtype here; when None, the
    # engine's KV dtype (which is actually a misnamed autocast dtype) is used.
    dtype: torch.dtype | None = None

    # Each label's FlashInfer workspace, per slot; None takes
    # MSTAR_WORKSPACE_BUFFER_MB. A ragged prefill only spends it on split-KV
    # partials (tens of MB), so a resource with many labels can declare less.
    workspace_mb: int | None = None

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
        if self.max_segments_per_step is not None:
            return self.max_segments_per_step
        return bs * self.max_segments_per_request

    def max_tokens_for(self, bs: int) -> int | None:
        if self.max_tokens_per_request is None:
            return None
        return bs * self.max_tokens_per_request


@dataclass
class _RaggedSpecBase(NodeResourceSpec):
    """The config and deployment overrides both ragged specs share."""

    config: RaggedAttentionConfig
    # the sequence shard over its nodes, if any (`derive`, on the kinds that take split steps)
    shard_key: str | None = field(default=None, init=False, repr=False)
    # how a split label's tokens reach the heads: declared on self-attention (`RaggedAttentionSpec`),
    # derived per pair on cross-attention
    sp_exchange = "none"

    def depends_on(self) -> set[str]:
        return set() if self.shard_key is None else {self.shard_key}

    def apply_yaml_overrides(
        self,
        backend: str | RaggedAttnBackend | None = None,
        flashinfer_backend: str | None = None,
        max_segments_per_request: int | None = None,
        max_segments_per_step: int | None = None,
        max_tokens_per_request: int | None = None,
    ):
        """Which kernel to run is the deployment's call as much as the model's —
        an image that cannot build FA3 pins FA2 here.

        The ceilings are here rather than on the model because they size
        CUDA-graph buckets, which is a deployment's memory/coverage trade. A
        resource sized per step (``max_segments_per_step``) never reads the
        per-request segment ceiling, so an override of that is refused.
        """
        per_step = self.config.max_segments_per_step if max_segments_per_step is None else max_segments_per_step
        if max_segments_per_request is not None and per_step is not None:
            raise ValueError(
                f"resource {self.resource_key!r} is sized per step: set max_segments_per_step, "
                "not max_segments_per_request"
            )
        if backend is not None:
            self.config.backend = RaggedAttnBackend(backend)
        if flashinfer_backend is not None:
            self.config.flashinfer_backend = flashinfer_backend
        if max_segments_per_request is not None:
            self.config.max_segments_per_request = max_segments_per_request
        if max_segments_per_step is not None:
            self.config.max_segments_per_step = max_segments_per_step
        if max_tokens_per_request is not None:
            self.config.max_tokens_per_request = max_tokens_per_request


@dataclass
class RaggedAttentionSpec(_RaggedSpecBase):
    """Cacheless self-attention within each declared span (``AttentionStep``).

    ``sp_exchange="ulysses"`` attends a label its node's sequence shard splits by Ulysses (one
    all-to-all from split tokens to split heads, and back), and shards heads over every TP and SP
    rank, slicing them on a copied step. ``"none"`` (the default) attends only copied labels, heads
    over TP. A no-op on a node with no ``SeqShardSpec``. On SP ranks every label of a ``ulysses``
    resource goes through an exchange, so ``run(..., return_lse=True)`` raises there."""

    # the model's declaration, not a deployment's (``sp_exchanges`` stays empty)
    sp_exchange: Literal["none", "ulysses"] = field(default="none", kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.sp_exchange not in ("none", "ulysses"):
            raise ValueError(f"sp_exchange must be 'none' or 'ulysses', not {self.sp_exchange!r}")

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        from mstar.engine.resources.sequence_shard import take_shard_edge

        return take_shard_edge(self, specs)

    def resolve_exchange(self, specs: Mapping[str, NodeResourceSpec]) -> Literal["none", "ulysses"]:
        return self.sp_exchange

    def split_step_refusals(self, node: str, shard: "SeqShardSpec", tp: int, sp: int) -> list[str]:
        cfg = self.config
        q, kv = cfg._unsharded_qo_heads, cfg._unsharded_kv_heads
        if self.sp_exchange != "ulysses":
            return []
        if shard.stripe_pool is not None:
            return [
                f"ragged attention {self.resource_key!r} declares sp_exchange 'ulysses', whose plan reads a split "
                f"label's tokens in step order, and node {node!r}'s sequence shard stripes them by KV "
                f"{shard.stripe_pool!r}'s positions (its exchange ring)"
            ]
        if q % (tp * sp) or kv % (tp * sp):
            return [
                f"ragged attention {self.resource_key!r} declares sp_exchange 'ulysses', and its {q} query and "
                f"{kv} KV heads don't split over {tp} TP x {sp} SP ranks"
            ]
        return []

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.ragged.base import RaggedAttnManager

        return RaggedAttnManager


@dataclass
class RaggedCrossAttentionSpec(_RaggedSpecBase):
    """Cacheless cross-attention between two spans of each request: a forward's
    ``(q_label, kv_label)`` pairs (``RaggedCrossAttentionStep``), each run through
    ``resource.run(q, k, v, label=cross_label(q_label, kv_label))``.

    Separate from ``RaggedAttentionSpec`` so a resource is one kind of attention:
    a model with both declares one of each per head geometry.

    On a node whose sequence shard splits a step, each pair's exchange follows from which side
    is split: split queries over copied keys attend locally, copied queries over split keys merge
    every rank's partial (``sharded_key_merge``), and both split is refused. The shard must split
    each row (``axis=1``): a packed split (``axis=0``) can leave a rank none of a row.
    """

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        from mstar.engine.resources.sequence_shard import take_shard_edge

        return take_shard_edge(self, specs)

    def split_step_refusals(self, node: str, shard: "SeqShardSpec", tp: int, sp: int) -> list[str]:
        if shard.axis == 0:
            return [
                f"ragged cross-attention {self.resource_key!r} on node {node!r}, whose sequence shard "
                f"{shard.resource_key!r} splits packed tokens (axis 0), which can leave a rank no tokens of a "
                "row to attend: split each row (axis 1)"
            ]
        return []

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
