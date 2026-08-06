"""Attention state for nodes whose attention is not backed by a KV cache.

``BatchedCacheManager`` serves paged, cached attention for ``KV_CACHE`` nodes.
Encoder-style nodes (audio/vision towers) attend within variable-length segments
of a single packed forward — no paging, no cross-step state, nothing to advance.
They need only a planned FlashInfer wrapper.

``AttentionState`` is the narrow surface both shapes satisfy; a submodule declares
a ``RaggedAttentionConfig`` and the engine builds and owns the state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from mstar.utils.flashinfer_utils import RaggedPrefillWrapper  # noqa: F401

DEFAULT_WORKSPACE_BYTES = 128 * 1024 * 1024


class AttentionMode(Enum):
    """Which execution paths a node's ragged attention is built for.

    States are not interchangeable across paths — a captured bucket owns static
    buffers the graph recorded — so each path a node uses costs its own state
    and workspace. Declaring the paths you actually use avoids paying for the
    ones you don't.
    """

    #: Eager only. No capture-bucket states, and missing bucket bounds are not
    #: a misconfiguration.
    EAGER = "eager"
    #: Capture buckets only; ``ragged_attention_state`` stays None outside
    #: captured regions. For submodules keeping a different eager backend.
    CUDA_GRAPH = "cuda_graph"
    #: Both, the default.
    BOTH = "both"


@runtime_checkable
class AttentionState(Protocol):
    """Plan a layout, then run attention against it.

    Deliberately narrow: everything kernel-selecting is fixed when the state is
    built, so a captured graph can re-plan between replays without changing the
    kernel it recorded.
    """

    def plan(self, cu_seqlens: torch.Tensor) -> None:
        ...

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        ...


@dataclass
class RaggedAttentionConfig:
    """What a submodule declares to get ragged attention built for it.

    Head counts are **post-sharding** — a submodule that shards its attention
    across tensor-parallel ranks reports the per-rank counts, since it already
    knows how it built its own layers.

    ``make_attn_state`` is the escape hatch: return a factory to build something
    other than the default ``RaggedPrefillWrapper`` (a different backend, a
    fused variant). It receives the same keyword arguments
    ``build_ragged_attention_state`` would have used. Leave ``None`` for the
    default, which is what encoder towers want.
    """

    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool = False
    # Defaults to head_dim ** -0.5 on the TRUE head dim. Override only for
    # models that scale attention non-standardly.
    sm_scale: float | None = None
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES
    make_attn_state: Callable[..., AttentionState] | None = None
    # Per-request upper bounds used to size a CUDA-graph bucket: a capture at
    # batch size ``bs`` gets ``bs`` times this upper bound segment. A "segment"
    # is an independently-attending span, not a request, useful for requests with
    # multiple images/audio/etc.
    #
    # ``max_segments_per_request`` is required to capture ragged attention at
    # all. ``max_tokens_per_request`` is only needed by runners that don't
    # already bucket by token count (the piecewise runner takes the bucket's
    # ``total_tokens`` instead). Leave both None to opt out of captured ragged
    # attention; the eager state is still built.
    max_segments_per_request: int | None = None
    max_tokens_per_request: int | None = None
    # Which execution paths to build states for. Narrow it when a submodule
    # keeps a different backend on one of them — BAGEL's ViT declares
    # ``CUDA_GRAPH`` because eager runs flash-attn, which needs no head-dim
    # padding, so an eager state would be an unread ``workspace_bytes``.
    enabled_for: AttentionMode = AttentionMode.BOTH

    @property
    def eager_enabled(self) -> bool:
        return self.enabled_for in (AttentionMode.EAGER, AttentionMode.BOTH)

    @property
    def cuda_graph_enabled(self) -> bool:
        return self.enabled_for in (AttentionMode.CUDA_GRAPH, AttentionMode.BOTH)

    def max_segments_for(self, bs: int) -> int | None:
        if self.max_segments_per_request is None:
            return None
        return bs * self.max_segments_per_request

    def max_tokens_for(self, bs: int) -> int | None:
        if self.max_tokens_per_request is None:
            return None
        return bs * self.max_tokens_per_request


def build_ragged_attention_state(
    config: RaggedAttentionConfig,
    workspace_buffer: torch.Tensor,
    device: torch.device,
    q_data_type: torch.dtype = torch.bfloat16,
    use_cuda_graph: bool = False,
    max_num_segments: int | None = None,
    max_total_tokens: int | None = None,
) -> AttentionState:
    """Build one attention state from a submodule's config.

    Engines own the result. A node needs one eager state, plus one more per
    CUDA-graph capture bucket: a captured graph records its state's static
    buffers, so re-planning a state another graph recorded would corrupt it.
    ``max_num_segments`` / ``max_total_tokens`` are required when
    ``use_cuda_graph`` is set and size that bucket.
    """
    kwargs = {
        "workspace_buffer": workspace_buffer,
        "device": device,
        "q_data_type": q_data_type,
        "use_cuda_graph": use_cuda_graph,
        "max_num_segments": max_num_segments,
        "max_total_tokens": max_total_tokens,
    }
    if config.make_attn_state is not None:
        return config.make_attn_state(config=config, **kwargs)

    from mstar.utils.flashinfer_utils import RaggedPrefillWrapper

    return RaggedPrefillWrapper(
        num_qo_heads=config.num_qo_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        causal=config.causal,
        sm_scale=config.sm_scale,
        **kwargs,
    )


@dataclass
class RaggedAttention:
    """An engine's eager ``AttentionState`` s, one per node, and their workspaces.

    Workspaces are held here because the FlashInfer kernels read them: they must
    outlive every state built against them. Captured graph buckets get their own
    states, not these — see ``build_ragged_attention_state``.
    """

    device: torch.device
    states: dict[str, AttentionState] = field(default_factory=dict)
    workspaces: dict[str, torch.Tensor] = field(default_factory=dict)

    def build(
        self,
        key: str,
        config: RaggedAttentionConfig,
        q_data_type: torch.dtype = torch.bfloat16,
        use_cuda_graph: bool = False,
        max_num_segments: int | None = None,
        max_total_tokens: int | None = None,
    ) -> AttentionState:
        """Build and retain one state. ``key`` is a node name for the engine's
        eager states, or a capture-bucket id for a graph runner's."""
        workspace = torch.empty(
            config.workspace_bytes, dtype=torch.uint8, device=self.device
        )
        state = build_ragged_attention_state(
            config,
            workspace,
            self.device,
            q_data_type=q_data_type,
            use_cuda_graph=use_cuda_graph,
            max_num_segments=max_num_segments,
            max_total_tokens=max_total_tokens,
        )
        self.workspaces[key] = workspace
        self.states[key] = state
        return state

    def get(self, key: str) -> AttentionState | None:
        return self.states.get(key)
