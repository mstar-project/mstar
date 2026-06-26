"""Ling-2.0 attention (TP-aware, packed-tokens, cache-handle-aware).

Uses mstar's :class:`QKVParallelLinear` + :class:`RowParallelLinear` for
TP-sharded projections. Per-rank head counts come from the QKV proj —
when ``tp_size > 1``, attention runs on this rank's slice of heads and
the output `dense` projection all-reduces across ranks.

The architecture-specific bits (per-head QK-norm, partial 3D
``video_rope`` rotation) stay inline — they only operate on this rank's
heads, no cross-rank comm.

Reference: mstar's :class:`ParallelAttention`
(`mstar/model/components/distributed/attention.py`) +
Qwen3-Omni's :class:`Qwen3OmniAttention`
(`mstar/model/qwen3_omni/components/attention.py`).
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components.distributed.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.ming_omni_flash.components.rope import LingPartialMRotaryEmbedding


class LingAttention(nn.Module):
    """Ling-2.0 attention layer (TP-aware).

    Constructor takes TOTAL head counts; per-rank counts are derived from
    ``qkv_proj.num_heads`` / ``qkv_proj.num_kv_heads`` after construction
    (computed by :class:`QKVParallelLinear` based on ``comm_group.world_size``).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rotary: LingPartialMRotaryEmbedding,
        use_qkv_bias: bool = False,
        use_bias: bool = False,
        comm_group: TPCommGroup | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by "
                f"num_kv_heads={num_kv_heads} for GQA"
            )
        if rotary.head_dim != head_dim:
            raise ValueError(
                f"rotary.head_dim={rotary.head_dim} must equal head_dim={head_dim}"
            )
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.comm_group = comm_group

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        # Packed QKV projection — TP-sharded along the heads axis.
        # Q rows: total_num_heads * head_dim; K rows: total_num_kv_heads *
        # head_dim; V rows: same. Stored ordered [Q, K, V] along dim 0 —
        # same packing the released ckpt uses for ``query_key_value.weight``,
        # so the manual q/k/v split in loader.py copies into the right
        # slots automatically.
        self.qkv_proj = QKVParallelLinear(
            comm_group=comm_group,
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=use_qkv_bias,
        )
        # Per-rank head counts; everything downstream uses these.
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5

        # Output projection — input dim is sharded (per-rank q_size),
        # output dim is full hidden_size; row-parallel runs all-reduce
        # across ranks.
        self.dense = RowParallelLinear(
            comm_group=comm_group,
            input_size=num_heads * head_dim,  # full pre-shard input
            output_size=hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            reduce_results=True,
        )

        # Per-head normalisation on q and k before rope. Operates on the
        # head_dim axis, so identical math at each rank's local heads.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary = rotary

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Engine-facing forward (packed tokens, cache-aware, TP-aware).

        Args:
            hidden_states: ``(num_tokens, hidden_size)``. NOT pre-sharded
                — QKVParallelLinear takes the full hidden dim as input.
            cache_handle: see step 3d.
            position_ids: see step 3d.
            rope_cos_sin: optional precomputed ``(cos, sin)`` from
                ``rotary.compute_cos_sin(position_ids)``. When provided,
                skip the per-layer cos/sin recompute (the model hoists it
                out of the layer loop — cos/sin are position-only and thus
                identical across layers). Falls back to recompute when None.

        Returns:
            ``(num_tokens, hidden_size)`` — full hidden dim after the
            row-parallel dense all-reduces across ranks.
        """
        num_tokens = hidden_states.shape[0]

        # qkv_proj returns this rank's slice along the heads axis:
        # (num_tokens, num_heads * head_dim + 2 * num_kv_heads * head_dim).
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # QK-norm: per-head RMSNorm on the head_dim axis. Each rank
        # operates on its own slice of heads — no comm.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(
            num_tokens, self.num_heads, self.head_dim
        )
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        # Partial 3D rope on this rank's heads (rope cos/sin are
        # head_dim-shaped, identical at every rank). Use the model-hoisted
        # cos/sin when provided to avoid recomputing the tables per layer.
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        if rope_cos_sin is not None:
            q, k = self.rotary.apply_cos_sin(q, k, rope_cos_sin[0], rope_cos_sin[1])
        else:
            q, k = self.rotary(q, k, position_ids)
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()

        # Cache attention on per-rank heads. mstar's BatchedCacheManager
        # is per-worker, so its KV cache config already accounts for the
        # per-rank head counts (worker derives this from ShardingConfig).
        attn_output = cache_handle.run_attention(q=q, k=k, v=v)
        attn_output = attn_output.reshape(num_tokens, self.q_size)
        # dense is row-parallel: it consumes the per-rank slice along the
        # input dim and all-reduces the (full hidden_size) output.
        return self.dense(attn_output)

    @staticmethod
    def head_norm_check(q_after_norm: torch.Tensor) -> float:
        """Diagnostic: returns max abs deviation of per-head RMS from 1."""
        norms = q_after_norm.float().pow(2).mean(dim=-1).sqrt()
        return (norms - 1.0).abs().max().item()
