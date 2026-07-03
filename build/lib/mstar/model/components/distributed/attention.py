"""TP-aware multi-head attention.

Mirrors ``mstar.model.components.Attention`` but with the QKV projection
sharded across heads via ``QKVParallelLinear`` and the output projection
all-reduced via ``RowParallelLinear``. QK-norm (if enabled) and RoPE
each operate on this rank's local slice of heads — no cross-rank
communication beyond the AllReduce hidden inside ``o_proj``.

Worker integration:
  * The per-rank ``num_heads`` / ``num_kv_heads`` come from
    ``self.qkv_proj`` (already computed by ``QKVParallelLinear`` based on
    the comm group's world size and GQA replica count).
  * The cache handle's ``KVCacheConfig`` must be set up with the
    per-rank head counts so paged attention reads / writes the right
    slice. This is the caller's responsibility — typically the worker
    derives it from the request's ``ShardingConfig``.

For non-standard RoPE (qwen3's 3D MRoPE), subclass and override
``_apply_rope`` — same shape as the non-parallel ``Attention``.
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


class ParallelAttention(nn.Module):
    def __init__(
        self,
        *,
        comm_group: TPCommGroup | None = None,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_bias: bool = False,
        o_bias: bool = False,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        rope_scale: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_old_context_len: int = 8192,
        input_hidden_size: int | None = None,
    ):
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.comm_group = comm_group
        self.hidden_size = hidden_size
        self.input_hidden_size = input_hidden_size or hidden_size
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        self.qkv_proj = QKVParallelLinear(
            comm_group=comm_group,
            hidden_size=self.input_hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=qkv_bias,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads

        self.o_proj = RowParallelLinear(
            comm_group=comm_group,
            input_size=num_heads * head_dim,
            output_size=self.input_hidden_size,
            bias=o_bias,
            input_is_parallel=True,
            reduce_results=True,
        )

        self.rope_theta = rope_theta
        self.rope_scale = rope_scale
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope_old_context_len = rope_old_context_len

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def _project_qkv(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        return q, k, v

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None:
            return q, k
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q_shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k_shape)
        return q, k

    def _apply_rope(
        self,
        q: torch.Tensor, k: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard 1D RoPE via ``cache_handle.apply_rope``. Override for
        non-standard schemes (3D MRoPE, etc.)."""
        return cache_handle.apply_rope(
            q, k,
            rope_theta=self.rope_theta,
            rope_scale=self.rope_scale,
            low_freq_factor=self.rope_low_freq_factor,
            high_freq_factor=self.rope_high_freq_factor,
            old_context_len=self.rope_old_context_len,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        q, k = self._apply_qk_norm(q, k)
        q, k = self._apply_rope(q, k, cache_handle)
        attn_output = cache_handle.run_attention(q=q, k=k, v=v)
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)
