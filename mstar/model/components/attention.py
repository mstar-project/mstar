"""Multi-head attention with GQA, optional QK-norm, and pluggable RoPE.

Constructed with separate ``q_proj`` / ``k_proj`` / ``v_proj`` Linears
matching the HF checkpoint layout. After loading, call
``consolidate_qkv_weight()`` to fuse them into a single
``qkv_proj_weight`` buffer (one fused GEMM instead of three) and null
out the originals; the forward branches on whether consolidation has
happened.

Variations supported:
  * GQA via ``num_kv_heads`` < ``num_heads``.
  * Optional bias on qkv / o projections.
  * Optional per-head QK-norm (RMSNorm applied to q / k after projection,
    before RoPE) — used by qwen3.
  * Different ``input_hidden_size`` from the model's nominal ``hidden_size``
    (used by pi05's action expert, which shares K/V dims with PaliGemma
    but has its own width).
  * Llama-style RoPE scaling parameters (``rope_scale``,
    ``low_freq_factor``, etc.) for the cache-handle path.

For non-standard RoPE schemes (e.g. qwen3's 3D MRoPE), subclass and
override ``_apply_rope`` rather than going through ``cache_handle.apply_rope``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components.norm import RMSNorm


class Attention(nn.Module):
    def __init__(
        self,
        *,
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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.input_hidden_size = input_hidden_size or hidden_size

        # RoPE configuration: stashed for ``_apply_rope`` to pass through
        # to the cache handle. Subclasses overriding the rope path can
        # ignore these.
        self.rope_theta = rope_theta
        self.rope_scale = rope_scale
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope_old_context_len = rope_old_context_len

        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim
        self.q_proj = nn.Linear(self.input_hidden_size, q_out, bias=qkv_bias)
        self.k_proj = nn.Linear(self.input_hidden_size, kv_out, bias=qkv_bias)
        self.v_proj = nn.Linear(self.input_hidden_size, kv_out, bias=qkv_bias)
        self.o_proj = nn.Linear(q_out, self.input_hidden_size, bias=o_bias)

        if qk_norm:
            # Per-head RMSNorm; standard (non-Gemma) mode.
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def consolidate_qkv_weight(self) -> None:
        """Fuse ``q_proj``/``k_proj``/``v_proj`` weights into a single
        ``qkv_proj_weight`` buffer and null out the originals. Idempotent.
        """
        if self.q_proj is None:
            return
        if self.q_proj.bias is not None:
            raise NotImplementedError(
                "consolidate_qkv_weight does not yet handle biases."
            )
        qkv = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0,
        ).contiguous()
        self.register_buffer("qkv_proj_weight", qkv, persistent=False)
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    def _project_qkv(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        if self.q_proj is not None:
            q = self.q_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)
            k = self.k_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_kv_heads * self.head_dim
            qkv = F.linear(hidden_states, self.qkv_proj_weight)
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
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
        """Standard 1D RoPE via ``cache_handle.apply_rope``.

        Override in subclasses for non-standard schemes (3D MRoPE, etc.).
        """
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
