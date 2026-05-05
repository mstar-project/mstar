"""FlashInfer-based attention with QK-norm for Qwen3-Omni.

Supports two RoPE modes:
- **3D MRoPE** (Thinker): position embeddings are computed externally as
  ``cos_sin_3d`` and applied via ``apply_interleaved_mrope``.
- **1D RoPE** (Talker): standard rotary embeddings applied via
  ``cache_handle.apply_rope()``.

Follows the same shape conventions as ``OrpheusAttention``:
  q: [tokens, num_heads, head_dim]
  k: [tokens, num_kv_heads, head_dim]
"""

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.utils.flashinfer_utils import run_rms_norm


class Qwen3OmniRMSNorm(nn.Module):
    """RMSNorm whose forward is a no-op; call ``run_rms_norm`` directly."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # No-op: FlashInfer's run_rms_norm is called directly instead
        pass


class Qwen3OmniAttention(nn.Module):
    """Multi-head attention with QK-norm.

    Supports both 3D MRoPE (Thinker, when ``use_mrope=True``) and standard
    1D RoPE (Talker, when ``use_mrope=False``).

    Args:
        hidden_size: model hidden dimension.
        num_heads: number of query heads.
        num_kv_heads: number of key/value heads (GQA).
        head_dim: dimension per attention head.
        rope_theta: base frequency for standard 1D RoPE (Talker only).
        rms_norm_eps: epsilon for QK-norm RMSNorm layers.
        use_mrope: if True, expect external ``cos_sin_3d`` for 3D MRoPE
            instead of using ``cache_handle.apply_rope()``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        use_mrope: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.use_mrope = use_mrope

        # Linear projections (no bias, matching Qwen3-Omni checkpoint)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK-norm: per-head RMSNorm on q and k after projection, before RoPE
        self.q_norm = Qwen3OmniRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3OmniRMSNorm(head_dim, eps=rms_norm_eps)
    
    def set_qkv_proj_weight(self) -> torch.Tensor:
        if self.q_proj is not None:
            qkv_proj_weight = torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
                dim=0,
            ).contiguous()
            self.register_buffer("qkv_proj_weight", qkv_proj_weight, persistent=False)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [tokens, hidden_size]
            cache_handle: BatchedCacheManager with pre-planned attention.
            cos_sin_3d: tuple of (cos, sin) each [tokens, head_dim] for 3D
                MRoPE.  Required when ``use_mrope=True``.
            mrope_section: section sizes for interleaved 3D MRoPE, e.g.
                [24, 20, 20].  Required when ``use_mrope=True``.

        Returns:
            output: [tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]

        # 1. Project q, k, v
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.qkv_proj_weight.dtype)

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        qkv = F.linear(hidden_states, self.qkv_proj_weight)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query_states, key_states, value_states = qkv.split(
            (q_size, kv_size, kv_size), dim=-1
        )

        # 2. Reshape to [tokens, heads, head_dim]
        query_states = query_states.view(num_tokens, self.num_heads, self.head_dim)
        key_states = key_states.view(num_tokens, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(num_tokens, self.num_kv_heads, self.head_dim)

        # 3. QK-norm: apply RMSNorm per-head
        #    Reshape to [tokens * heads, head_dim], apply norm, reshape back
        query_states = run_rms_norm(
            query_states.reshape(-1, self.head_dim),
            self.q_norm.weight,
            eps=self.q_norm.variance_epsilon,
        ).view(num_tokens, self.num_heads, self.head_dim)

        key_states = run_rms_norm(
            key_states.reshape(-1, self.head_dim),
            self.k_norm.weight,
            eps=self.k_norm.variance_epsilon,
        ).view(num_tokens, self.num_kv_heads, self.head_dim)

        # 4. Apply RoPE
        if self.use_mrope and cos_sin_3d is not None:
            # 3D MRoPE (Thinker): applied externally via interleaved rotation.
            # cos_sin_3d is a tuple (cos, sin) already interleaved by
            # compute_3d_cos_sin (which consumed mrope_section).
            from mminf.model.qwen3_omni.components.rope import apply_interleaved_mrope

            cos, sin = cos_sin_3d
            query_states, key_states = apply_interleaved_mrope(
                query_states, key_states, cos, sin
            )
        else:
            # Standard 1D RoPE (Talker): use cache_handle's built-in RoPE
            query_states, key_states = cache_handle.apply_rope(
                query_states,
                key_states,
                rope_theta=self.rope_theta,
            )

        # 5. FlashInfer paged attention via cache_handle
        attn_output = cache_handle.run_attention(
            q=query_states, k=key_states, v=value_states
        )

        # 6. Reshape and project output
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output).to(orig_dtype)
