"""Qwen3-Omni attention: ``ParallelAttention`` + 3D MRoPE override.

Reuses ``ParallelAttention`` (which already supports per-head QK-norm,
fused QKV projection, and TP-sharded o_proj). The Qwen3-specific piece
is the 3D MRoPE path used by the Thinker — for ``use_mrope=True`` q/k
norm, the rotation by externally provided ``cos_sin_3d`` and the KV
write run in one Triton kernel, the state op ``qk_norm_rope_table_kv_write``.
Talker uses standard 1D RoPE (``use_mrope=False``) through its array's
positions. Every state call names its layer.

Follows the same shape conventions as the shared attention:
  q: [tokens, num_heads, head_dim]
  k: [tokens, num_kv_heads, head_dim]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.state import PagedNHD, PagedNHDTarget, rope, state_op
from mstar.model.components.distributed import ParallelAttention
from mstar.model.components.norm import rms_norm


def _norm_rotate(
    x: torch.Tensor, weight: torch.Tensor, eps: float, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    half = x.shape[-1] // 2
    normed = rms_norm(x.reshape(-1, x.shape[-1]), weight, eps).view(x.shape)
    return normed * cos + torch.cat((-normed[..., half:], normed[..., :half]), dim=-1) * sin


def _norm_rotate_add(
    kv, layer, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, q_eps: float, k_eps: float,
) -> None:
    cos, sin = cos[: q.shape[0], None], sin[: q.shape[0], None]
    rotated_k = _norm_rotate(k, k_weight, k_eps, cos, sin)
    q.copy_(_norm_rotate(q, q_weight, q_eps, cos, sin))
    kv.add(layer, k=rotated_k, v=v)


# The kernel is CUDA's, so any other device takes the reference.
@state_op(adds=("kv",), mutates=("q",), formats=(PagedNHD(devices={"cuda"}),), reference=_norm_rotate_add)
def qk_norm_rope_table_kv_write(
    kv, layer, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, q_eps: float, k_eps: float, *, target: PagedNHDTarget,
) -> None:
    """Per-head q/k norm and the rotation by rotate-half ``cos`` and ``sin`` rows, in place on ``q``, and this layer's
    K/V write, in one launch."""
    from mstar.utils import attention

    attention.qk_norm_rope_table_kv_write(
        q, k, v, q_weight, k_weight, cos, sin, target.cache, target.page_of_token, target.slot_of_token,
        min(target.count, q.shape[0]), q_eps, k_eps,
    )


class Qwen3OmniAttention(ParallelAttention):
    """TP-aware attention with QK-norm and pluggable 1D / 3D RoPE.

    When ``use_mrope=True`` (Thinker) the forward expects a
    ``cos_sin_3d`` tuple of ``(cos, sin)`` tensors, the rotate-half tables
    of its interleaved 3D MRoPE. When False (Talker), the parent's
    standard cache-handle RoPE is used.
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
        comm_group: CommGroup | None = None,
        attn_key: str = "attn",
        kv_key: str = "kv",
        pos_key: str | None = "rope",
        layer: int | None = None,
    ):
        super().__init__(
            comm_group=comm_group,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            o_bias=False,
            qk_norm=True,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attn_key=attn_key,
            kv_key=kv_key,
            pos_key=pos_key,
            layer=layer,
        )
        self.use_mrope = use_mrope

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)

        if self.use_mrope and cos_sin_3d is not None:
            qk_norm_rope_table_kv_write(
                self.state, self.layer, q, k, v, self.q_norm.weight, self.k_norm.weight, *cos_sin_3d,
                float(self.q_norm.variance_epsilon), float(self.k_norm.variance_epsilon),
            )
            attn_output = self.state.read(self.layer, q)
        else:
            q, k = self._apply_qk_norm(q, k)
            # Here, not through `_attend_state`, whose break in `add` would compile the rotation in a frame of its own.
            if self._pos_key is not None:
                q, k = rope(self.state, q, k)
            self.state.add(self.layer, k=k, v=v)
            attn_output = self.state.read(self.layer, q)

        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)
