"""Qwen3-Omni attention: ``ParallelAttention`` + 3D MRoPE override.

Reuses ``ParallelAttention`` (which already supports per-head QK-norm,
fused QKV projection, and TP-sharded o_proj). The Qwen3-specific piece
is the 3D MRoPE path used by the Thinker — for ``use_mrope=True`` the
RoPE call goes through ``apply_interleaved_mrope`` with externally
provided ``cos_sin_3d`` instead of the cache handle. Talker uses
standard 1D RoPE (``use_mrope=False``) through the step surface's
``apply_rope``. Every rope and attention call names its layer and plan
key explicitly.

Follows the same shape conventions as the shared attention:
  q: [tokens, num_heads, head_dim]
  k: [tokens, num_kv_heads, head_dim]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ParallelAttention

_fa2_maxlen_patched = False


def patch_hf_fa2_int_maxlen() -> None:
    """Make HF's FA2 encoder path torch.compile-able.

    ``Qwen3OmniMoeAudioAttention`` / ``Qwen3OmniMoeVisionAttention`` compute
    ``max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()`` and hand that
    0-dim tensor to ``flash_attn_varlen_func``, whose schema declares
    ``max_seqlen_q/k`` as ``SymInt``. Eager tolerates it (the arg parser
    unwraps a 0-dim tensor); dynamo's fake-tensor propagation does not, and
    the encoder forward dies with "Expected a value of type 'int'".

    Wrap the registered ``flash_attention_2`` interface so those two kwargs
    arrive as ints. Idempotent, and a no-op for callers already passing ints.
    """
    global _fa2_maxlen_patched
    if _fa2_maxlen_patched:
        return

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    inner = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    def fa2_int_maxlen(*args, max_length_q=None, max_length_k=None, **kwargs):
        if isinstance(max_length_q, torch.Tensor):
            max_length_q = max_length_q.item()
        if isinstance(max_length_k, torch.Tensor):
            max_length_k = max_length_k.item()
        return inner(*args, max_length_q=max_length_q, max_length_k=max_length_k, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", fa2_int_maxlen)
    _fa2_maxlen_patched = True


class Qwen3OmniAttention(ParallelAttention):
    """TP-aware attention with QK-norm and pluggable 1D / 3D RoPE.

    When ``use_mrope=True`` (Thinker) the forward expects a
    ``cos_sin_3d`` tuple of ``(cos, sin)`` tensors and applies
    ``apply_interleaved_mrope``. When False (Talker), the parent's
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
        pos_key: str | None = "rope"
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
            pos_key=pos_key

        )
        self.use_mrope = use_mrope

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
        *,
        layer_idx: int,
        label: str,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        q, k = self._apply_qk_norm(q, k)

        if self.use_mrope and cos_sin_3d is not None:
            from mstar.model.qwen3_omni.components.rope import apply_interleaved_mrope
            cos, sin = cos_sin_3d
            q, k = apply_interleaved_mrope(q, k, cos, sin)
        else:
            q, k = self._apply_rope(q, k, label)

        if self.attn.requires_kv_write:
            self.kv.write_kv(k, v, layer_idx=layer_idx, label=label)
        attn_output = self.attn.run(
            q, label, self.kv.layer_view(layer_idx),
            k=k, v=v, layer_idx=layer_idx,
        )
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)
