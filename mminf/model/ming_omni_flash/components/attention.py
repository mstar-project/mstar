"""Ling-2.0 attention block (with QK-norm + partial 3D MRoPE).

This module captures the **architecture-novel** pieces of Ling-2.0's
attention without taking on the full mminf KV-cache / TP attention path
yet — those land in step 3b when the decoder layer assembles. Here we
expose:

  * The QKV projection (kept dense for now; will become
    :class:`QKVParallelLinear` in step 3b).
  * Per-head RMSNorm on q and k **before** applying RoPE
    (``use_qk_norm: true`` on this checkpoint).
  * The :class:`LingPartialMRotaryEmbedding` rotation on the rotary half.
  * A plain scaled-dot-product attention forward — bypasses mminf's
    KV-cache because step 3a is unit-test scope (small dim, no batching,
    no real prefill/decode).

The exact same forward shape is what the eventual
``LingDecoderLayer`` will call, except the projections will be the
TP-sharded variants.

Reference: vllm-omni's :class:`BailingMoeV2Attention`
``/tmp/vllm-omni/vllm_omni/model_executor/models/ming_flash_omni/modeling_bailing_moe_v2.py:436-563``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mminf.model.components.norm import RMSNorm
from mminf.model.ming_omni_flash.components.rope import LingPartialMRotaryEmbedding


class LingAttention(nn.Module):
    """Plain multi-head attention with QK-norm + partial MRoPE.

    Args:
        hidden_size: model hidden dim.
        num_heads: total query heads (no TP split here — step 3b handles TP).
        num_kv_heads: total KV heads (GQA).
        head_dim: per-head dim.
        rms_norm_eps: epsilon for RMSNorm on q and k.
        rotary: pre-built :class:`LingPartialMRotaryEmbedding`. Injecting it
            (rather than constructing here) lets a decoder layer share one
            rope instance across layers — the inv_freq buffer is identical.
        use_qkv_bias: bias on the qkv projection (False for released ckpt).
        use_bias: bias on the output projection (False for released ckpt).
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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5

        # Packed QKV projection (matches upstream QKVParallelLinear layout
        # at total_num_heads*head_dim + 2*total_num_kv_heads*head_dim).
        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=use_qkv_bias,
        )
        self.dense = nn.Linear(self.q_size, hidden_size, bias=use_bias)

        # Per-head normalisation on q and k (one RMSNorm per head_dim,
        # applied identically across heads — that's what mirrors the
        # upstream ``RMSNorm(head_dim)`` call sites).
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary = rotary

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run attention.

        Args:
            hidden_states: ``(num_tokens, hidden_size)`` or
                ``(batch, num_tokens, hidden_size)``.
            position_ids: ``(num_tokens,)`` or ``(3, num_tokens)`` — passed
                to the rotary module.

        Returns:
            Output of shape matching ``hidden_states``.
        """
        squeezed = hidden_states.dim() == 2
        if squeezed:
            hidden_states = hidden_states.unsqueeze(0)  # (1, T, H)
        bsz, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head reshape so RMSNorm operates per-head on head_dim.
        # Shape after view: (B, T, num_heads_or_kv, head_dim).
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # RMSNorm across head_dim, broadcast across heads.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE — expects (..., num_tokens, head_dim) and we have
        # (B, T, H, head_dim). Squeeze B for the single-batch step-3a
        # path; eventual TP path will handle batched ropes natively.
        if bsz != 1:
            raise NotImplementedError(
                "step-3a LingAttention only validates batch=1; full TP path "
                "with batched rope lands in step 3b"
            )
        q_t = q.squeeze(0).transpose(0, 1)  # (H, T, head_dim)
        k_t = k.squeeze(0).transpose(0, 1)
        # rope expects shape (..., T, head_dim) — H prefix is broadcast over.
        q_t, k_t = self.rotary(q_t, k_t, position_ids)
        q = q_t.transpose(0, 1).unsqueeze(0)
        k = k_t.transpose(0, 1).unsqueeze(0)

        # SDP attention. F.scaled_dot_product_attention expects
        # (B, num_heads, T, head_dim).
        q = q.transpose(1, 2)                      # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)                      # (B, num_kv_heads, T, head_dim)
        v = v.transpose(1, 2)
        # GQA: expand kv heads to num_heads via repeat_interleave.
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scaling,
        )
        # Back to (B, T, num_heads * head_dim) then dense.
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            bsz, seq_len, self.q_size,
        )
        out = self.dense(attn_out)
        if squeezed:
            out = out.squeeze(0)
        return out

    @staticmethod
    def head_norm_check(q_after_norm: torch.Tensor) -> float:
        """Diagnostic helper used in tests — returns the max abs deviation
        of per-head L2 norm from sqrt(head_dim) after RMSNorm. Should be
        ~0 for a freshly initialised RMSNorm (weight=1 → unit-RMS output).

        Mostly exists so the test can verify QK-norm actually fired
        without monkey-patching the forward.
        """
        # RMSNorm makes per-token, per-head RMS == 1, so L2 norm ==
        # sqrt(head_dim).
        head_dim = q_after_norm.shape[-1]
        norms = q_after_norm.float().pow(2).mean(dim=-1).sqrt()  # RMS per head
        return (norms - 1.0).abs().max().item()
