"""Joint attention over a DiT's packed tokens.

A diffusion transformer attends bidirectionally over all of a request's tokens
(text + image + reference-image) and keeps nothing between steps, so there is
no KV cache to page. Two backends serve the same ``[B, S, H, D]`` contract:

* ``sdpa``   ``torch.nn.functional.scaled_dot_product_attention`` — the reference
             pipelines' own kernel, hence the parity baseline; used eagerly and on CPU.
* the engine's **ragged attention resource** (``RaggedAttentionSpec``): one segment
  per request, planned by the runner outside the captured graph, so a per-step
  accelerator graph can replay it. The kernel is the platform's (FlashInfer's ragged
  prefill on FA2/FA3 on NVIDIA GPUs, AITER's varlen flash attention on AMD GPUs) and
  a deployment can pin another under ``resources:``. The resource is bound by the
  submodule and threaded to the layers as a callable.

Layers call :func:`joint_attention`; the choice is made once per forward by
whoever owns the resource, not per layer. A model's ``attention_backend`` serving
knob picks between the two (:data:`ATTENTION_BACKENDS`).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

# ``(q, k, v) -> out`` over packed ``[total_tokens, H, D]`` tensors, one segment per request.
RaggedAttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

# Values of a DiT model's ``attention_backend`` serving knob. ``auto`` runs every attention on the
# engine's ragged resource with the platform's default kernel; ``sdpa`` keeps torch's kernel inline,
# the reference pipelines' own, for parity runs. ``flashinfer`` is the earlier spelling of ``auto``
# (the ragged resource was FlashInfer-only when it was named) and means the same thing.
ATTENTION_BACKENDS = ("auto", "sdpa", "flashinfer")


def uses_ragged_attention(attention_backend: str) -> bool:
    """Whether ``attention_backend`` puts the model's attention on the engine's ragged resource."""
    if attention_backend not in ATTENTION_BACKENDS:
        raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {attention_backend!r}")
    return attention_backend != "sdpa"


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bidirectional attention over ``[B, S, H, D]`` tensors via SDPA (``k`` / ``v`` may
    have their own length); returns ``[B, S, H, D]``."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False,
    )
    return out.transpose(1, 2)


def joint_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ragged: RaggedAttentionFn | None = None,
) -> torch.Tensor:
    """Attention of ``[B, S, H, D]`` queries over ``[B, S_kv, H_kv, D]`` keys / values
    (all requests in the batch share ``S`` and ``S_kv``; ``S_kv == S`` for
    self-attention) through the ragged resource when one is bound, else SDPA.

    The ragged kernel wants the batch packed to ``[B * S, H, D]`` / ``[B * S_kv, ...]``
    with one segment per request — exactly what the owning submodule declared in its
    step (a cross-attention pair for ``S_kv != S``) — and hands back the query packing.
    """
    if ragged is None:
        return sdpa_attention(q, k, v)
    bsz, seq, heads, dim = q.shape
    kv_seq, kv_heads = k.shape[1], k.shape[2]
    out = ragged(
        q.reshape(bsz * seq, heads, dim),
        k.reshape(bsz * kv_seq, kv_heads, dim),
        v.reshape(bsz * kv_seq, kv_heads, dim),
    )
    return out.view(bsz, seq, heads, dim).to(q.dtype)
