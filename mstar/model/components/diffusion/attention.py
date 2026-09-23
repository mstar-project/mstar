"""Joint attention over a DiT's packed tokens.

A diffusion transformer attends bidirectionally over all of a request's tokens
(text + image + reference-image) and keeps nothing between steps, so there is
no KV cache to page. Two backends serve the same ``[B, S, H, D]`` contract:

* ``sdpa``   ``torch.nn.functional.scaled_dot_product_attention`` — the reference
             pipelines' own kernel, hence the parity baseline; used eagerly and on CPU.
* the engine's **ragged attention resource** (``RaggedAttentionSpec``, FlashInfer's
  ragged prefill on FA2/FA3) — one segment per request, planned by the runner
  outside the captured graph, so a per-step CUDA graph can replay it. The
  resource is bound by the submodule and threaded to the layers as a callable.

Layers call :func:`joint_attention`; the choice is made once per forward by
whoever owns the resource, not per layer.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

# ``(q, k, v) -> out`` over packed ``[total_tokens, H, D]`` tensors, one segment per request.
RaggedAttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bidirectional attention over ``[B, S, H, D]`` tensors via SDPA; returns ``[B, S, H, D]``."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False,
    )
    return out.transpose(1, 2)


def joint_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ragged: RaggedAttentionFn | None = None,
) -> torch.Tensor:
    """Attention over ``[B, S, H, D]`` q/k/v (all requests in the batch share
    ``S``) through the ragged resource when one is bound, else SDPA.

    The ragged kernel wants the batch packed to ``[B * S, H, D]`` with one
    segment of ``S`` tokens per request — exactly what the owning submodule
    declared in its step — and hands back the same packing.
    """
    if ragged is None:
        return sdpa_attention(q, k, v)
    bsz, seq, heads, dim = q.shape
    out = ragged(
        q.reshape(bsz * seq, heads, dim), k.reshape(bsz * seq, heads, dim), v.reshape(bsz * seq, heads, dim),
    )
    return out.view(bsz, seq, heads, dim).to(q.dtype)
