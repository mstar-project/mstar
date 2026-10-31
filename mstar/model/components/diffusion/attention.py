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

import functools
from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn.functional as F

# ``(q, k, v) -> out`` over ``[B, L, H, D]`` tensors (``k`` / ``v`` may have their own
# ``L``): ``sdpa_attention``, ``joint_attention`` over a ragged resource, or a wrapper.
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class RaggedAttentionFn(Protocol):
    """``(q, k, v) -> out`` over packed ``[total_tokens, H, D]`` tensors, one segment per
    request; with ``return_lse``, also ``lse`` ``[total_tokens, H]`` (fp32, natural log)."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


class RaggedAttentionMixin:
    """:meth:`ragged_for` for a ``NodeSubmodule`` whose layers attend through the ragged
    attention resource named ``attn_resource_key`` (None: SDPA everywhere)."""

    attn_resource_key: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One callable per attention label for the life of a binding.
        self._ragged_fns: dict[str, RaggedAttentionFn] = {}

    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._ragged_fns.clear()  # the memoized callables hold the old resource

    def ragged_for(self, label: str) -> RaggedAttentionFn | None:
        """``(q, k, v) -> out`` over the segments declared under ``label``, or None
        when no ragged resource is bound (layers then fall back to SDPA).

        The resource's ``run`` with the label bound, built once per label for the life
        of the binding rather than per step."""
        if self.attn_resource_key is None:
            return None
        if label not in self._ragged_fns:
            resource = self.node_resources.get(self.attn_resource_key)
            if resource is None:
                return None
            self._ragged_fns[label] = functools.partial(resource.run, label=label)
        return self._ragged_fns[label]


def sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional attention over ``[B, S, H, D]`` tensors via SDPA (``k`` / ``v`` may
    have their own length); returns ``[B, S, H, D]``, and with ``return_lse`` also the
    log-sum-exp of each query's scaled scores, ``[B, S, H]`` (fp32, natural log)."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if not return_lse:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2)
    # SDPA doesn't return the log-sum-exp; the queries that ask for it (replicated audio
    # over a video shard) are short, so the scores are cheap to materialise
    scores = (q.float() @ k.float().transpose(-2, -1)) * q.shape[-1] ** -0.5
    out = torch.softmax(scores, dim=-1) @ v.float()
    return out.to(q.dtype).transpose(1, 2), torch.logsumexp(scores, dim=-1).transpose(1, 2)


def joint_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ragged: RaggedAttentionFn | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attention of ``[B, S, H, D]`` queries over ``[B, S_kv, H_kv, D]`` keys / values
    (all requests in the batch share ``S`` and ``S_kv``; ``S_kv == S`` for
    self-attention) through the ragged resource when one is bound, else SDPA. With
    ``return_lse``, also the log-sum-exp ``[B, S, H]`` (see :func:`sdpa_attention`).

    The ragged kernel wants the batch packed to ``[B * S, H, D]`` / ``[B * S_kv, ...]``
    with one segment per request — exactly what the owning submodule declared in its
    step (a cross-attention pair for ``S_kv != S``) — and hands back the query packing.
    """
    if ragged is None:
        return sdpa_attention(q, k, v, return_lse=return_lse)
    bsz, seq, heads, dim = q.shape
    kv_seq, kv_heads = k.shape[1], k.shape[2]
    packed = (
        q.reshape(bsz * seq, heads, dim),
        k.reshape(bsz * kv_seq, kv_heads, dim),
        v.reshape(bsz * kv_seq, kv_heads, dim),
    )
    if not return_lse:
        return ragged(*packed).view(bsz, seq, heads, dim).to(q.dtype)
    out, lse = ragged(*packed, return_lse=True)
    return out.view(bsz, seq, heads, dim).to(q.dtype), lse.reshape(bsz, seq, heads)
