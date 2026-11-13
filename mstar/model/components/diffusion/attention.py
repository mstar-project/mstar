"""Joint attention over a DiT's packed tokens.

A diffusion transformer attends bidirectionally over all of a request's tokens
(text + image + reference-image) and keeps nothing between steps, so there is
no KV cache to page. Two backends serve the same ``[B, S, H, D]`` contract:

* ``sdpa``   ``torch.nn.functional.scaled_dot_product_attention`` — the reference
             pipelines' own kernel, hence the parity baseline; used eagerly and on CPU.
* the engine's **ragged attention resource** (FlashInfer's ragged prefill on
  FA2/FA3) — one segment per request, planned by the runner outside the captured
  graph, so a per-step CUDA graph can replay it.

Layers call :func:`attend`, and the array's read picks the backend at bind from its
resource's ``backend``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from mstar.engine.state import Entry, TensorType

# ``(q, k, v) -> out`` over ``[B, L, H, D]`` tensors (``k`` / ``v`` may have their own
# ``L``): ``sdpa_attention``, ``joint_attention`` over a ragged resource, or a wrapper.
Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class RaggedAttentionFn(Protocol):
    """``(q, k, v) -> out`` over packed ``[total_tokens, H, D]`` tensors, one segment per
    request; with ``return_lse``, also ``lse`` ``[total_tokens, H]`` (fp32, natural log)."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def dit_entry(heads: int, head_dim: int) -> Entry:
    """A DiT's k and v per token, in the bf16 its attention runs in: it has no autocast dtype to resolve to."""
    t = TensorType((heads, head_dim), dtype=torch.bfloat16, shard=0)
    return Entry({"k": t, "v": t})


def refuse_attention_backend_kwarg(kwargs: Mapping[str, Any]) -> None:
    """A constructor's ``**kwargs`` would swallow the old knob, and a config that asks for SDPA would run FlashInfer."""
    if "attention_backend" in kwargs:
        raise ValueError("attention_backend is a resource's setting: move it from model_kwargs to "
                         "`resources: {<the read's resource key>: {backend: sdpa}}`")


def attend(state: Any, layer: torch.Tensor | None, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``[B, S, H, D]`` queries over ``[B, S_kv, H_kv, D]`` keys and values through ``state``'s read, or SDPA on a layer
    no engine bound a handle to."""
    if state is None:
        return sdpa_attention(q, k, v)
    state.add(layer, k=k, v=v)
    return state.read(layer, q)


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


class FrameSDPA:
    """SDPA within each segment of ``lengths``, set before each call. It stands in for a ragged resource's ``run``
    where none is declared (a model's SDPA mode, the reference's own kernel)."""

    def __init__(self, lengths=()):
        self.lengths = list(lengths)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, label: str | None = None) -> torch.Tensor:
        # [1, heads, segment, head_dim] per segment, the call transformers makes (and so its kernel)
        out = [
            F.scaled_dot_product_attention(*(t.transpose(0, 1)[None] for t in segment))[0].transpose(0, 1)
            for segment in zip(q.split(self.lengths), k.split(self.lengths), v.split(self.lengths), strict=True)
        ]
        return torch.cat(out)


def joint_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ragged: RaggedAttentionFn | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attention of ``[B, S, H, D]`` queries over ``[B, S_kv, H_kv, D]`` keys / values
    (all requests in the batch share ``S`` and ``S_kv``; ``S_kv == S`` for
    self-attention) through the ragged resource when one is bound, else SDPA. With
    ``return_lse``, also the log-sum-exp ``[B, S, H]`` (see :func:`sdpa_attention`).

    The ragged kernel wants the batch packed to ``[B * S, H, D]`` / ``[B * S_kv, ...]``
    with one segment per request — exactly what the step declared (a cross read for
    ``S_kv != S``) — and hands back the query packing.
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
