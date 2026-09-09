"""GLM-5.2 DSA engine plumbing: per-request indexer k-cache + selection threading.

Two pieces the sparse path needs that the Phase C components deliberately left
to the engine half:

``Glm52DsaKStore`` — the indexer k-cache. v1 keeps it OUTSIDE the paged KV
pool: a glm52-local ``request_id -> FULL-layer -> growing buffer`` of the
roped+normed ``index_head_dim`` keys ``Glm52Indexer.compute_k`` produces,
owned by ``Glm52LLMSubmodule`` and evicted in its ``cleanup_request`` (the
engine calls that from ``KVCacheEngine.remove_request`` — the contract
``test/modular/test_kv_cache_engine_cleanup.py`` pins). Cost at full dims:
128 dims x bf16 = 256 B/token/layer over 21 FULL layers = 5.4 KB/token,
replicated per TP rank (the whole indexer is replicated, so every rank
recomputes the identical selection with no collective). The reference
layout — fp8 e4m3 + ue8m0 scale packed 132 B/token in a paged pool
(dsa-indexer-spec.md section 2) — is the perf follow-up; the spec marks
bf16 storage semantically equivalent up to fp8 rounding, set-exact at
ctx <= index_topk.

``Glm52DsaForwardContext`` — one per forward pass, built by the submodule's
``preprocess`` and threaded ``Glm52LanguageModel -> decoder layer ->
attention``. It carries the per-request token spans of the flattened batch
plus the ONE transient the IndexShare scheme needs: ``last_selection``,
written by each FULL layer and consumed as-is by the SHARED layers after it
(spec section 3: layers 3-5 reuse 2's selection, 7-9 reuse 6's, ...). The
selection never persists across forwards — only the k-store does.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

_INITIAL_CAPACITY = 64


class Glm52DsaKStore:
    """Per-request, per-FULL-layer growing buffers of roped+normed index keys.

    Buffers grow by doubling so a long decode is O(n) amortized copies
    rather than the O(n^2) of per-step ``torch.cat``. Rows are appended in
    strict position order — ``append`` refuses any gap or overlap loudly,
    because a silent desync would corrupt every later selection for the
    request.
    """

    def __init__(self) -> None:
        # request_id -> layer_idx -> [buffer (capacity, head_dim), filled rows]
        self._buffers: dict[str, dict[int, list]] = {}

    def append(
        self,
        request_id: str,
        layer_idx: int,
        keys: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Append one chunk of ``(n, head_dim)`` keys at positions
        ``start_pos .. start_pos + n - 1``; ``start_pos`` must equal the rows
        already stored (prefill and decode both append exactly once per
        token, in order)."""
        per_layer = self._buffers.setdefault(request_id, {})
        num_new, head_dim = keys.shape
        entry = per_layer.get(layer_idx)
        if entry is None:
            capacity = max(_INITIAL_CAPACITY, num_new)
            entry = per_layer[layer_idx] = [keys.new_empty((capacity, head_dim)), 0]
        buffer, filled = entry
        if start_pos != filled:
            raise RuntimeError(
                f"DSA k-store desync for request {request_id!r} layer {layer_idx}: "
                f"append at position {start_pos} but {filled} keys are stored"
            )
        needed = filled + num_new
        if needed > buffer.shape[0]:
            grown = buffer.new_empty((max(needed, 2 * buffer.shape[0]), head_dim))
            grown[:filled] = buffer[:filled]
            entry[0] = buffer = grown
        buffer[filled:needed] = keys.detach()
        entry[1] = needed

    def history(self, request_id: str, layer_idx: int, length: int) -> torch.Tensor:
        """The first ``length`` stored keys — a view, not a copy. Raises if
        fewer are stored (a scoring window must never exceed the appended
        history: the current chunk is appended before selection)."""
        entry = self._buffers[request_id][layer_idx]
        buffer, filled = entry
        if length > filled:
            raise RuntimeError(
                f"DSA k-store for request {request_id!r} layer {layer_idx} holds "
                f"{filled} keys; selection window wants {length}"
            )
        return buffer[:length]

    def evict(self, request_id: str) -> None:
        """Drop all of a request's buffers. Idempotent — the engine may retire
        a request that never reached a DSA forward."""
        self._buffers.pop(request_id, None)

    # -- introspection (tests + leak asserts) ---------------------------------

    def tracked_requests(self) -> set[str]:
        return set(self._buffers)

    def tokens(self, request_id: str, layer_idx: int) -> int:
        """Stored rows for one (request, layer); 0 when absent."""
        entry = self._buffers.get(request_id, {}).get(layer_idx)
        return 0 if entry is None else entry[1]


@dataclass
class Glm52DsaRequestSpan:
    """One request's slice of the flattened token batch, frozen at preprocess.

    ``ctx_start`` is the request's cached length BEFORE this chunk (the
    alloc-manager ``position_id_start``, which ``advance_seq_lens`` bumps only
    after the forward), so the chunk covers absolute positions
    ``ctx_start .. ctx_start + q_len - 1``. ``page_indices`` is a snapshot of
    the request's latent-cache page table taken AFTER ``plan_attention``
    allocated this chunk's pages, so it covers every position the sparse path
    may scatter or gather.
    """

    request_id: str
    q_start: int
    q_len: int
    ctx_start: int
    page_indices: list[int]


@dataclass
class Glm52DsaForwardContext:
    """Per-forward DSA state threaded through the decoder stack.

    ``needs_selection`` is the batch-level routing bit: False means every
    request's post-chunk context fits ``index_topk``, where top-k of <= topk
    candidates selects the full prefix — the identity regime — so layers skip
    selection entirely and run the UNTOUCHED dense paged path (bit-identical
    to the flag-off serve, M1's foundation). FULL layers still append to the
    k-store either way: history must be complete from token 0 for the step
    that first crosses topk.

    ``last_selection`` is the IndexShare transient: ``(batch_tokens, topk)``
    int32 rows, -1 padded, overwritten by each FULL layer and read as-is by
    SHARED layers (which carry no indexer weights). It is never reused across
    forwards — the context object dies with the forward pass.
    """

    spans: list[Glm52DsaRequestSpan]
    k_store: Glm52DsaKStore
    needs_selection: bool
    last_selection: torch.Tensor | None = field(default=None)
    last_selection_layer: int | None = field(default=None)
