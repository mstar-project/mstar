"""GLM-5.2 DSA engine plumbing: per-request indexer k-cache + selection threading."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

_INITIAL_CAPACITY = 64


class Glm52DsaKStore:
    """Per-request, per-FULL-layer growing buffers of roped+normed index keys."""

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
        """Append one chunk of ``(n, head_dim)`` keys at positions ``start_pos .."""
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
    """One request's slice of the flattened token batch, frozen at preprocess."""

    request_id: str
    q_start: int
    q_len: int
    ctx_start: int
    page_indices: list[int]


@dataclass
class Glm52DsaForwardContext:
    """Per-forward DSA state threaded through the decoder stack."""

    spans: list[Glm52DsaRequestSpan]
    k_store: Glm52DsaKStore
    needs_selection: bool
    last_selection: torch.Tensor | None = field(default=None)
    last_selection_layer: int | None = field(default=None)
