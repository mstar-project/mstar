"""3D (temporal, height, width) MRoPE positions: a media prompt's own, from
its intake layout, and a packed step's (Qwen3.5, Qwen3-Omni's Thinker, the
Cosmos3 reasoner)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.model.submodule_base import ARNodeInputs


def mrope_position_ids(pos_ids: torch.Tensor | None, inputs: list[ARNodeInputs]) -> torch.Tensor:
    """``[3, total_tokens]`` for the step, in packed request order.

    A pure-text step is ``pos_ids``, the position resource's planned 1D
    positions, broadcast across the three grids (a view, no copy). An image's
    grids do not advance together, so a request with `custom_pos_ids` supplies
    its own and the step concatenates, using the slice of ``pos_ids`` for text.
    """
    text_rows = [inp.custom_pos_ids is None for inp in inputs]
    assert pos_ids is not None or not any(text_rows), (
        "position resource has no plan for this step; `plan` must run before `preprocess`"
    )
    if all(text_rows):
        return pos_ids[:sum(inp.input_seq_len for inp in inputs)].unsqueeze(0).expand(3, -1)

    parts: list[torch.Tensor] = []
    offset = 0
    for inp in inputs:
        span = inp.input_seq_len
        if inp.custom_pos_ids is not None:
            parts.append(inp.custom_pos_ids.float())
        else:
            parts.append(pos_ids[offset:offset + span].float().unsqueeze(0).expand(3, -1))
        offset += span
    return torch.cat(parts, dim=1)


def layout_position_ids(
    layout: dict, merge_size: int, num_tokens: int, start: int = 0, device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """``[3, num_tokens]`` MRoPE ids of a prompt whose media runs ``layout``
    places (``offset``, ``tokens`` and a ``(t, h, w)`` patch ``grid`` per run),
    counted from ``start``, and how far the prompt moves the position cursor.

    Text takes one position a token on all three axes. A media run spans
    ``max(h', w')`` positions (``h' = h // merge_size``) but ``t * h' * w'``
    tokens: T is constant across a frame while H and W sweep the merged grid,
    row-major, all based at the cursor. Built on ``device`` from host values.
    """
    pieces: list[torch.Tensor] = []
    pos, cursor = start, 0

    def axis(n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.float, device=device) + pos

    for offset, tokens, (t, h, w) in zip(
        layout.get("offset", ()), layout.get("tokens", ()), layout.get("grid", ()), strict=True,
    ):
        if offset > cursor:
            pieces.append(axis(offset - cursor).expand(3, -1))
            pos += offset - cursor
        h, w = h // merge_size, w // merge_size
        pieces.append(torch.stack(torch.meshgrid(axis(t), axis(h), axis(w), indexing="ij")).reshape(3, -1))
        pos += max(h, w)
        cursor = offset + tokens
    if num_tokens > cursor:
        pieces.append(axis(num_tokens - cursor).expand(3, -1))
        pos += num_tokens - cursor
    return torch.cat(pieces, dim=1), pos - start


def layout_extent(layout: dict, merge_size: int, length: int) -> int:
    """How far a prompt of ``length`` tokens whose media runs ``layout`` places
    moves the position cursor (``layout_position_ids``' advance), for its KV
    span."""
    return length + sum(
        max(h, w) // merge_size - tokens
        for tokens, (_, h, w) in zip(layout.get("tokens", ()), layout.get("grid", ()), strict=True)
    )
