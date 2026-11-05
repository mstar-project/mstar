"""3D positions for a packed MRoPE step (Qwen3.5, Qwen3-Omni's Thinker)."""

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
