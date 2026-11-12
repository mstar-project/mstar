"""A Qwen-VL prompt's images, as the scaffold's Qwen encoder takes them.

transformers' ``Qwen3VLModel`` / ``Qwen2_5_VLModel`` splice the vision tower's merged tokens
into the token embeddings at the ``<|image_pad|>`` rows and position the prompt with 3-D
MRoPE (``get_rope_index``: text advances all three grids, an image holds T and sweeps H and
W and moves the cursor by ``max(h', w')``); Qwen3-VL also adds the tower's deepstack features
at the image rows after the first decoder layers. The encoder does all three from the tower's
output (``VisionTokens``); this module holds that output's type and the positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch

from mstar.model.components.mrope import image_position_advance, image_position_ids, text_position_ids


class VisionTokens(NamedTuple):
    """The vision tower's output for one prompt's images, every image end to end."""

    grids: Sequence[tuple[int, int, int]]   # each image's (t, h, w) patch grid, in prompt order
    merge: int                              # the tower's spatial merge size
    embeds: torch.Tensor                    # [N, hidden] merged tokens
    deepstack: Sequence[torch.Tensor] = ()  # [N, hidden] per deepstack layer (Qwen3-VL), else none


def vl_positions(
    token_ids: torch.Tensor, image_token_id: int, grids: Sequence[tuple[int, int, int]], merge: int,
) -> torch.Tensor:
    """``[3, L]`` 3-D MRoPE positions (``get_rope_index`` for one prompt); a run of image
    pads takes the next grid, everything else is text."""
    is_image = (token_ids == image_token_id).tolist()
    parts, pos, start, grid_iter = [], 0, 0, iter(grids)
    while start < len(is_image):
        end = start
        while end < len(is_image) and is_image[end] == is_image[start]:
            end += 1
        if is_image[start]:
            grid = torch.tensor(next(grid_iter))
            ids = image_position_ids(grid, merge, pos)
            if ids.shape[1] != end - start:
                raise ValueError(f"an image run of {end - start} pads does not match its grid {tuple(grid.tolist())}")
            parts.append(ids)
            pos += image_position_advance(grid, merge)
        else:
            parts.append(text_position_ids(end - start, pos))
            pos += end - start
        start = end
    if next(grid_iter, None) is not None:
        raise ValueError("more image grids than image-pad runs in the prompt")
    return torch.cat(parts, dim=1).to(token_ids.device)
