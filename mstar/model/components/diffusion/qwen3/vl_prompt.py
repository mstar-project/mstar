"""A Qwen3-VL prompt with images, as the scaffold's Qwen3 encoder takes it.

transformers' ``Qwen3VLModel.forward`` splices the vision tower's merged tokens into the
token embeddings at the ``<|image_pad|>`` rows, positions the prompt with 3-D MRoPE
(``get_rope_index``: text advances all three grids, an image holds T and sweeps H and W
and moves the cursor by ``max(h', w')``), and adds the tower's deepstack features at the
image rows after the first decoder layers. ``qwen3_vl_prompt`` builds the encoder's three
inputs for that; a prompt without images reduces to the encoder's text path.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from mstar.model.components.mrope import (
    compute_3d_cos_sin,
    image_position_advance,
    image_position_ids,
    text_position_ids,
)


class VLPrompt(NamedTuple):
    inputs_embeds: torch.Tensor                   # [1, L, hidden]
    rotary: tuple[torch.Tensor, torch.Tensor]     # fp32 cos, sin [L, head_dim]
    deepstack: tuple[torch.Tensor, list[torch.Tensor]] | None  # image rows [N], features [N, hidden] per layer


def qwen3_vl_positions(
    token_ids: torch.Tensor, image_token_id: int, grids: list[tuple[int, int, int]], merge: int,
) -> torch.Tensor:
    """``[3, L]`` 3-D MRoPE positions (``Qwen3VLModel.get_rope_index`` for one prompt); a
    run of image pads takes the next grid, everything else is text."""
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


def qwen3_vl_prompt(
    token_ids: torch.Tensor,
    image_token_id: int,
    grids: list[tuple[int, int, int]],
    merge: int,
    image_embeds: torch.Tensor,
    deepstack: list[torch.Tensor],
    embed_tokens: nn.Embedding,
    inv_freq: torch.Tensor,
    mrope_section: list[int],
) -> VLPrompt:
    """``token_ids [L]`` with each image's ``<|image_pad|>`` run, and the tower's merged
    tokens (``image_embeds [N, hidden]``, every image end to end) and deepstack features."""
    embeds = embed_tokens(token_ids)
    rows = (token_ids == image_token_id).nonzero().squeeze(-1)
    if rows.numel() != image_embeds.shape[0]:
        raise ValueError(f"{rows.numel()} image pads for {image_embeds.shape[0]} image tokens")
    embeds = embeds.index_copy(0, rows, image_embeds.to(embeds.dtype))
    positions = qwen3_vl_positions(token_ids, image_token_id, grids, merge)
    rotary = compute_3d_cos_sin(positions, inv_freq.to(token_ids.device), mrope_section)
    return VLPrompt(embeds[None], rotary, (rows, deepstack) if deepstack else None)
