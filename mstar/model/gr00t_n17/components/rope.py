"""Where each of a request's tokens sits on M-RoPE's three (T, H, W) grids.

transformers builds these in ``Qwen3VLModel.get_rope_index`` while scanning the
prompt; we build them from the same two inputs, reusing ``qwen3_5``'s helpers
for the two kinds of span. A text span advances all three grids together; an
image span holds T and sweeps H and W from the span's base, and whatever comes
next starts one past the block's largest index.

The LLM runs one request per ragged segment, so these are one unpadded
request's tokens -- there is no padding to mask out.

Images are located by their runs of ``image_token_id``, not by the
``vision_start`` marker transformers 4.57 looks for. On every prompt our
processor builds the two agree; on a marker-less one they do not, and runs are
the more robust of the two (4.57 mis-positions such a prompt silently).
"""
from __future__ import annotations

import torch

from mstar.model.qwen3_5.components.rope import (
    text_position_ids,
    vision_position_advance,
    vision_position_ids,
)


def mrope_position_ids(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    spatial_merge_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``[3, len(input_ids)]``, float, in the order ``compute_3d_cos_sin`` wants."""
    # the LLM node prepares one ragged segment at a time; a batch flattened
    # here would run the second request's positions on from the first's
    if input_ids.dim() > 1 and input_ids.shape[0] != 1:
        raise ValueError(
            f"one request at a time, got a batch of {input_ids.shape[0]}"
        )
    tokens = input_ids.reshape(-1).tolist()
    grids = list(image_grid_thw.reshape(-1, 3))
    spans: list[torch.Tensor] = []
    cursor, start, image = 0.0, 0, 0

    while start < len(tokens):
        is_image = tokens[start] == image_token_id
        end = start
        while end < len(tokens) and (tokens[end] == image_token_id) == is_image:
            end += 1

        if not is_image:
            spans.append(text_position_ids(end - start, cursor, device))
            cursor += end - start
            start = end
            continue

        if image >= len(grids):
            raise ValueError(
                f"{len(grids)} image grid(s) for more image span(s) than that"
            )
        grid = grids[image]
        t, h, w = (int(v) for v in grid.tolist())
        # Refused, not implemented: the two transformers versions disagree on
        # what a t > 1 image does to the cursor (4.57 advances by
        # max(t, h', w'), 5.19 ignores t), so there is no one answer to match.
        # Videos are the only t > 1 source and both versions split them to one
        # grid per frame before any position is computed, so a t > 1 image
        # grid means the caller built it wrong.
        if t != 1:
            raise ValueError(f"image grid {t, h, w} has t != 1")
        merged = (h // spatial_merge_size) * (w // spatial_merge_size)
        if end - start != merged:
            raise ValueError(
                f"{end - start} image tokens for a grid worth {merged}"
            )
        spans.append(vision_position_ids(grid, spatial_merge_size, cursor, device))
        cursor += vision_position_advance(grid, spatial_merge_size)
        image += 1
        start = end

    if image != len(grids):
        raise ValueError(f"{len(grids)} image grid(s) for {image} image span(s)")
    return torch.cat(spans, dim=1)
