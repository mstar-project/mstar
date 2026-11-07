"""The Cosmos3-Edge reasoner places a media prompt's 3D mRoPE positions from
its ``MEDIA_LAYOUT`` (``layout_position_ids``) instead of reading the ids. Those
must be the positions the ids-driven reference (``mrope_position_ids``, checked
against transformers by the parity tests) gives, and the counter must advance
to where decoding continues, ``max(position) + 1``."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from mstar.model.cosmos3.components.reasoner import (
    MediaGrid,
    layout_position_ids,
    media_runs,
    mrope_position_ids,
)
from mstar.model.multimodal import media_layout

IMAGE, VIDEO, MERGE = 900, 901, 2
CFG = SimpleNamespace(image_token_id=IMAGE, video_token_id=VIDEO, vision=SimpleNamespace(spatial_merge_size=MERGE))


def _prompt(rng: random.Random) -> tuple[torch.Tensor, list[MediaGrid], list[MediaGrid]]:
    """Text with images and video frames interleaved in a random order; each
    frame is its own pad run, separated by text as the chat template renders
    them, and the prompt may end on media."""
    images = [MediaGrid(1, 2 * rng.randint(1, 6), 2 * rng.randint(1, 6)) for _ in range(rng.randint(0, 2))]
    videos = [
        MediaGrid(rng.randint(1, 3), 2 * rng.randint(1, 4), 2 * rng.randint(1, 4)) for _ in range(rng.randint(0, 2))
    ]
    image_runs = [(IMAGE, g) for g in images]
    video_runs = [(VIDEO, g) for g in videos for _ in range(g.t)]
    ids: list[int] = []
    first = True
    while image_runs or video_runs:
        runs = image_runs if image_runs and (not video_runs or rng.random() < 0.5) else video_runs
        pad, grid = runs.pop(0)
        ids += [rng.randint(1, 800) for _ in range(rng.randint(0 if first else 1, 5))]
        ids += [pad] * ((grid.h // MERGE) * (grid.w // MERGE))
        first = False
    if rng.random() < 0.7 or not ids:
        ids += [rng.randint(1, 800) for _ in range(rng.randint(1, 5))]
    return torch.tensor(ids, dtype=torch.long), images, videos


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("start", [0, 17])
def test_layout_positions_are_the_reference_positions(seed, start):
    rng = random.Random(seed)
    for _ in range(100):
        ids, images, videos = _prompt(rng)
        ref, next_pos = mrope_position_ids(ids, CFG, images, videos)
        pads, grids = media_runs(ids, CFG, images, videos)
        layout = media_layout(pads, dict.fromkeys(grids, "vision_embeds"), grids)
        pos, advance = layout_position_ids(start, layout, ids.numel(), MERGE)
        assert torch.equal(pos, ref + start)
        assert advance == next_pos


def test_media_runs_refuse_a_run_its_grid_does_not_fill():
    ids = torch.tensor([5, IMAGE, IMAGE, IMAGE, 6])
    with pytest.raises(ValueError, match="does not match"):
        media_runs(ids, CFG, [MediaGrid(1, 4, 4)], [])
    with pytest.raises(ValueError, match="fewer media placeholder runs"):
        media_runs(torch.tensor([5, 6]), CFG, [MediaGrid(1, 4, 4)], [])
