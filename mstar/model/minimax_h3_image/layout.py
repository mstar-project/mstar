"""One MiniMax-H3-Image request's packed sequence, ``[text | reference rows | generated
rows]``: each row's modality tag and timestep class, and the ``(t, h, w)`` rotary positions.
Ported from diffusers' ``build_image_packed_sequence``: every image is one frame on the
rotary clock, which starts where the text ends; a reference takes one integer slot each
and the generated image the slot after the last, each with its own aspect-normalized
spatial grid (H3's).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mstar.model.minimax_h3.components.layout import spatial_position_grid
from mstar.model.minimax_h3.config import TEXT_TAG, VIDEO_TAG

# Timestep classes: the generated rows (and the text rows, which no head reads) at the
# step's timestep, the references at REFERENCE_NOISE_AUG. The reference sorts a step's
# distinct timesteps, and every step's is below 0.999, so the references' entry comes second.
T_GENERATED, T_REFERENCE = 0, 1


@dataclass(frozen=True)
class H3ImageLayout:
    """What fixes a request's packed sequence, and so what two requests must share to share
    a forward (the DiT's bucket key). Latent sizes are the VAE's (before the 2x2 patchify);
    a reference's grid is cropped to whole patches."""

    text_len: int
    latent_height: int
    latent_width: int
    references: tuple[tuple[int, int], ...] = ()
    # (start, length) of each reference's vision block among the prompt's rows (tagged image)
    text_image_spans: tuple[tuple[int, int], ...] = ()
    patch: tuple[int, int] = (2, 2)
    # each reference's whole latent grid before that crop, the shape of its first noise draw
    # (``seed_loop_back``); () where it is ``references``
    reference_grids: tuple[tuple[int, int], ...] = ()

    def rows_of(self, height: int, width: int) -> int:
        return (height // self.patch[0]) * (width // self.patch[1])

    @property
    def num_condition_rows(self) -> int:
        return sum(self.rows_of(h, w) for h, w in self.references)

    @property
    def num_audio_rows(self) -> int:
        return 0

    @property
    def num_video_rows(self) -> int:
        """The generated image's rows."""
        return self.rows_of(self.latent_height, self.latent_width)

    @property
    def total_rows(self) -> int:
        return self.text_len + self.num_condition_rows + self.num_video_rows


def frame_grid(height: int, width: int, patch: tuple[int, int]) -> torch.Tensor:
    """``[rows, 2]`` float64 ``(h, w)`` coordinates of one image's rows (``_frame_position_grid``)."""
    sqrt_area = np.sqrt(height * width)
    grids = torch.meshgrid(
        torch.from_numpy(spatial_position_grid(height, patch[0], sqrt_area)),
        torch.from_numpy(spatial_position_grid(width, patch[1], sqrt_area)),
        indexing="ij",
    )
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


def position_ids(layout: H3ImageLayout) -> torch.Tensor:
    """``[S, 3]`` float64 ``(t, h, w)`` of every row."""
    text = layout.text_len
    pos = torch.zeros(layout.total_rows, 3, dtype=torch.float64)
    pos[:text, 0] = torch.arange(text, dtype=torch.float64)
    start, clock = text, float(text)
    for height, width in (*layout.references, (layout.latent_height, layout.latent_width)):
        rows = slice(start, start + layout.rows_of(height, width))
        pos[rows, 0] = clock
        pos[rows, 1:] = frame_grid(height, width, layout.patch)
        start, clock = rows.stop, clock + 1.0
    return pos


def modality_tags(layout: H3ImageLayout) -> torch.Tensor:
    """``[S]``: text rows text but for the vision blocks, every image row image (H3's video tag)."""
    tags = torch.full((layout.total_rows,), VIDEO_TAG, dtype=torch.long)
    tags[:layout.text_len] = TEXT_TAG
    for start, length in layout.text_image_spans:
        tags[start:start + length] = VIDEO_TAG
    return tags


def timestep_classes(layout: H3ImageLayout) -> torch.Tensor:
    """``[S]``: the reference rows ``T_REFERENCE``, the rest ``T_GENERATED``."""
    classes = torch.full((layout.total_rows,), T_GENERATED, dtype=torch.long)
    classes[layout.text_len:layout.text_len + layout.num_condition_rows] = T_REFERENCE
    return classes
