"""One MiniMax-H3 request's packed sequence, ``[text | keyframe rows | audio | video]``:
each row's modality tag and timestep class, and the ``(t, h, w)`` rotary positions.
Ported from diffusers' ``before_denoise.py``. The positions are built in float64 with
numpy's arithmetic (its ``linspace``, its pairwise sum), as the reference builds them:
they feed every attention of every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mstar.model.minimax_h3.config import AUDIO_CHANNELS, AUDIO_TAG, TEXT_TAG, VIDEO_TAG

# One latent frame spans 5/3 * (1, 4, 4, 4, 4) rotary units, mirroring the VAE's
# 17-pixel-frames-to-5-latent-frames grouping; the spatial axes are normalized by the
# square root of the latent area and scaled by 32.
ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32

# Timestep classes: which of a step's timesteps conditions a row. Text rows never reach
# an output head and take the video timestep, as the reference does.
T_VIDEO, T_AUDIO, T_COND = 0, 1, 2
NUM_T_CLASSES = 3
ANCHORS = ("first", "last")


@dataclass(frozen=True)
class H3Layout:
    """What fixes a request's packed sequence, and so what two requests must share to
    share a forward (the DiT's bucket key). Latent sizes are the VAE's (before the 1x2x2
    patchify)."""

    text_len: int
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents: int              # per stereo channel
    anchors: tuple[str, ...] = ()   # one keyframe block per entry, packed in this order
    patch: tuple[int, int] = (2, 2)
    # (start, length) of each keyframe's vision block among the prompt's rows (tagged video)
    text_video_spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        bad = [a for a in self.anchors if a not in ANCHORS]
        if bad:
            raise ValueError(f"keyframe anchors must be in {ANCHORS}, got {bad}")

    @property
    def rows_per_frame(self) -> int:
        return (self.latent_height // self.patch[0]) * (self.latent_width // self.patch[1])

    @property
    def num_condition_rows(self) -> int:
        return len(self.anchors) * self.rows_per_frame

    @property
    def num_audio_rows(self) -> int:
        return self.audio_latents * AUDIO_CHANNELS

    @property
    def num_video_rows(self) -> int:
        return self.latent_frames * self.rows_per_frame

    @property
    def condition_start(self) -> int:
        return self.text_len

    @property
    def audio_start(self) -> int:
        return self.text_len + self.num_condition_rows

    @property
    def video_start(self) -> int:
        return self.audio_start + self.num_audio_rows

    @property
    def total_rows(self) -> int:
        return self.video_start + self.num_video_rows


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    """One aspect-normalized spatial axis: ``dim // patch`` coordinates centred on the unit
    interval, scaled by 32, right end excluded. numpy's ``linspace(endpoint=False)`` is
    ``start + arange(n) * (stop - start) / n``, which ``torch.linspace`` is not."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE


def temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """The rotary time of every latent frame from ``origin``: steps of ``5/3 * (1, 4, 4, 4, 4)``."""
    spans = torch.tensor(
        [ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)]
         for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def frame_position_grid(layout: H3Layout) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(h, w)`` coordinates of one latent frame's rows, and the width axis."""
    sqrt_area = np.sqrt(layout.latent_height * layout.latent_width)
    height = torch.from_numpy(spatial_position_grid(layout.latent_height, layout.patch[0], sqrt_area))
    width = torch.from_numpy(spatial_position_grid(layout.latent_width, layout.patch[1], sqrt_area))
    grids = torch.meshgrid(height, width, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1), width


def _last_anchor_time(layout: H3Layout) -> float:
    """The rotary time of the last latent frame, summed with numpy's pairwise summation as
    the reference computes this anchor (a sequential sum differs in the last ulp from 16
    latent frames on)."""
    spans = np.ones(layout.latent_frames, dtype=np.float64) * ROPE_FRAME_RESCALE
    for offset in range(len(ROPE_FRAMES_PER_LATENT)):
        spans[offset::len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[offset]
    return float(layout.text_len) + float(spans.sum()) - ROPE_FRAME_RESCALE


def position_ids(layout: H3Layout) -> torch.Tensor:
    """``[S, 3]`` float64 ``(t, h, w)`` of every row. Text rows sit on the time axis at
    their index; the media clock starts where the text ends."""
    text = layout.text_len
    pos = torch.zeros(layout.total_rows, 3, dtype=torch.float64)
    pos[:text, 0] = torch.arange(text, dtype=torch.float64)
    frame_grid, width_grid = frame_position_grid(layout)
    for i, anchor in enumerate(layout.anchors):
        start = layout.condition_start + i * layout.rows_per_frame
        rows = slice(start, start + layout.rows_per_frame)
        pos[rows, 0] = float(text) if anchor == "first" else _last_anchor_time(layout)
        pos[rows, 1:] = frame_grid
    # Audio rows share the video's clock (one unit per latent at 40/s equals 24 fps * 5/3),
    # carry no height and sit at the two ends of the width axis, one per channel.
    audio = slice(layout.audio_start, layout.video_start)
    n = layout.audio_latents
    pos[audio, 0] = (float(text) + torch.arange(n, dtype=torch.float64)).repeat(AUDIO_CHANNELS)
    pos[audio, 2] = torch.cat([
        torch.full((n,), float(width_grid[0]), dtype=torch.float64),
        torch.full((layout.num_audio_rows - n,), float(width_grid[-1]), dtype=torch.float64),
    ])
    video = torch.empty(layout.latent_frames, layout.rows_per_frame, 3, dtype=torch.float64)
    video[:, :, 0] = temporal_position_grid(layout.latent_frames, float(text))[:, None]
    video[:, :, 1:] = frame_grid[None]
    pos[layout.video_start:] = video.reshape(-1, 3)
    return pos


def modality_tags(layout: H3Layout, text_tags: torch.Tensor | None = None) -> torch.Tensor:
    """``[S]`` modality tag of every row. ``text_tags`` tags the prompt's rows; by default
    they are text but for the layout's ``text_video_spans`` (keyframe vision blocks)."""
    tags = torch.empty(layout.total_rows, dtype=torch.long)
    if text_tags is None:
        tags[:layout.text_len] = TEXT_TAG
        for start, length in layout.text_video_spans:
            tags[start:start + length] = VIDEO_TAG
    else:
        if text_tags.shape != (layout.text_len,):
            raise ValueError(f"text_tags {tuple(text_tags.shape)} do not match {layout.text_len} text rows")
        tags[:layout.text_len] = text_tags.to(torch.long)
    tags[layout.condition_start:layout.audio_start] = VIDEO_TAG
    tags[layout.audio_start:layout.video_start] = AUDIO_TAG
    tags[layout.video_start:] = VIDEO_TAG
    return tags


def timestep_classes(layout: H3Layout) -> torch.Tensor:
    """``[S]`` timestep class of every row: keyframe rows ``T_COND``, audio ``T_AUDIO``, the
    rest (text, generated video) ``T_VIDEO``."""
    classes = torch.full((layout.total_rows,), T_VIDEO, dtype=torch.long)
    classes[layout.condition_start:layout.audio_start] = T_COND
    classes[layout.audio_start:layout.video_start] = T_AUDIO
    return classes


def patchify_video(latents: torch.Tensor, patch: tuple[int, int]) -> torch.Tensor:
    """``[C, F, H, W] -> [F * H/ph * W/pw, C * ph * pw]``, frame-major then row-major
    (``patchify_video_latents`` at a temporal patch of 1)."""
    c, f, h, w = latents.shape
    ph, pw = patch
    x = latents.reshape(c, f, h // ph, ph, w // pw, pw).permute(1, 2, 4, 0, 3, 5)
    return x.reshape(-1, c * ph * pw).contiguous()


def unpatchify_video(rows: torch.Tensor, layout: H3Layout, channels: int) -> torch.Tensor:
    """The inverse of :func:`patchify_video`, batched: ``[B, rows, C*ph*pw] -> [B, C, F, H, W]``."""
    ph, pw = layout.patch
    b = rows.shape[0]
    x = rows.reshape(b, layout.latent_frames, layout.latent_height // ph, layout.latent_width // pw, channels, ph, pw)
    x = x.permute(0, 4, 1, 2, 5, 3, 6)
    return x.reshape(b, channels, layout.latent_frames, layout.latent_height, layout.latent_width).contiguous()
