"""MiniCPM-o 4.5's prompt preprocessing: image slicing, audio pieces and the
placeholder layout, ported from the checkpoint's ``processing_minicpmo.py`` so
no Python from the checkpoint repo runs (#509).

The rendered chat marks each attachment with a tag; ``expand_tags`` swaps each
for its placeholder text: upstream's, with one ``<unk>`` for each run of them
(an image slice, an audio clip), which intake expands to the run's tokens.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
from PIL import Image

from mstar.model.components.audio_features import HOP_LENGTH, SAMPLE_RATE
from mstar.model.minicpm_o.config import AudioConfig

IMAGE_TAG = "<image>./</image>"
AUDIO_TAG = "<audio>./</audio>"

# the 4.5 checkpoint's preprocessor_config.json
SCALE_RESOLUTION = 448
MAX_SLICE_NUMS = 9
# upstream encodes a clip in pieces of at most this long
MAX_AUDIO_SECONDS = 30


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------


def _ensure_divide(length: float, patch_size: int) -> int:
    return max(round(length / patch_size) * patch_size, patch_size)


def _best_resize(size: tuple[float, float], patch_size: int, allow_upscale: bool = False) -> tuple[int, int]:
    width, height = size
    if width * height > SCALE_RESOLUTION * SCALE_RESOLUTION or allow_upscale:
        r = width / height
        height = int(SCALE_RESOLUTION / math.sqrt(r))
        width = int(height * r)
    return _ensure_divide(width, patch_size), _ensure_divide(height, patch_size)


def slice_grid(size: tuple[int, int], max_slice_nums: int) -> tuple[int, int] | None:
    """The ``(cols, rows)`` grid an image of ``size`` is cut into, or None
    when it is small enough to go whole."""
    width, height = size
    multiple = min(math.ceil(width * height / (SCALE_RESOLUTION * SCALE_RESOLUTION)), max_slice_nums)
    if multiple <= 1:
        return None
    log_ratio = math.log(width / height)
    best, min_error = (1, 1), float("inf")
    for n in (multiple - 1, multiple, multiple + 1):
        if n == 1 or n > max_slice_nums:
            continue
        for cols in range(1, n + 1):
            if n % cols == 0:
                error = abs(log_ratio - math.log(cols / (n // cols)))
                if error < min_error:
                    best, min_error = (cols, n // cols), error
    return best


def slice_image(image: Image.Image, max_slice_nums: int, patch_size: int) -> tuple[list[Image.Image], tuple | None]:
    """The source image resized to whole patches, then its grid's crops in
    row-major order; and the grid."""
    grid = slice_grid(image.size, max_slice_nums)
    if grid is None:
        size = _best_resize(image.size, patch_size, allow_upscale=True)
        return [image.resize(size, resample=Image.Resampling.BICUBIC)], None
    source = image.resize(_best_resize(image.size, patch_size), resample=Image.Resampling.BICUBIC)
    cols, rows = grid
    width, height = image.size
    cell = _best_resize(
        (_ensure_divide(width, cols) / cols, _ensure_divide(height, rows) / rows), patch_size, allow_upscale=True,
    )
    refined = image.resize((cell[0] * cols, cell[1] * rows), resample=Image.Resampling.BICUBIC)
    crops = [
        refined.crop((c * cell[0], r * cell[1], (c + 1) * cell[0], (r + 1) * cell[1]))
        for r in range(rows) for c in range(cols)
    ]
    return [source, *crops], grid


def image_runs(grid: tuple | None) -> int:
    """The ``<unk>`` runs of an image cut into ``grid``: the source's, then a
    slice's each."""
    return 1 if grid is None else 1 + grid[0] * grid[1]


def image_placeholder(index: int, grid: tuple | None) -> str:
    """An image's placeholder text: its id, the source's run, then a line of
    slice runs per grid row, each run one ``<unk>``."""
    text = f"<image_id>{index}</image_id><image><unk></image>"
    if grid is not None:
        cols, rows = grid
        text += "\n".join(["<slice><unk></slice>" * cols] * rows)
    return text


def slice_patches(image: Image.Image, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """A slice's normalized ``[n, 3, p, p]`` patches in row-major order, and
    its ``(h, w)`` in patches."""
    pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255
    pixels = (pixels - 0.5) / 0.5
    p = patch_size
    h, w = pixels.shape[1] // p, pixels.shape[2] // p
    patches = pixels.view(3, h, p, w, p).permute(1, 3, 0, 2, 4).reshape(h * w, 3, p, p)
    return patches, (h, w)


# ----------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------


def audio_pieces(audio: torch.Tensor) -> list[torch.Tensor]:
    """A clip cut into the pieces upstream encodes one by one. Each clip is
    its own: upstream merges the clips of one message into one, but sizes
    their placeholders clip by clip, and the two round differently (upstream's
    own `chat` fails on such prompts)."""
    step = MAX_AUDIO_SECONDS * SAMPLE_RATE
    return [audio[i:i + step] for i in range(0, len(audio), step)] or [audio]


def audio_tokens(num_samples: int, audio: AudioConfig) -> int:
    """The tokens upstream sizes a clip's placeholder at: off the whole clip,
    not its pieces."""
    return audio.pooled_tokens(math.ceil(num_samples / HOP_LENGTH))


def audio_placeholder(runs: int) -> str:
    """A clip's placeholder text: its run's one ``<unk>``, or none for a clip
    too short to encode."""
    return "<|audio_start|>" + "<unk>" * runs + "<|audio_end|>"


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------


def expand_tags(text: str, image_placeholders: list[str], audio_placeholders: list[str]) -> str:
    """``text`` with each attachment tag replaced by its placeholder, in order."""
    chunks = re.split(f"({re.escape(IMAGE_TAG)}|{re.escape(AUDIO_TAG)})", text)
    # a tag typed into the prompt or system prompt leaves a tag with no attachment
    for tag, placeholders in ((IMAGE_TAG, image_placeholders), (AUDIO_TAG, audio_placeholders)):
        if (n := chunks.count(tag)) != len(placeholders):
            raise ValueError(f"the prompt contains {n} {tag} attachment tags for {len(placeholders)} attachments")
    images, audios = iter(image_placeholders), iter(audio_placeholders)
    return "".join(
        next(images) if chunk == IMAGE_TAG else next(audios) if chunk == AUDIO_TAG else chunk
        for chunk in chunks
    )
