"""MiniCPM-o 4.5's prompt preprocessing: image slicing, the audio mel and the
placeholder layout, ported from the checkpoint's ``processing_minicpmo.py`` so
no Python from the checkpoint repo runs (#509).

The rendered chat marks each attachment with a tag; ``expand_tags`` swaps each
for its run of ``<unk>`` placeholder tokens, and ``placeholder_positions``
finds the runs in the tokenized prompt, as upstream's ``_convert`` does.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
from PIL import Image

from mstar.model.minicpm_o.config import AudioConfig

IMAGE_TAG = "<image>./</image>"
AUDIO_TAG = "<audio>./</audio>"

# the 4.5 checkpoint's preprocessor_config.json
SCALE_RESOLUTION = 448
MAX_SLICE_NUMS = 9
SAMPLE_RATE = 16000
# upstream encodes a clip in pieces of at most this long
MAX_AUDIO_SECONDS = 30
_MEL_HOP = 160


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


def image_placeholder(index: int, grid: tuple | None, num_queries: int) -> str:
    """An image's placeholder text: its id, the source's run, then a line of
    slice runs per grid row."""
    unk = "<unk>" * num_queries
    text = f"<image_id>{index}</image_id><image>{unk}</image>"
    if grid is not None:
        cols, rows = grid
        text += "\n".join([f"<slice>{unk}</slice>" * cols] * rows)
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


def audio_pieces(audio: np.ndarray) -> list[np.ndarray]:
    """A clip cut into the pieces upstream encodes one by one."""
    step = MAX_AUDIO_SECONDS * SAMPLE_RATE
    return [audio[i:i + step] for i in range(0, len(audio), step)] or [audio]


def audio_placeholder(num_samples: int, audio: AudioConfig) -> str:
    """A clip's placeholder text, sized off the whole clip (not its pieces)."""
    tokens = audio.pooled_tokens(math.ceil(num_samples / _MEL_HOP))
    return "<|audio_start|>" + "<unk>" * tokens + "<|audio_end|>"


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------


def expand_tags(text: str, image_placeholders: list[str], audio_placeholders: list[str]) -> str:
    """``text`` with each attachment tag replaced by its placeholder, in order."""
    images, audios = iter(image_placeholders), iter(audio_placeholders)
    chunks = re.split(f"({re.escape(IMAGE_TAG)}|{re.escape(AUDIO_TAG)})", text)
    return "".join(
        next(images) if chunk == IMAGE_TAG else next(audios) if chunk == AUDIO_TAG else chunk
        for chunk in chunks
    )


def _runs(ids: torch.Tensor, starts: list[int], ends: list[int]) -> torch.Tensor:
    """The positions strictly inside each ``start ... end`` token pair."""
    first = torch.where(torch.isin(ids, torch.tensor(starts)))[0] + 1
    last = torch.where(torch.isin(ids, torch.tensor(ends)))[0]
    if first.numel() != last.numel():
        raise ValueError("the prompt's placeholder markers are unbalanced")
    if first.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat([torch.arange(int(a), int(b)) for a, b in zip(first.tolist(), last.tolist(), strict=True)])


def prompt_ids(tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The prompt's ids (upstream drops ``<|listen|>``), and the positions of
    its image and audio placeholders."""
    listen = tokenizer.convert_tokens_to_ids("<|listen|>")
    ids = torch.tensor([t for t in tokenizer.encode(text) if t != listen], dtype=torch.long)
    tok = tokenizer.convert_tokens_to_ids
    image = _runs(ids, [tok("<image>"), tok("<slice>")], [tok("</image>"), tok("</slice>")])
    audio = _runs(ids, [tok("<|audio_start|>")], [tok("<|audio_end|>")])
    return ids, image, audio
