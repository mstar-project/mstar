"""Image views and placeholder counts, as the model card's ``infer`` builds them.

PIL does the resizing and padding so the pixels match the reference's exactly;
a torch resize differs in its filter taps and moves the OCR output.
"""

from typing import NamedTuple

import numpy as np
import torch
from PIL import Image, ImageOps

from mstar.model.deepseek_ocr.config import ImageMode

# ``ImageOps.pad``'s fill: the normalization mean, 0.5, in bytes
_PAD_COLOR = (127, 127, 127)


class ImageViews(NamedTuple):
    # (3, global_size, global_size): the padded whole image
    global_view: torch.Tensor
    # (rows * cols, 3, image_size, image_size) row-major tiles, or None
    crops: torch.Tensor | None
    cols: int
    rows: int


def _closest_grid(aspect: float, width: int, height: int, mode: ImageMode) -> tuple[int, int]:
    """``find_closest_aspect_ratio`` over every grid of min..max crops, ties
    broken toward the larger grid when the image covers half of it."""
    grids = sorted(
        {(i, j) for n in range(mode.min_crops, mode.max_crops + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if mode.min_crops <= i * j <= mode.max_crops},
        key=lambda g: g[0] * g[1],
    )
    best, best_diff, area = (1, 1), float("inf"), width * height
    for cols, rows in grids:
        diff = abs(aspect - cols / rows)
        if diff < best_diff:
            best, best_diff = (cols, rows), diff
        elif diff == best_diff and area > 0.5 * mode.image_size ** 2 * cols * rows:
            best = (cols, rows)
    return best


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """ToTensor + Normalize(0.5, 0.5) in float32, then bfloat16, as the reference."""
    array = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    return array.float().div_(255.0).sub_(0.5).div_(0.5).to(torch.bfloat16)


def image_views(image: Image.Image, mode: ImageMode) -> ImageViews:
    image = image.convert("RGB")
    width, height = image.size
    if not mode.crop_mode:
        if mode.image_size <= 640:
            image = image.resize((mode.image_size, mode.image_size))
        view = ImageOps.pad(image, (mode.image_size, mode.image_size), color=_PAD_COLOR)
        return ImageViews(_to_tensor(view), None, 1, 1)
    cols = rows = 1
    tiles = None
    if width > mode.image_size or height > mode.image_size:
        cols, rows = _closest_grid(width / height, width, height, mode)
        size = mode.image_size
        resized = image.resize((size * cols, size * rows))
        tiles = torch.stack([
            _to_tensor(resized.crop((c * size, r * size, (c + 1) * size, (r + 1) * size)))
            for r in range(rows) for c in range(cols)
        ])
    view = ImageOps.pad(image, (mode.base_size, mode.base_size), color=_PAD_COLOR)
    if cols * rows == 1:
        # a 1x1 grid has no local view: the reference drops its tile
        tiles = None
    return ImageViews(_to_tensor(view), tiles, cols, rows)


def num_image_tokens(mode: ImageMode, cols: int, rows: int) -> int:
    """Placeholders for one image: the global view's rows each end in a newline
    column, then one separator, then the tiled local view's rows likewise."""
    side = mode.side_tokens(mode.global_size)
    count = side * (side + 1) + 1
    if mode.crop_mode and cols * rows > 1:
        tile = mode.side_tokens(mode.image_size)
        count += (tile * rows) * (tile * cols + 1)
    return count


def max_image_tokens(mode: ImageMode) -> int:
    """Placeholders of the largest image ``mode`` can cut: any tile grid it may pick, or none."""
    if not mode.crop_mode:
        return num_image_tokens(mode, 1, 1)
    grids = [(cols, rows) for cols in range(1, mode.max_crops + 1) for rows in range(1, mode.max_crops + 1)
             if mode.min_crops <= cols * rows <= mode.max_crops]
    return max(num_image_tokens(mode, cols, rows) for cols, rows in [(1, 1), *grids])


def to_pil(image: torch.Tensor) -> Image.Image:
    """A ``(C, H, W)`` uint8 image as PIL RGB."""
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) uint8 image, got {tuple(image.shape)} {image.dtype}")
    return Image.fromarray(image.permute(1, 2, 0).contiguous().cpu().numpy())
