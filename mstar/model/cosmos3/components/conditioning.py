"""Conditioning-frame preprocessing for image-to-video.

Two references disagree on how a conditioning image reaches the target
resolution, and both are served:

* ``stretch`` — the diffusers 0.39 pipeline (``VideoProcessor.preprocess``):
  a plain bilinear resize to ``(height, width)``, aspect ratio not preserved,
  values mapped to [-1, 1] without 8-bit rounding. What the Nano checkpoints
  were validated against.
* ``aspect_crop`` — the diffusers 0.40 pipeline (``_preprocess_conditioning_image``)
  and the vLLM-Omni recipe: scale so the image covers the target
  (``max(width / w, height / h)``), antialiased bilinear resize to the ceiled
  size, center crop, round to 8-bit, then ``x / 127.5 - 1``. The Edge model
  card's recipe.

``prepare_conditioning_frames`` takes the frames the data worker loaded
(``[T, C, H, W]`` in [0, 1], or a single ``[C, H, W]``) and returns
``[1, 3, T, height, width]`` in [-1, 1] for the VAE encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

CONDITIONING_RESIZE_MODES = ("stretch", "aspect_crop")


def _to_float_0_255(frames: torch.Tensor) -> torch.Tensor:
    if frames.dtype == torch.uint8:
        return frames.float()
    frames = frames.float()
    if frames.numel() and frames.min() < 0:
        return (frames + 1.0) * 127.5
    if frames.numel() and frames.max() <= 1.0:
        return frames * 255.0
    return frames


def prepare_conditioning_frames(
    frames: torch.Tensor, height: int, width: int, mode: str = "stretch",
) -> torch.Tensor:
    """Resize conditioning pixels to the generation size; see the module docstring."""
    if mode not in CONDITIONING_RESIZE_MODES:
        raise ValueError(f"conditioning_resize must be one of {CONDITIONING_RESIZE_MODES}, got {mode!r}")
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError(f"expected [T, C, H, W] or [C, H, W] frames, got {tuple(frames.shape)}")
    if frames.shape[1] == 1:
        frames = frames.expand(-1, 3, -1, -1)
    elif frames.shape[1] == 4:
        frames = frames[:, :3]
    x = _to_float_0_255(frames)
    if mode == "stretch":
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        x = (x / 255.0) * 2.0 - 1.0
    else:
        src_h, src_w = x.shape[-2:]
        scale = max(width / src_w, height / src_h)
        rh, rw = math.ceil(scale * src_h), math.ceil(scale * src_w)
        x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False, antialias=True)
        top = round((rh - height) / 2)
        left = round((rw - width) / 2)
        x = x[:, :, top:top + height, left:left + width]
        x = x.round().clamp(0, 255) / 127.5 - 1.0
    # [T, 3, H, W] -> [1, 3, T, H, W]
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
