"""Pixel <-> latent-token layout helpers and PNG encoding shared by the image DiTs.

The conventions are the FLUX family's: a VAE latent ``[B, C, H, W]`` is
patchified 2x2 into ``[B, 4C, H/2, W/2]`` and packed to tokens ``[B, (H/2)(W/2),
4C]`` in row-major order. The inverse pair restores the latent grid for the VAE.
"""

from __future__ import annotations

import io

import torch


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W] -> [B, 4C, H/2, W/2]`` (diffusers ``_patchify_latents``)."""
    batch, channels, height, width = latents.shape
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, 4C, h, w] -> [B, C, 2h, 2w]`` (diffusers ``_unpatchify_latents``)."""
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, h, w] -> [B, h*w, C]`` tokens in row-major order."""
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``[B, h*w, C] -> [B, C, h, w]`` for tokens laid out row-major (the
    inverse of :func:`pack_latents`; equals the reference's id-scatter unpack
    for the identity layout it always emits)."""
    batch, num_tokens, channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(f"{num_tokens} tokens do not form a {height}x{width} grid")
    return tokens.permute(0, 2, 1).reshape(batch, channels, height, width)


def image_grid_ids(height: int, width: int, t: int = 0, num_axes: int = 4) -> torch.Tensor:
    """Row-major ``(t, h, w[, 0])`` position ids for an ``h x w`` latent token
    grid, ``[h*w, num_axes]`` int64 (diffusers ``_prepare_latent_ids`` /
    ``_prepare_image_ids`` with a fixed time coordinate)."""
    hh, ww = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    cols = [torch.full_like(hh, t).flatten(), hh.flatten(), ww.flatten()]
    if num_axes == 4:
        cols.append(torch.zeros(height * width, dtype=torch.int64))
    elif num_axes != 3:
        raise ValueError(f"num_axes must be 3 or 4, got {num_axes}")
    return torch.stack(cols, dim=-1)


def text_ids(num_tokens: int, num_axes: int = 4) -> torch.Tensor:
    """``(0, 0, 0, l)`` ids for ``num_tokens`` text tokens (FLUX.2 puts text
    on the fourth axis; ``[num_tokens, num_axes]`` int64)."""
    ids = torch.zeros(num_tokens, num_axes, dtype=torch.int64)
    ids[:, -1] = torch.arange(num_tokens)
    return ids


def pixels_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """VAE output in ``[-1, 1]`` -> uint8 ``[B, 3, H, W]`` with the reference
    image processor's rounding (``(x/2+0.5).clamp(0,1)``, then
    ``round(x*255)``)."""
    image = (image.to(torch.float32) / 2 + 0.5).clamp(0, 1)
    return (image * 255).round().to(torch.uint8)


def uint8_to_png(image: torch.Tensor) -> bytes:
    """Encode one uint8 ``[3, H, W]`` (or ``[1, 3, H, W]``) tensor as PNG bytes."""
    from PIL import Image

    if image.ndim == 4:
        image = image[0]
    array = image.permute(1, 2, 0).cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_pixels(image: torch.Tensor) -> torch.Tensor:
    """``[0, 1]`` float (or uint8) pixels -> ``[-1, 1]`` float, the VAE's input range."""
    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    return image.to(torch.float32) * 2.0 - 1.0
