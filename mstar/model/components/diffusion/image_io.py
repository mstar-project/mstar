"""Pixel <-> latent-token layout helpers and PNG encoding shared by the image DiTs.

The conventions are the FLUX family's: a VAE latent ``[B, C, H, W]`` is
patchified 2x2 into ``[B, 4C, H/2, W/2]`` and packed to tokens ``[B, (H/2)(W/2),
4C]`` in row-major order. The inverse pair restores the latent grid for the VAE.
"""

from __future__ import annotations

import io
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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
    """VAE output in ``[-1, 1]`` -> uint8 ``[B, 3, H, W]`` with the reference image
    processor's rounding: ``(x * 0.5 + 0.5).clamp(0, 1)`` in the VAE's own dtype
    (``denormalize``), then ``float()`` and ``round(x * 255)`` (``pt_to_numpy`` +
    ``numpy_to_pil``). The order matters in bf16, whose spacing in ``[0.5, 1)`` is
    ``1/256``: denormalizing after an fp32 upcast moves one pixel in five by a level."""
    image = (image * 0.5 + 0.5).clamp(0, 1).to(torch.float32)
    return (image * 255).round().to(torch.uint8)


PNG_WORKERS = min(8, os.cpu_count() or 1)


def uint8_to_png(
    image: torch.Tensor, compress_level: int = 1, adaptive_filters: bool = False, workers: int = PNG_WORKERS,
) -> bytes:
    """Encode one uint8 ``[3, H, W]`` (or ``[1, 3, H, W]``) tensor as PNG bytes, losslessly.

    The default writer emits every scanline with PNG filter 0 and deflates the image in
    ``workers`` independent blocks stitched into one zlib stream (pigz-style: each block ends
    on a byte boundary with a sync flush, one adler32 over the whole image). At level 1 on
    8 threads a 1024x1024 image encodes in ~15 ms against ~440 ms for PIL's default
    (adaptive filters, zlib 6), which was half of a served request, at ~30% larger files.
    ``adaptive_filters=True`` hands the encode to PIL at the same zlib level for the smaller
    file when latency matters less. Every variant decodes to the same pixels.
    """
    if image.ndim == 4:
        image = image[0]
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a uint8 [3, H, W] image, got {tuple(image.shape)} {image.dtype}")
    rows = image.permute(1, 2, 0).contiguous().cpu().numpy()  # [H, W, 3]
    if adaptive_filters:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rows, mode="RGB").save(buffer, format="PNG", compress_level=compress_level)
        return buffer.getvalue()
    height, width, channels = rows.shape
    raw = np.empty((height, 1 + width * channels), dtype=np.uint8)
    raw[:, 0] = 0  # filter type "None" for every scanline
    raw[:, 1:] = rows.reshape(height, width * channels)
    return b"".join((
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),  # 8-bit RGB
        _png_chunk(b"IDAT", _zlib_parallel(raw.tobytes(), compress_level, workers)),
        _png_chunk(b"IEND", b""),
    ))


def _zlib_parallel(data: bytes, level: int, workers: int, min_block: int = 1 << 18) -> bytes:
    """One valid zlib stream from ``workers`` independently deflated blocks.

    Raw deflate blocks ended with ``Z_SYNC_FLUSH`` finish on a byte boundary, so their
    concatenation is one deflate stream once the last block carries the final bit; zlib
    releases the GIL, so the blocks compress in parallel. Small inputs take the plain path.
    """
    blocks = max(1, min(workers, len(data) // min_block))
    if blocks == 1:
        return zlib.compress(data, level)
    size = -(-len(data) // blocks)
    parts = [data[i * size:(i + 1) * size] for i in range(blocks)]

    def deflate(index: int, part: bytes) -> bytes:
        compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
        out = compressor.compress(part)
        return out + compressor.flush(zlib.Z_FINISH if index == blocks - 1 else zlib.Z_SYNC_FLUSH)

    with ThreadPoolExecutor(max_workers=blocks) as pool:
        deflated = list(pool.map(deflate, range(blocks), parts))
    # zlib header: deflate, 32K window, no preset dictionary, "fastest" level hint (FCHECK ok)
    return b"\x78\x01" + b"".join(deflated) + struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF)


OUTPUT_FORMATS = ("png", "jpeg", "webp")


def encode_image(image: torch.Tensor, request_kwargs: dict | None = None) -> bytes:
    """Encode a uint8 ``[3, H, W]`` image for the client per the OpenAI images knobs.

    ``output_format`` (``png`` default | ``jpeg`` | ``webp``) and ``output_compression``
    (0-100; the JPEG / WebP quality, default 95 / 80) come from the request; for PNG,
    ``png_compress_level`` (0-9, default 1) selects the zlib level of the fast writer.
    Everything else decodes to the pixels the VAE produced (PNG losslessly).
    """
    kwargs = request_kwargs or {}
    fmt = str(kwargs.get("output_format") or "png").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {fmt!r}")
    if fmt == "png":
        return uint8_to_png(image, compress_level=int(kwargs.get("png_compress_level", 1)))
    from PIL import Image

    if image.ndim == 4:
        image = image[0]
    quality = kwargs.get("output_compression")
    quality = int(quality) if quality is not None else (95 if fmt == "jpeg" else 80)
    buffer = io.BytesIO()
    Image.fromarray(image.permute(1, 2, 0).contiguous().cpu().numpy(), mode="RGB").save(
        buffer, format=fmt.upper(), quality=max(0, min(100, quality)),
    )
    return buffer.getvalue()


def _png_chunk(tag: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)


def normalize_pixels(image: torch.Tensor) -> torch.Tensor:
    """``[0, 1]`` float (or uint8) pixels -> ``[-1, 1]`` float, the VAE's input range."""
    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    return image.to(torch.float32) * 2.0 - 1.0
