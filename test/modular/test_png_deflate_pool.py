"""PNG deflate on the process's shared pool: the bytes (so the pixels) are the ones the
per-call threads wrote, concurrent encodes share the pool, and the pool is sized from
the CPU plan, whose CPUs come from the cgroup quota rather than ``os.cpu_count()``."""

from __future__ import annotations

import io
import struct
import threading
import zlib

import numpy as np
import pytest
import torch

from mstar.model.components.diffusion import image_io
from mstar.utils import cpus
from mstar.utils.threads import ThreadPool


def _old_zlib_parallel(data: bytes, level: int, workers: int, min_block: int = 1 << 18) -> bytes:
    """The writer before the shared pool: bytes slices on threads started per call."""
    blocks = max(1, min(workers, len(data) // min_block))
    if blocks == 1:
        return zlib.compress(data, level)
    size = -(-len(data) // blocks)
    parts = [data[i * size:(i + 1) * size] for i in range(blocks)]

    def deflate(index: int, part: bytes) -> bytes:
        compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
        out = compressor.compress(part)
        return out + compressor.flush(zlib.Z_FINISH if index == blocks - 1 else zlib.Z_SYNC_FLUSH)

    with ThreadPool(max_workers=blocks) as pool:
        deflated = list(pool.map(deflate, range(blocks), parts))
    return b"\x78\x01" + b"".join(deflated) + struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF)


def _old_png(image: torch.Tensor, level: int, workers: int) -> bytes:
    rows = image.permute(1, 2, 0).contiguous().numpy()
    height, width, channels = rows.shape
    raw = np.empty((height, 1 + width * channels), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = rows.reshape(height, width * channels)
    return b"".join((
        b"\x89PNG\r\n\x1a\n",
        image_io._png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        image_io._png_chunk(b"IDAT", _old_zlib_parallel(raw.tobytes(), level, workers)),
        image_io._png_chunk(b"IEND", b""),
    ))


def _image(seed: int, height: int = 384, width: int = 512) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    image = torch.randint(0, 256, (3, height, width), dtype=torch.uint8, generator=gen)
    image[:, : height // 3] = image[:, : height // 3] // 64 * 64  # flatter rows deflate differently
    return image


def _pixels(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


@pytest.mark.parametrize(("level", "workers"), [(1, 8), (1, 3), (0, 8), (6, 2), (1, 1)])
def test_the_shared_pool_writes_the_old_bytes(level, workers):
    image = _image(level * 10 + workers)
    new = image_io.uint8_to_png(image, compress_level=level, workers=workers)
    assert new == _old_png(image, level, workers)
    np.testing.assert_array_equal(_pixels(new), image.permute(1, 2, 0).numpy())


def test_images_encoded_at_once_share_the_pool_and_keep_their_bytes():
    images = [_image(seed) for seed in range(6)]
    expected = [_old_png(image, 1, 4) for image in images]
    out: list[bytes | None] = [None] * len(images)

    def encode(i: int) -> None:
        out[i] = image_io.uint8_to_png(images[i], workers=4)

    threads = [threading.Thread(target=encode, args=(i,)) for i in range(len(images))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert out == expected


def test_the_deflate_pool_is_sized_from_the_cgroup_quota(monkeypatch):
    monkeypatch.setattr(cpus.os, "cpu_count", lambda: 192)
    monkeypatch.setattr(cpus.os, "sched_getaffinity", lambda pid: set(range(192)))
    monkeypatch.setattr(cpus, "cgroup_cpu_quota", lambda: 5.5)
    monkeypatch.setattr(cpus, "_plan", None)
    monkeypatch.setattr(image_io, "_deflate_threads", None)
    try:
        assert image_io._deflate_pool()._max_workers == cpus.plan().png_threads == 5
    finally:
        image_io._deflate_pool().shutdown(wait=True)
