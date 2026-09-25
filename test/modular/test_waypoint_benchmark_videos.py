from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from mstar.client import VideoFrameChunk


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(
        str(Path(__file__).parents[1] / "waypoint" / "benchmark_streaming.py")
    )


def _chunk(fill: int, frame_index: int, width: int, height: int, frame_count: int = 4) -> VideoFrameChunk:
    data = bytes([fill]) * (frame_count * height * width * 3)
    return VideoFrameChunk(
        data,
        {
            "width": width,
            "height": height,
            "fps": 60.0,
            "pixel_format": "rgb24",
            "frame_index": frame_index,
            "frame_count": frame_count,
        },
    )


def test_encode_mp4_writes_every_frame_at_the_chunk_geometry(benchmark, tmp_path):
    av = pytest.importorskip("av")
    encode_mp4 = benchmark["_encode_mp4"]

    width, height = 4, 2
    chunks = [_chunk(1, 0, width, height), _chunk(2, 4, width, height)]

    out = tmp_path / "clip.mp4"
    frame_count, fps = encode_mp4(chunks, out)

    assert frame_count == 8
    assert fps == 60.0
    container = av.open(str(out))
    try:
        stream = container.streams.video[0]
        assert stream.width == width
        assert stream.height == height
        assert sum(1 for _ in container.decode(stream)) == 8
    finally:
        container.close()
