"""``mux_rgb_frames`` converts frames ahead of the encoder on a thread of its own; the
mp4 bytes are the ones the plain ``stream.encode`` loop writes, for each setting the
models encode with."""

from __future__ import annotations

import io
from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from mstar.model.components.video_encode import mux_rgb_frames  # noqa: E402


def _mux(frames: np.ndarray, options: dict | None, threaded: bool) -> bytes:
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=Fraction(24))
    stream.height, stream.width = frames.shape[1], frames.shape[2]
    stream.pix_fmt = "yuv420p"
    if options:
        stream.options = options
    if threaded:
        mux_rgb_frames(container, stream, frames)
    else:
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    container.close()
    return buffer.getvalue()


def _clip(frames: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(frames * height + width)
    # a moving gradient plus noise, so the encoder has motion to search
    y, x = np.mgrid[:height, :width]
    clip = np.stack([(x + 3 * t + y // 2) % 256 for t in range(frames)])[..., None].repeat(3, -1)
    noise = rng.integers(0, 24, size=clip.shape)
    return ((clip + noise) % 256).astype(np.uint8)


@pytest.mark.parametrize("options", [
    None,  # Wan2.2: libx264's defaults
    {"crf": "18", "preset": "ultrafast", "threads": "0"},  # LTX-2.5, Cosmos3's PyAV path
])
@pytest.mark.parametrize(("frames", "height", "width"), [(1, 16, 16), (9, 48, 64), (17, 66, 98)])
def test_the_threaded_mux_writes_the_plain_loops_bytes(options, frames, height, width):
    clip = _clip(frames, height, width)
    assert _mux(clip, options, threaded=True) == _mux(clip, options, threaded=False)


def test_frames_of_another_size_scale_as_the_encoder_would_scale_them():
    clip = _clip(5, 48, 64)
    buffers = []
    for threaded in (True, False):
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=Fraction(24))
        stream.height, stream.width = 32, 48
        stream.pix_fmt = "yuv420p"
        if threaded:
            mux_rgb_frames(container, stream, clip)
        else:
            for frame in clip:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        container.close()
        buffers.append(buffer.getvalue())
    assert buffers[0] == buffers[1]


def test_a_conversion_failure_reaches_the_caller():
    container = av.open(io.BytesIO(), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=Fraction(24))
    stream.height, stream.width = 16, 16
    stream.pix_fmt = "yuv420p"
    with pytest.raises(ValueError):
        mux_rgb_frames(container, stream, np.zeros((2, 16, 16, 3), dtype=np.float32))
