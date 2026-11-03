"""Muxing RGB frames through a PyAV video stream with their pixel-format
conversion running ahead of the encoder.

``stream.encode`` converts each frame to the codec's pixel format (swscale) on
the calling thread before handing it to the encoder, so a plain mux loop
alternates the two: the encoder's threads wait while a frame converts, and the
conversion waits while a frame is handed over. Here one thread of the call's
own converts the frames, in order, with the same conversion ``stream.encode``
makes (the stream's size and format, default interpolation and colorspace),
and the encoder gets frames already in its format, which it takes as they are.
The muxed bytes are the ones the plain loop writes.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from mstar.utils.threads import ThreadPool


def mux_rgb_frames(container, stream, frames: np.ndarray) -> None:
    """Encode uint8 ``[F, H, W, 3]`` RGB ``frames`` on the PyAV video ``stream``
    and mux the packets into ``container``, flushing the encoder at the end.
    The stream's size, pixel format and options must be set; the container is
    left open."""
    for frame in _converted(stream, frames):
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():  # flush the encoder's delay
        container.mux(packet)


def _converted(stream, frames: np.ndarray) -> Iterator:
    import av

    context = stream.codec_context
    width, height, pix_fmt = context.width, context.height, context.pix_fmt

    def convert(frame: np.ndarray):
        # the arguments VideoCodecContext's encode path reformats with
        return av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(
            width, height, pix_fmt, threads=context.thread_count,
        )

    # map() queues every frame at once; the one thread converts them in order
    # while the encoder takes the finished ones
    with ThreadPool(max_workers=1, thread_name_prefix="mstar-yuv") as pool:
        yield from pool.map(convert, frames)
