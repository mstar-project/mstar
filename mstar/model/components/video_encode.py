"""The H.264 mp4 writer every PyAV video output goes through.

``stream.encode`` converts each frame to the codec's pixel format (swscale) on
the calling thread before handing it to the encoder, so a plain mux loop
alternates the two: the encoder's threads wait while a frame converts, and the
conversion waits while a frame is handed over. Here one thread of the call's
own converts the frames, in order, a few ahead of the encoder, with the
conversion ``stream.encode`` makes (the stream's size and format, default
interpolation and colorspace), so the encoder gets the frames the plain loop
would have handed it, already in its format, and takes them as they are.

libx264 with threads and a lookahead (its defaults) can write different bytes
for the same clip from one run to the next, plain loop or not; settings without
one of the two (``preset=ultrafast``, or ``threads=1``) write the same bytes
every time.
"""
from __future__ import annotations

import io
from collections import deque
from collections.abc import Iterator, Mapping
from fractions import Fraction

import numpy as np

from mstar.utils.threads import ThreadPool

# Frames converted ahead of the encoder: enough to keep it fed, and few enough
# that a clip the encoder falls behind on isn't held a second time, converted
# (a 121-frame 544x960 clip is 95 MB in yuv420p).
LOOKAHEAD = 4


def encode_mp4(
    frames: np.ndarray,
    fps: float | Fraction,
    *,
    options: Mapping[str, str] | None = None,
    rate_denominator: int = 1001,
    pad_even: bool = False,
) -> bytes:
    """H.264 mp4 bytes from uint8 RGB ``frames`` ``[F, H, W, 3]`` through PyAV's
    bundled libx264, in yuv420p.

    The container's rate is ``fps`` as the nearest fraction with a denominator up
    to ``rate_denominator`` (PyAV takes no float rate: 24 -> 24, 23.976 ->
    2997/125). ``options`` are the stream's codec options (None: libx264's
    defaults). yuv420p needs even sides; ``pad_even`` repeats the last row or
    column of an odd-sized clip, else such a clip fails to encode.
    """
    import av

    if pad_even and (frames.shape[1] % 2 or frames.shape[2] % 2):
        frames = np.pad(frames, ((0, 0), (0, frames.shape[1] % 2), (0, frames.shape[2] % 2), (0, 0)), mode="edge")
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(rate_denominator))
        stream.height, stream.width = frames.shape[1], frames.shape[2]
        stream.pix_fmt = "yuv420p"
        if options:
            stream.options = dict(options)
        for frame in converted_frames(stream.codec_context, frames):
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder's delay
            container.mux(packet)
    return buffer.getvalue()


def converted_frames(context, frames: np.ndarray, lookahead: int = LOOKAHEAD) -> Iterator:
    """``frames`` as PyAV frames in the video codec ``context``'s size and pixel
    format, converted on a thread of their own up to ``lookahead`` ahead of the
    caller, with the arguments the context's encode path reformats with."""
    import av

    width, height, pix_fmt, threads = context.width, context.height, context.pix_fmt, context.thread_count

    def convert(frame: np.ndarray):
        return av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(width, height, pix_fmt, threads=threads)

    with ThreadPool(max_workers=1, thread_name_prefix="mstar-yuv") as pool:
        pending: deque = deque()
        try:
            for frame in frames:
                pending.append(pool.submit(convert, frame))
                if len(pending) > lookahead:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:  # the caller stopped early: drop what hasn't started
                future.cancel()
