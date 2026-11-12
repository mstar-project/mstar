"""The H.264 mp4 writers video outputs go through: ``encode_mp4`` (PyAV, one
libx264 encode) and ``encode_mp4_chunked`` (torchcodec, a clip in parallel
pieces joined into one stream).

``stream.encode`` converts each frame to the codec's pixel format (swscale) on
the calling thread before handing it to the encoder, so a plain mux loop
alternates the two: the encoder's threads wait while a frame converts, and the
conversion waits while a frame is handed over. ``encode_mp4`` has one thread of
the call's own convert the frames, in order, a few ahead of the encoder, with the
conversion ``stream.encode`` makes (the stream's size and format, default
interpolation and colorspace), so the encoder gets the frames the plain loop
would have handed it, already in its format, and takes them as they are.

libx264 at its defaults can write different bytes for the same clip from one
run to the next, plain loop or not. ``threads=1`` has been seen to vary as
well, and ``MALLOC_PERTURB_`` makes it stable, which points at a read of
uninitialised memory rather than at the threading. ``preset=ultrafast`` wrote
the same bytes in every run here, so that is what the byte tests use.
"""
from __future__ import annotations

import io
import threading
from collections import deque
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from mstar.utils import cpus
from mstar.utils.threads import ThreadPool

if TYPE_CHECKING:
    import torch

# Frames converted ahead of the encoder: enough to keep it fed, and few enough
# that a clip the encoder falls behind on isn't held a second time, converted
# (a 121-frame 544x960 clip is 95 MB in yuv420p).
LOOKAHEAD = 4


def encode_mp4(
    frames: np.ndarray,
    fps: float | Fraction,
    *,
    options: Mapping[str, str] | None = None,
    pad_even: bool = False,
) -> bytes:
    """H.264 mp4 bytes from uint8 RGB ``frames`` ``[F, H, W, 3]`` through PyAV's
    bundled libx264, in yuv420p.

    The container's rate is ``fps`` as the nearest fraction with a denominator up
    to 1001, so NTSC rates are exact (PyAV takes no float rate: 24 -> 24,
    30000/1001 -> 30000/1001). ``options`` are the stream's codec options (None: libx264's
    defaults). yuv420p needs even sides; ``pad_even`` repeats the last row or
    column of an odd-sized clip, else such a clip fails to encode.
    """
    import av

    if pad_even and (frames.shape[1] % 2 or frames.shape[2] % 2):
        frames = np.pad(frames, ((0, 0), (0, frames.shape[1] % 2), (0, frames.shape[2] % 2), (0, 0)), mode="edge")
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
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


# A clip splits only into at least this many pieces of at least this many
# frames (48 frames and up): each piece pays x264's setup, and the join a mux,
# so fewer, shorter pieces measured slower than one encode (832x480 on an H200
# host: 24 frames 54 ms whole, 69 ms in 2; 48 frames 104 ms whole, 73 ms in 4).
MIN_PIECES = 4
MIN_PIECE_FRAMES = 12


def encode_mp4_chunked(
    frames: torch.Tensor, fps: float | Fraction, *, crf: int, preset: str, pieces: int | None = None,
) -> bytes:
    """H.264 mp4 bytes from uint8 frames ``[F, 3, H, W]`` (a CPU tensor, torchcodec's
    layout), encoded as ``pieces`` runs of frames at once (default: the CPU plan's
    ``video_chunks``) and joined into one stream; a clip too short to split
    (``MIN_PIECES`` x ``MIN_PIECE_FRAMES``) is one plain torchcodec encode.

    Each piece is a libx264 encode of its own, so it starts on a keyframe: the
    clip gets one per piece, and is a few percent larger. The pieces are raw
    H.264 (Annex B), which carries no timestamps, so they are encoded without
    B-frames: decode order is then display order, and one mux stamps every frame
    by its index at the container's rate (the nearest fraction to ``fps`` with a
    denominator up to 1001, as ``encode_mp4``). Pieces run on one pool the
    process shares, sized for ``encode_workers`` clips at once, and x264's threads
    are split over it from the CPU plan rather than left to x264, which sizes
    itself from the affinity mask. 121 frames at 832x480, ultrafast, crf 18, on an
    H200 host: about 105 ms in 8 pieces against 150-235 ms in one.
    """
    from torchcodec.encoders import VideoEncoder

    plan = cpus.plan()
    rate = Fraction(fps).limit_denominator(1001)
    count = min(plan.video_chunks if pieces is None else pieces, len(frames) // MIN_PIECE_FRAMES)
    options = {"codec": "libx264", "crf": crf, "preset": preset}
    if count < MIN_PIECES:
        # one encode's share of the CPUs, as the data worker runs encode_workers at once
        threads = str(max(1, plan.left // plan.encode_workers))
        return VideoEncoder(frames, frame_rate=float(rate)).to_tensor(
            "mp4", **options, extra_options={"threads": threads},
        ).numpy().tobytes()
    threads = str(max(1, plan.left // plan.video_encode_threads))

    def encode(piece):
        return VideoEncoder(piece, frame_rate=float(rate)).to_tensor(
            "h264", **options, extra_options={"threads": threads, "x264-params": "bframes=0"},
        ).numpy().tobytes()

    stream = b"".join(_encode_pool().map(encode, frames.tensor_split(count)))
    return _mux_h264(stream, rate)


def _mux_h264(stream: bytes, rate: Fraction) -> bytes:
    """An Annex B H.264 stream in an mp4, one frame per packet at ``rate``."""
    import av

    buffer = io.BytesIO()
    with av.open(io.BytesIO(stream), format="h264") as source, av.open(buffer, mode="w", format="mp4") as container:
        src = source.streams.video[0]
        dst = container.add_stream_from_template(src)
        # Packets count frames; the muxer rescales them to the timescale it picks
        # when it writes the header (and so changes ``dst.time_base``).
        frame = 1 / rate
        dst.time_base = frame
        index = 0
        for packet in source.demux(src):
            if packet.size == 0:  # the demuxer's end-of-stream marker
                continue
            packet.pts = packet.dts = index
            packet.duration = 1
            packet.time_base = frame
            packet.stream = dst
            container.mux(packet)
            index += 1
    return buffer.getvalue()


_encode_threads: ThreadPool | None = None
_encode_lock = threading.Lock()


def _encode_pool() -> ThreadPool:
    """The threads every chunked clip of this process encodes its pieces on, the CPU
    plan's ``video_encode_threads``: clips finishing together queue their pieces
    here instead of each starting threads of its own."""
    global _encode_threads
    if _encode_threads is None:
        with _encode_lock:
            if _encode_threads is None:
                _encode_threads = ThreadPool(
                    max_workers=cpus.plan().video_encode_threads, thread_name_prefix="mstar-h264",
                )
    return _encode_threads


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
