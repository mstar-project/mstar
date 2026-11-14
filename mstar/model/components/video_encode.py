"""The H.264 mp4 writer every PyAV video output goes through.

``stream.encode`` converts each frame to the codec's pixel format (swscale) on
the calling thread before handing it to the encoder, so a plain mux loop
alternates the two: the encoder's threads wait while a frame converts, and the
conversion waits while a frame is handed over. Here one thread of the call's
own converts the frames, in order, a few ahead of the encoder, with the
conversion ``stream.encode`` makes (the stream's size and format, default
interpolation and colorspace), so the encoder gets the frames the plain loop
would have handed it, already in its format, and takes them as they are.

:class:`Mp4Stream` writes the same file while its frames are still being made:
a decoder that finishes a clip in parts hands each over as it lands, and the
generated audio goes into the same file as an AAC track, so there is no second
pass over the mp4 to add it.

libx264 at its defaults can write different bytes for the same clip from one
run to the next, plain loop or not. ``threads=1`` has been seen to vary as
well, and ``MALLOC_PERTURB_`` makes it stable, which points at a read of
uninitialised memory rather than at the threading. ``preset=ultrafast`` wrote
the same bytes in every run here, so that is what the byte tests use.
"""
from __future__ import annotations

import io
import queue
import struct
import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from mstar.utils.threads import Thread, ThreadPool

# Frames converted ahead of the encoder: enough to keep it fed, and few enough
# that a clip the encoder falls behind on isn't held a second time, converted
# (a 121-frame 544x960 clip is 95 MB in yuv420p).
LOOKAHEAD = 4

# The AAC track's bit rate: what the API layer's ffmpeg mux of the generated audio wrote.
AAC_BIT_RATE = 192_000


class Mp4Encoding(NamedTuple):
    """How a model writes its ``video`` output (``encode_mp4``'s arguments)."""

    fps: float | Fraction
    options: Mapping[str, str] | None = None
    pad_even: bool = False


class AudioFormat(NamedTuple):
    """Interleaved 16-bit PCM: its rate and channel count."""

    sample_rate: int
    channels: int


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
    writer = Mp4Writer(Mp4Encoding(fps, options, pad_even))
    try:
        writer.write(frames)
        return writer.close()
    finally:
        writer.abort()


class Mp4Writer:
    """One mp4 written frames-first: :meth:`write` the frames in order, as many
    calls as they come in, then :meth:`close` for the bytes. With ``audio``, the
    file also gets an AAC track of the PCM :meth:`write_audio` hands it, muxed in
    this same pass, and its index goes to the front (:func:`faststart`), as the API
    layer's ffmpeg mux wrote it. Not thread-safe: one thread drives it."""

    def __init__(self, encoding: Mp4Encoding, audio: AudioFormat | None = None):
        self.encoding = encoding
        self.audio = audio
        self._buffer = io.BytesIO()
        self._container = None
        self._video = None
        self._audio_stream = None
        self._pcm: bytes | None = None
        self._audio_written = False

    def write(self, frames: np.ndarray, stop: Callable[[], bool] | None = None) -> None:
        """Encode uint8 RGB ``frames`` ``[F, H, W, 3]``, the next ones of the clip. ``stop``,
        asked before each frame, ends the call early (the file is then to be dropped)."""
        if self.encoding.pad_even and (frames.shape[1] % 2 or frames.shape[2] % 2):
            frames = np.pad(frames, ((0, 0), (0, frames.shape[1] % 2), (0, frames.shape[2] % 2), (0, 0)), mode="edge")
        if self._container is None:
            self._open(frames.shape[1], frames.shape[2])
        elif (frames.shape[1], frames.shape[2]) != (self._video.height, self._video.width):
            raise ValueError(
                f"frames of {frames.shape[2]}x{frames.shape[1]} for an mp4 of {self._video.width}x{self._video.height}"
            )
        for frame in converted_frames(self._video.codec_context, frames):
            if stop is not None and stop():
                return
            for packet in self._video.encode(frame):
                self._container.mux(packet)

    def write_audio(self, pcm: bytes) -> None:
        """The whole audio track, interleaved 16-bit PCM in the ``audio`` format; encoded
        now if the file is open, else once its first frames arrive."""
        if self.audio is None:
            raise ValueError("this mp4 has no audio track")
        if self._pcm is not None:
            raise ValueError("the audio track was already given")
        self._pcm = pcm
        if self._container is not None:
            self._encode_audio()

    @property
    def awaits_audio(self) -> bool:
        return self.audio is not None and self._pcm is None

    def close(self) -> bytes:
        """Flush the encoders and finish the file; the mp4's bytes."""
        if self._container is None:
            raise ValueError("an mp4 needs at least one frame")
        if self.awaits_audio:
            raise ValueError("the mp4's audio track never arrived")
        for packet in self._video.encode():  # flush the encoder's delay
            self._container.mux(packet)
        container, self._container = self._container, None
        container.close()
        data = self._buffer.getvalue()
        return faststart(data) if self.audio is not None else data

    def abort(self) -> None:
        """Drop an unfinished file (a no-op after :meth:`close`)."""
        container, self._container = self._container, None
        if container is not None:
            try:
                container.close()
            except Exception:  # noqa: BLE001 — the file is being dropped anyway
                pass

    def _open(self, height: int, width: int) -> None:
        import av

        container = av.open(self._buffer, mode="w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=Fraction(self.encoding.fps).limit_denominator(1001))
            stream.height, stream.width = height, width
            stream.pix_fmt = "yuv420p"
            if self.encoding.options:
                stream.options = dict(self.encoding.options)
            if self.audio is not None:
                layout = {1: "mono", 2: "stereo"}.get(self.audio.channels)
                if layout is None:
                    raise ValueError(f"an AAC track of {self.audio.channels} channels")
                self._audio_stream = container.add_stream("aac", rate=self.audio.sample_rate, layout=layout)
                self._audio_stream.bit_rate = AAC_BIT_RATE
        except BaseException:
            container.close()
            raise
        self._container, self._video = container, stream
        if self._pcm is not None:
            self._encode_audio()

    def _encode_audio(self) -> None:
        import av

        if self._audio_written:
            return
        self._audio_written = True
        samples = np.frombuffer(self._pcm, dtype="<i2").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout=self._audio_stream.layout.name)
        frame.sample_rate = self.audio.sample_rate
        frame.pts = 0
        for packet in self._audio_stream.encode(frame):
            self._container.mux(packet)
        for packet in self._audio_stream.encode():
            self._container.mux(packet)


# boxes whose children hold a track's chunk offsets (moov/trak/mdia/minf/stbl/stco|co64)
_CONTAINER_BOXES = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl"})


def _boxes(data, start: int, end: int):
    """``(type, box start, payload start, box end)`` of the boxes in ``data[start:end]``."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            size, header = struct.unpack_from(">Q", data, pos + 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f"a malformed mp4 box {kind!r} at {pos}")
        yield kind, pos, pos + header, pos + size
        pos += size


def _shift_chunk_offsets(moov: bytearray, start: int, end: int, shift: int) -> None:
    for kind, _, payload, box_end in _boxes(moov, start, end):
        if kind in _CONTAINER_BOXES:
            _shift_chunk_offsets(moov, payload, box_end, shift)
        elif kind in (b"stco", b"co64"):
            count = struct.unpack_from(">I", moov, payload + 4)[0]
            fmt, width = (">I", 4) if kind == b"stco" else (">Q", 8)
            for i in range(count):
                at = payload + 8 + i * width
                offset = struct.unpack_from(fmt, moov, at)[0] + shift
                if kind == b"stco" and offset >= 1 << 32:
                    raise ValueError("an mp4 past 4 GiB with 32-bit chunk offsets")
                struct.pack_into(fmt, moov, at, offset)


def faststart(data: bytes) -> bytes:
    """``data`` with its ``moov`` box moved ahead of its ``mdat``, the chunk offsets moved
    with it: ffmpeg's ``-movflags +faststart``, which PyAV can't apply to a file it writes
    in memory (the muxer reopens the file by name to rewrite it)."""
    boxes = list(_boxes(data, 0, len(data)))
    kinds = [kind for kind, *_ in boxes]
    if b"moov" not in kinds or b"mdat" not in kinds or kinds.index(b"moov") < kinds.index(b"mdat"):
        return data
    _, moov_start, _, moov_end = boxes[kinds.index(b"moov")]
    moov = bytearray(data[moov_start:moov_end])
    _shift_chunk_offsets(moov, 8, len(moov), len(moov))
    # where ffmpeg puts it: right after the ftyp box (ahead of its free box), else ahead of the mdat
    insert = boxes[0][3] if kinds[0] == b"ftyp" else boxes[kinds.index(b"mdat")][1]
    return b"".join((data[:insert], moov, data[insert:moov_start], data[moov_end:]))


class Mp4Stream:
    """An :class:`Mp4Writer` on a thread of its own, fed as a clip's frames are made.

    ``parts`` calls of :meth:`add_part` hand over the clip in pieces, in any order
    (``index`` 0 is the first piece); with ``audio``, :meth:`add_audio` hands over the
    track at any point. The writer encodes each piece once every earlier one has
    arrived, so the clip is written while the rest of it is still being decoded, and
    :attr:`future` gets the mp4's bytes when the last piece and the audio are in.
    Neither call blocks, and a piece is read in place: the future :meth:`add_part`
    returns says when the writer is done with it. An error fails :attr:`future` and
    drops what comes after; :meth:`cancel` stops the writer at its next frame.
    """

    _CANCEL = object()

    def __init__(self, encoding: Mp4Encoding, parts: int, audio: AudioFormat | None = None):
        if parts < 1:
            raise ValueError(f"an mp4 in {parts} parts")
        self.parts = int(parts)
        self.future: Future = Future()
        self._writer = Mp4Writer(encoding, audio)
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._cancelled = False
        # each piece's "done with it" future; resolved by the writer, or all at its exit
        self._lock = threading.Lock()
        self._released: list[Future] = []
        self._closed = False
        self._thread = Thread(target=self._run, name="mstar-mp4", daemon=True)
        self._thread.start()

    @property
    def audio(self) -> AudioFormat | None:
        return self._writer.audio

    def add_part(self, index: int, frames: np.ndarray) -> Future:
        """Piece ``index`` of the clip, uint8 RGB ``[F, H, W, 3]``, read until the returned
        future is done: set once the piece is written, cancelled if the writer stops first."""
        released: Future = Future()
        with self._lock:
            if self._closed:
                released.cancel()
                return released
            self._released.append(released)
            self._inbox.put((int(index), frames, released))
        return released

    def add_audio(self, pcm: bytes) -> None:
        """The audio track, interleaved 16-bit PCM."""
        self._inbox.put(bytes(pcm))

    def cancel(self) -> None:
        """Stop writing: the future is cancelled and the frames still queued are dropped."""
        self._cancelled = True
        self._inbox.put(self._CANCEL)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            data = self._write()
        except _CancelledError:
            self.future.cancel()
        except BaseException as exc:  # noqa: BLE001 — the future carries it to the caller
            self.future.set_exception(exc)
        else:
            self.future.set_result(data)
        finally:
            self._writer.abort()
            with self._lock:  # the writer reads no piece from here on
                self._closed = True
                for released in self._released:
                    released.cancel()

    def _write(self) -> bytes:
        writer = self._writer
        pending: dict[int, tuple[np.ndarray, Future]] = {}
        written = 0
        while written < self.parts or writer.awaits_audio:
            item = self._inbox.get()
            if item is self._CANCEL or self._cancelled:
                raise _CancelledError
            if isinstance(item, bytes):
                writer.write_audio(item)
                continue
            index, frames, released = item
            if not 0 <= index < self.parts or index < written or index in pending:
                raise ValueError(f"part {index} of an mp4 in {self.parts} parts, given twice or out of range")
            pending[index] = (frames, released)
            while written in pending:
                frames, released = pending.pop(written)
                writer.write(frames, stop=lambda: self._cancelled)
                if self._cancelled:
                    raise _CancelledError
                released.set_result(None)
                written += 1
        return writer.close()


class _CancelledError(Exception):
    pass


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
