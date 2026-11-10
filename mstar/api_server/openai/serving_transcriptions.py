"""/v1/audio/transcriptions handler (speech-to-text).

The route is multipart: the router reads the audio upload and the form
fields, validates the fields as a :class:`TranscriptionRequest`, and hands
both here. The adapter turns them into a ``submit_request`` and, once the
text stream is in, lifts the model's control tokens (language, timestamps)
into a :class:`Transcript`.

Long uploads
    A model that hears a bounded clip (``adapter.max_audio_seconds``, 30 s
    for Whisper) gets the upload served as consecutive windows of at most
    that length, each its own engine request; the windows' texts are joined
    and their timestamps offset by the window start. An adapter's
    ``window_seconds`` cuts shorter windows than its bound: Qwen3-ASR hears
    20 minutes but cuts 30 s windows, and a request's ``window_seconds`` can
    ask for the whole file.

    The adapter's ``long_form`` picks the mode, and a request can override it;
    every transcription adapter in ``adapters.py`` defaults to ``parallel``.

    ``sequential`` is openai-whisper's ``transcribe`` loop.
    Windows run in order, each conditioned on the transcript so far
    (``initial_prompt``, when the model reads it) and on the language the
    first window settled. When
    the adapter ``seeks_by_timestamps`` every window is decoded with
    timestamps, and a window that stopped inside a segment gives that
    unfinished segment up: the next window starts where the last closed
    segment ended, so no word is split by a boundary. A window whose text
    compresses better than ``compression_ratio_threshold`` is a repetition
    loop and is decoded again, first without the conditioning text (the
    usual cause), then at rising temperatures; once a temperature above 0.5
    was needed, the transcript so far stops conditioning the windows after
    it, which keeps one bad window from infecting the rest.
    (openai-whisper's log-probability and no-speech thresholds need
    per-token probabilities the engine does not report; they are not applied.)

    ``parallel`` cuts fixed windows, each at the quietest moment shortly
    before its nominal boundary, with no conditioning, and keeps up to
    :data:`PARALLEL_INFLIGHT` of them on the engine at once, fewer while other
    uploads share the server's concurrency cap (:class:`WindowShare`). With timestamps
    (a non-streamed timed format), a window that stopped inside a segment
    keeps its closed segments as in sequential mode, and the audio from the
    last of them to the cut is decoded as one more window.

    Uploads longer than :data:`MAX_DURATION_SECONDS` are rejected. Every
    window still on the engine when the request ends (error, disconnect) is
    released, and the upload and window files are deleted.

Response formats
    ``json`` (text only), ``verbose_json`` (language, duration, segments,
    words), ``text``, ``srt``, ``vtt``. Streaming returns OpenAI's
    transcription event stream — ``transcript.text.delta`` per text chunk
    (per accepted window in sequential long form, whose text is only final
    once its window is), ``transcript.text.done`` with the full text at the
    end, then ``[DONE]``.
"""

from __future__ import annotations

import threading
import zlib
from dataclasses import dataclass
from pathlib import Path

from mstar.api_server import media_io
from mstar.api_server.openai.adapters import Transcript

SAMPLE_RATE = 16000
# parallel mode: how far before a fixed boundary to look for the quietest cut,
# and the frame length compared
CUT_SEARCH_SECONDS = 2.0
CUT_FRAME_SECONDS = 0.02
# a tail shorter than this holds no speech, and some models refuse one so short
MIN_TAIL_SECONDS = 0.02
# longest upload the route accepts, in seconds and in bytes
MAX_DURATION_SECONDS = 30 * 60
_SF_UNKNOWN_FRAMES = 2**62


def audio_duration_seconds(path: str) -> float | None:
    """Duration from the container, or None when it can't be read."""
    try:
        import soundfile as sf

        info = sf.info(path)
        # libsndfile reports an unknown length (a FLAC whose header leaves the
        # sample count out) as the largest frame count
        if info.frames < _SF_UNKNOWN_FRAMES:
            return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001 — try the FFmpeg decoder
        pass
    try:
        from torchcodec.decoders import AudioDecoder

        meta = AudioDecoder(path).metadata
        duration = getattr(meta, "duration_seconds_from_header", None)
        if duration is None:
            duration = getattr(meta, "duration_seconds", None)
        return float(duration) if duration is not None else None
    except Exception:  # noqa: BLE001 — best effort
        return None


def compression_ratio(text: str) -> float:
    """UTF-8 bytes over their zlib size: a repetition loop scores far above
    prose (openai-whisper's ``compression_ratio``)."""
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data)))


@dataclass
class Window:
    """One engine request's worth of audio and where it sits in the upload."""
    path: str
    offset: float
    duration: float
    request_id: str | None = None
    raw_text: str = ""
    # the accepted parse (sequential mode drops an unfinished last segment)
    transcript: Transcript | None = None
    # the temperature it was accepted at; None where no retry ladder ran (the request's)
    temperature: float | None = None
    compression_ratio: float = 0.0


class WindowPlanner:
    """Cuts an upload into windows for a model with a clip bound.

    ``fixed()`` cuts the whole file up front (parallel mode), moving each cut
    to the quietest frame within ``search_seconds`` before its nominal
    boundary. ``window_at(offset)`` cuts one window from ``offset`` on
    (sequential mode, where the previous window decides where the next
    starts). A file of at most ``whole_seconds`` (default: the window
    length) is a single window either way and is never re-encoded.
    """

    def __init__(
        self, audio_path: str, max_seconds: float | None, upload_dir: Path, *,
        balanced: bool = False, search_seconds: float = CUT_SEARCH_SECONDS,
        frame_seconds: float = CUT_FRAME_SECONDS, whole_seconds: float | None = None,
    ):
        self.audio_path = audio_path
        self.max_seconds = max_seconds
        self.whole_seconds = max_seconds if whole_seconds is None else whole_seconds
        self.balanced = balanced
        # a cut moves back at most a third of a window, so short windows a
        # request asks for keep most of their length
        self.search_seconds = search_seconds if max_seconds is None else min(search_seconds, max_seconds / 3)
        self.frame_seconds = frame_seconds
        self.upload_dir = Path(upload_dir)
        self.duration = audio_duration_seconds(audio_path)
        self.files = [audio_path]
        self._audio = None
        self._count = 0
        # a disconnect can clean up while a thread is still cutting windows
        self._lock = threading.Lock()
        self._closed = False

    @property
    def single(self) -> bool:
        return self.max_seconds is None or self.duration is None or self.duration <= self.whole_seconds

    def whole(self) -> Window:
        return Window(path=self.audio_path, offset=0.0, duration=self.duration or 0.0)

    @property
    def decodes(self) -> bool:
        """``prepare()`` decodes the upload (otherwise it does nothing)."""
        return not self.single or self.duration is None

    def prepare(self) -> None:
        """Decode an upload that will be windowed, so a file the decoder
        refuses (e.g. an unsupported sample rate, or audio past
        :data:`MAX_DURATION_SECONDS`, where the decode stops whatever the
        header says) fails before any window runs. A file with no header
        duration is decoded too: a refusal fails here the same way, and once
        its length is known it is windowed when longer than the model's clip.
        One that does not decode here (or with no audio decoder in this
        process) is sent whole to the data worker, whose decode has the same
        cap and reports its own errors."""
        if not self.single:
            self._decoded()
        elif self.duration is None:
            try:
                self._decoded()
            except media_io.AudioRefusedError:
                raise
            except Exception:  # noqa: BLE001 — not decodable here; the data worker reports it
                pass

    def done(self, offset: float) -> bool:
        return self.single or offset >= (self.duration or 0.0) - MIN_TAIL_SECONDS

    def fixed(self) -> list[Window]:
        if self.single:
            return [self.whole()]
        pieces = media_io.split_windows(
            self._decoded(), self.max_seconds, SAMPLE_RATE,
            search_seconds=self.search_seconds, frame_seconds=self.frame_seconds, balanced=self.balanced,
        )
        if len(pieces) > 1 and len(pieces[-1]) < MIN_TAIL_SECONDS * SAMPLE_RATE:
            # a cut at the boundary of audio that ends in silence leaves a sliver
            pieces.pop()
        windows, offset = [], 0.0
        for piece in pieces:
            windows.append(self._write(piece, offset))
            offset += len(piece) / SAMPLE_RATE
        return windows

    def window_at(self, offset: float) -> Window:
        if self.single:
            return self.whole()
        audio = self._decoded()
        start = int(round(offset * SAMPLE_RATE))
        stop = media_io.window_stop(start, len(audio), int(round(self.max_seconds * SAMPLE_RATE)), self.balanced)
        return self._write(audio[start:stop], offset)

    def cut(self, offset: float, duration: float) -> Window:
        """``duration`` seconds of the upload from ``offset`` on, as a window."""
        audio = self._decoded()
        start = int(round(offset * SAMPLE_RATE))
        return self._write(audio[start:start + int(round(duration * SAMPLE_RATE))], offset)

    def _decoded(self):
        if self._audio is None:
            self._audio = media_io.decode_audio(self.audio_path, SAMPLE_RATE, MAX_DURATION_SECONDS)
            self.duration = len(self._audio) / SAMPLE_RATE
        return self._audio

    def _write(self, piece, offset: float) -> Window:
        stem = Path(self.audio_path).stem
        path = media_io.write_wav(piece, str(self.upload_dir / f"{stem}_w{self._count:04d}.wav"), SAMPLE_RATE)
        with self._lock:
            if self._closed:
                Path(path).unlink(missing_ok=True)
                raise RuntimeError("the request was closed while its upload was being cut")
            self.files.append(path)
        self._count += 1
        return Window(path=path, offset=offset, duration=len(piece) / SAMPLE_RATE)

    def cleanup(self) -> None:
        with self._lock:
            self._closed = True
            files, self.files = self.files, []
        for path in files:
            Path(path).unlink(missing_ok=True)


def _closed(t: Transcript) -> Transcript:
    """``t`` without its unfinished last segment, whose audio the next window
    hears again from the last closed segment's end."""
    end = t.segments[-1]["end"]
    return Transcript(
        text=" ".join(s["text"].strip() for s in t.segments if s["text"].strip()),
        language=t.language,
        segments=list(t.segments),
        words=[w for w in t.words if w["end"] <= end + 1e-6],
    )

