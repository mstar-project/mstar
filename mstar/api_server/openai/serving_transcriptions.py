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
    and their timestamps offset by the window start. Models without a bound
    (Qwen3-ASR hears 20 minutes) see the whole file.

    ``sequential`` (the default) is openai-whisper's ``transcribe`` loop.
    Windows run in order, each conditioned on the transcript so far
    (``initial_prompt``) and on the language the first window settled. When
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

    ``parallel`` submits fixed windows all at once, each cut at the quietest
    moment shortly before its nominal boundary, with no conditioning.

Response formats
    ``json`` (text only), ``verbose_json`` (language, duration, segments,
    words), ``text``, ``srt``, ``vtt``. Streaming returns OpenAI's
    transcription event stream — ``transcript.text.delta`` per text chunk
    (per accepted window in sequential long form, whose text is only final
    once its window is), ``transcript.text.done`` with the full text at the
    end, then ``[DONE]``.
"""

from __future__ import annotations

import asyncio
import math
import os
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from mstar.api_server import media_io
from mstar.api_server.openai._util import SSE_DONE, rid, sse
from mstar.api_server.openai.adapters import Transcript

_TIMED_FORMATS = ("verbose_json", "srt", "vtt")
RESPONSE_FORMATS = ("json", "text", *_TIMED_FORMATS)
LONG_FORM_MODES = ("sequential", "parallel")
SAMPLE_RATE = 16000
# how much of the transcript so far conditions the next window (the model
# keeps only its last ~220 tokens anyway)
CARRY_OVER_CHARS = 1200
# openai-whisper's fallback schedule, used unless the caller pinned a temperature
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# a window that needed a temperature above this stops conditioning the next
PROMPT_RESET_TEMPERATURE = 0.5
# a seek shorter than this re-hears too much for too little; take the window
MIN_SEEK_SECONDS = 1.0
# parallel mode: how far before a fixed boundary to look for the quietest cut
CUT_SEARCH_SECONDS = 2.0


def save_upload(audio_bytes: bytes, filename: str | None, upload_dir: Path) -> str:
    """Persist the uploaded audio under ``upload_dir`` and return its path.

    Only the final path component of the client's name is kept, so an
    embedded ``../`` cannot escape the directory; the extension is kept
    because the data worker's decoder sniffs the container by it.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    base = os.path.basename(filename or "") or "audio.wav"
    path = upload_dir / f"{uuid.uuid4()}_{base}"
    path.write_bytes(audio_bytes)
    return str(path)


def audio_duration_seconds(path: str) -> float | None:
    """Duration from the container, or None when it can't be read."""
    try:
        import soundfile as sf

        info = sf.info(path)
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
    temperature: float = 0.0
    compression_ratio: float = 0.0


class WindowPlanner:
    """Cuts an upload into windows for a model with a clip bound.

    ``fixed()`` cuts the whole file up front (parallel mode), moving each cut
    to the quietest moment within :data:`CUT_SEARCH_SECONDS` before its
    nominal boundary. ``window_at(offset)`` cuts one window from ``offset``
    on (sequential mode, where the previous window decides where the next
    starts). A file the model can hear whole is a single window either way
    and is never re-encoded.
    """

    def __init__(self, audio_path: str, max_seconds: float | None, upload_dir: Path):
        self.audio_path = audio_path
        self.max_seconds = max_seconds
        self.upload_dir = Path(upload_dir)
        self.duration = audio_duration_seconds(audio_path)
        self._audio = None
        self._count = 0

    @property
    def single(self) -> bool:
        return self.max_seconds is None or self.duration is None or self.duration <= self.max_seconds

    def whole(self) -> Window:
        return Window(path=self.audio_path, offset=0.0, duration=self.duration or 0.0)

    def done(self, offset: float) -> bool:
        return self.single or offset >= (self.duration or 0.0) - 1e-3

    def fixed(self) -> list[Window]:
        if self.single:
            return [self.whole()]
        pieces = media_io.split_windows(
            self._decoded(), self.max_seconds, SAMPLE_RATE, search_seconds=CUT_SEARCH_SECONDS,
        )
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
        stop = min(len(audio), start + int(round(self.max_seconds * SAMPLE_RATE)))
        return self._write(audio[start:stop], offset)

    def _decoded(self):
        if self._audio is None:
            self._audio = media_io.decode_audio(self.audio_path, SAMPLE_RATE)
            self.duration = len(self._audio) / SAMPLE_RATE
        return self._audio

    def _write(self, piece, offset: float) -> Window:
        stem = Path(self.audio_path).stem
        path = media_io.write_wav(piece, str(self.upload_dir / f"{stem}_w{self._count:04d}.wav"), SAMPLE_RATE)
        self._count += 1
        return Window(path=path, offset=offset, duration=len(piece) / SAMPLE_RATE)


def plan_windows(audio_path: str, max_seconds: float | None, upload_dir: Path) -> list[Window]:
    """The whole file as one window, or fixed ``max_seconds`` windows when
    the model's clip length bounds it (each written next to the upload)."""
    return WindowPlanner(audio_path, max_seconds, upload_dir).fixed()


def _long_form_mode(adapter, req) -> str:
    mode = (getattr(req, "model_extra", None) or {}).get("long_form") or adapter.long_form
    if mode not in LONG_FORM_MODES:
        raise HTTPException(status_code=400, detail=f"long_form must be one of {list(LONG_FORM_MODES)}; got {mode!r}")
    return mode


def _window_request(req, language: str | None, carry_over: str | None, *, timestamps: bool = False):
    """The per-window request: the caller's, with the language settled by
    the first window, the transcript so far as the conditioning text and,
    when the driver seeks by timestamps, segment timestamps switched on."""
    update: dict = {}
    if language and not req.language:
        update["language"] = language
    if carry_over is not None:
        update["prompt"] = carry_over
    if timestamps and not req.timestamp_granularities and req.response_format not in _TIMED_FORMATS:
        update["timestamp_granularities"] = ["segment"]
    return req.model_copy(update=update) if update else req


def _attempts(req, carry_over: str | None) -> list[tuple[float, str | None]]:
    """The ``(temperature, conditioning text)`` ladder for one window.

    The first attempt is the caller's temperature with the transcript so far
    as the prompt. A repetition loop is nearly always the prompt's doing, so
    the first retry is the same greedy decode without it, and only then does
    the temperature climb (openai-whisper's schedule, whose retries keep the
    prompt but also sample five candidates and pick by log-probability, which
    the engine does not offer). A pinned temperature is never escalated.
    """
    base = req.temperature or 0.0
    ladder = [(base, carry_over)]
    if carry_over:
        ladder.append((base, None))
    if not req.temperature:
        ladder.extend((t, None) for t in TEMPERATURE_FALLBACK if t > base)
    return ladder


def _submit(api, adapter, req, window: Window, streaming: bool) -> str:
    args = adapter.transcription_to_request(req, window.path)
    args.model_kwargs.pop("long_form", None)
    request_id = rid("transcr")
    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        prompt_parts=args.prompt_parts,
        streaming=streaming,
        request_id=request_id,
    )
    window.request_id = request_id
    return request_id


def _text_of(chunks) -> str:
    parts: list[str] = []
    for c in chunks:
        if c.modality == "text":
            parts.append(c.data.decode("utf-8", "replace"))
        elif c.modality == "error":
            raise HTTPException(
                status_code=int((c.metadata or {}).get("status") or 500),
                detail=c.data.decode("utf-8", "replace"),
            )
    return "".join(parts)


def _carry_over(texts: list[str]) -> str:
    joined = " ".join(t for t in texts if t).strip()
    return joined[-CARRY_OVER_CHARS:]


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


async def _sequential(api, adapter, req, planner: WindowPlanner, raw_request=None):
    """openai-whisper's loop over the upload; yields each window once its
    transcript is accepted (see the module docstring)."""
    language = req.language
    texts: list[str] = []     # accepted window texts, the conditioning source
    reset_since = 0           # texts before this index no longer condition
    seeks = bool(adapter.seeks_by_timestamps) and not planner.single
    offset = 0.0
    while True:
        window = planner.window_at(offset)
        carry = _carry_over(texts[reset_since:]) if texts else None
        for temperature, prompt in _attempts(req, carry):
            attempt = _window_request(req, language, prompt, timestamps=seeks)
            if temperature != (req.temperature or 0.0):
                attempt = attempt.model_copy(update={"temperature": temperature})
            _submit(api, adapter, attempt, window, streaming=False)
            window.raw_text = _text_of(await api.collect_results(window.request_id, raw_request))
            parsed = adapter.parse_transcript(window.raw_text, attempt)
            window.temperature = temperature
            window.compression_ratio = compression_ratio(parsed.text)
            threshold = adapter.compression_ratio_threshold
            if threshold is None or window.compression_ratio <= threshold:
                break
        consumed = window.duration
        if seeks and parsed.unfinished and parsed.segments:
            end = parsed.segments[-1]["end"]
            if MIN_SEEK_SECONDS <= end < window.duration:
                consumed = end
                parsed = _closed(parsed)
        window.transcript = parsed
        texts.append(parsed.text)
        if window.temperature > PROMPT_RESET_TEMPERATURE:
            reset_since = len(texts)
        language = language or parsed.language
        yield window
        offset += consumed
        if planner.done(offset):
            return


def merge_windows(windows: list[Window], adapter, req) -> Transcript:
    """Stitch the windows: texts joined by a space, segments/words shifted by
    the window's start time (segments also carry the temperature and
    compression ratio their window was accepted at), the language from the
    first window that reported one."""
    texts: list[str] = []
    segments: list[dict] = []
    words: list[dict] = []
    language = None
    for window in windows:
        t = window.transcript or adapter.parse_transcript(window.raw_text, req)
        if t.text:
            texts.append(t.text)
        for seg in t.segments:
            segments.append({
                **seg, "start": seg["start"] + window.offset, "end": seg["end"] + window.offset,
                "temperature": window.temperature, "compression_ratio": window.compression_ratio,
            })
        for word in t.words:
            words.append({**word, "start": word["start"] + window.offset, "end": word["end"] + window.offset})
        language = language or t.language
    for idx, seg in enumerate(segments):
        seg["id"] = idx
    return Transcript(text=" ".join(texts), language=language or req.language, segments=segments, words=words)


async def create_transcription(
    api, model_name, adapter, req, audio_bytes: bytes, filename: str | None,
    raw_request=None,
):
    fmt = (req.response_format or "json").lower()
    if fmt not in RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"response_format must be one of {list(RESPONSE_FORMATS)}; got {fmt!r}",
        )
    mode = _long_form_mode(adapter, req)
    audio_path = save_upload(audio_bytes, filename, api.upload_dir)
    planner = WindowPlanner(audio_path, adapter.max_audio_seconds, api.upload_dir)

    if req.stream:
        return StreamingResponse(
            _stream(api, adapter, req, planner, mode, audio_path, raw_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    if mode == "parallel" or planner.single:
        windows = planner.fixed()
        for window in windows:
            _submit(api, adapter, req, window, streaming=False)
        for window in windows:
            window.raw_text = _text_of(await api.collect_results(window.request_id, raw_request))
    else:
        windows = [window async for window in _sequential(api, adapter, req, planner, raw_request)]

    transcript = merge_windows(windows, adapter, req)
    duration = audio_duration_seconds(audio_path)
    if fmt == "text":
        return PlainTextResponse(transcript.text)
    if fmt == "json":
        return JSONResponse({"text": transcript.text})
    if fmt == "srt":
        return PlainTextResponse(render_srt(transcript.segments or _whole(transcript, duration)))
    if fmt == "vtt":
        return PlainTextResponse(render_vtt(transcript.segments or _whole(transcript, duration)))
    return JSONResponse(_verbose(transcript, req, duration))


def _whole(transcript: Transcript, duration: float | None) -> list[dict]:
    """One segment spanning the file, for timed formats on a model that did
    not emit timestamps."""
    return [{"id": 0, "start": 0.0, "end": duration or 0.0, "text": transcript.text}]


def _verbose(transcript: Transcript, req, duration: float | None) -> dict:
    granularities = set(req.timestamp_granularities or ())
    body: dict = {
        "task": "transcribe",
        "language": transcript.language,
        "duration": duration,
        "text": transcript.text,
    }
    if not granularities or "segment" in granularities:
        body["segments"] = [
            {
                "id": seg.get("id", i),
                "seek": 0,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "tokens": [],
                "temperature": seg.get("temperature", req.temperature or 0.0),
                "avg_logprob": 0.0,
                "compression_ratio": seg.get("compression_ratio", 0.0),
                "no_speech_prob": 0.0,
            }
            for i, seg in enumerate(transcript.segments)
        ]
    if "word" in granularities:
        body["words"] = list(transcript.words)
    body["usage"] = _usage(duration)
    return body


def _usage(duration: float | None) -> dict:
    return {"type": "duration", "seconds": int(math.ceil(duration)) if duration else 0}


def _fmt_time(seconds: float, sep: str) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def render_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_time(seg['start'], ',')} --> {_fmt_time(seg['end'], ',')}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def render_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_fmt_time(seg['start'], '.')} --> {_fmt_time(seg['end'], '.')}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _error_event(message: str, status: int | None) -> str:
    return sse({
        "type": "error",
        "error": {"message": message, "type": "server_error", "code": status or 500},
    })


async def _stream(api, adapter, req, planner: WindowPlanner, mode: str, audio_path: str, raw_request=None):
    """Deltas as they come for one window or parallel windows (in window
    order); sequential long form emits each window's text once the window
    is accepted, since only then is it known to stand."""
    windows: list[Window] = []
    if mode == "parallel" or planner.single:
        windows = planner.fixed()
        for window in windows:
            _submit(api, adapter, req, window, streaming=True)
        for i, window in enumerate(windows):
            raw_parts: list[str] = []
            async for c in api.iter_result_chunks(window.request_id):
                if c.modality == "error":
                    yield _error_event(c.data.decode("utf-8", "replace"), (c.metadata or {}).get("status"))
                    yield SSE_DONE
                    return
                if c.modality != "text" or not c.data:
                    continue
                text = c.data.decode("utf-8", "replace")
                raw_parts.append(text)
                delta = adapter.stream_delta(text)
                if delta:
                    yield sse({"type": "transcript.text.delta", "delta": delta})
            window.raw_text = "".join(raw_parts)
            if i + 1 < len(windows):
                # windows are joined by a space in the final text; say so mid-stream
                yield sse({"type": "transcript.text.delta", "delta": " "})
            await asyncio.sleep(0)
    else:
        try:
            async for window in _sequential(api, adapter, req, planner, raw_request):
                if windows:
                    yield sse({"type": "transcript.text.delta", "delta": " "})
                windows.append(window)
                if window.transcript.text:
                    yield sse({"type": "transcript.text.delta", "delta": window.transcript.text})
                await asyncio.sleep(0)
        except HTTPException as exc:
            yield _error_event(str(exc.detail), exc.status_code)
            yield SSE_DONE
            return
    transcript = merge_windows(windows, adapter, req)
    yield sse({
        "type": "transcript.text.done",
        "text": transcript.text,
        "usage": _usage(audio_duration_seconds(audio_path)),
    })
    yield SSE_DONE
