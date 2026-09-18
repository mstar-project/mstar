"""/v1/audio/transcriptions handler (speech-to-text).

The route is multipart: the router reads the audio upload and the form
fields, validates the fields as a :class:`TranscriptionRequest`, and hands
both here. The adapter turns them into a ``submit_request`` and, once the
text stream is in, lifts the model's control tokens (language, timestamps)
into a :class:`Transcript`.

Long uploads
    A model that hears a bounded clip (``adapter.max_audio_seconds``, 30 s
    for Whisper) gets the upload cut into consecutive windows of that
    length, each served as its own engine request. ``sequential`` (the
    default, openai-whisper's algorithm) runs them in order and feeds each
    window the transcript so far as ``initial_prompt``, plus the language
    the first window detected; ``parallel`` submits them all at once. The
    windows' texts are joined and their timestamps offset by the window
    start. Models without a bound (Qwen3-ASR hears 20 minutes) see the
    whole file.

Response formats
    ``json`` (text only), ``verbose_json`` (language, duration, segments,
    words), ``text``, ``srt``, ``vtt``. Streaming returns OpenAI's
    transcription event stream — ``transcript.text.delta`` per text chunk,
    ``transcript.text.done`` with the full text at the end, then ``[DONE]``.
"""

from __future__ import annotations

import asyncio
import math
import os
import uuid
from dataclasses import dataclass, field
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


@dataclass
class Window:
    """One engine request's worth of audio and where it sits in the upload."""
    path: str
    offset: float
    duration: float
    request_id: str | None = None
    raw_text: str = ""
    error: str | None = None
    chunks: list = field(default_factory=list)


def plan_windows(audio_path: str, max_seconds: float | None, upload_dir: Path) -> list[Window]:
    """The whole file as one window, or ``max_seconds`` windows when the
    model's clip length bounds it (each written next to the upload)."""
    duration = audio_duration_seconds(audio_path)
    if max_seconds is None or duration is None or duration <= max_seconds:
        return [Window(path=audio_path, offset=0.0, duration=duration or 0.0)]
    audio = media_io.decode_audio(audio_path, SAMPLE_RATE)
    pieces = media_io.split_windows(audio, max_seconds, SAMPLE_RATE)
    stem = Path(audio_path).stem
    windows = []
    for i, piece in enumerate(pieces):
        path = media_io.write_wav(piece, str(Path(upload_dir) / f"{stem}_w{i:04d}.wav"), SAMPLE_RATE)
        windows.append(Window(path=path, offset=i * max_seconds, duration=len(piece) / SAMPLE_RATE))
    return windows


def _long_form_mode(adapter, req) -> str:
    mode = (getattr(req, "model_extra", None) or {}).get("long_form") or adapter.long_form
    if mode not in LONG_FORM_MODES:
        raise HTTPException(status_code=400, detail=f"long_form must be one of {list(LONG_FORM_MODES)}; got {mode!r}")
    return mode


def _window_request(req, language: str | None, carry_over: str | None):
    """The per-window request: the caller's, with the language settled by
    the first window and the transcript so far as the conditioning text."""
    update: dict = {}
    if language and not req.language:
        update["language"] = language
    if carry_over is not None:
        update["prompt"] = carry_over
    return req.model_copy(update=update) if update else req


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


def merge_windows(windows: list[Window], adapter, req) -> Transcript:
    """Parse every window's raw text and stitch them: texts joined by a
    space, segments/words shifted by the window's start time, the language
    from the first window that reported one."""
    texts: list[str] = []
    segments: list[dict] = []
    words: list[dict] = []
    language = None
    for window in windows:
        t = adapter.parse_transcript(window.raw_text, req)
        if t.text:
            texts.append(t.text)
        for seg in t.segments:
            segments.append({**seg, "start": seg["start"] + window.offset, "end": seg["end"] + window.offset})
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
    windows = plan_windows(audio_path, adapter.max_audio_seconds, api.upload_dir)

    if req.stream:
        return StreamingResponse(
            _stream(api, adapter, req, windows, mode, audio_path),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    if mode == "parallel" or len(windows) == 1:
        for window in windows:
            _submit(api, adapter, req, window, streaming=False)
        for window in windows:
            window.raw_text = _text_of(await api.collect_results(window.request_id, raw_request))
    else:
        language = req.language
        texts: list[str] = []
        for i, window in enumerate(windows):
            window_req = _window_request(req, language, _carry_over(texts) if i else None)
            _submit(api, adapter, window_req, window, streaming=False)
            window.raw_text = _text_of(await api.collect_results(window.request_id, raw_request))
            parsed = adapter.parse_transcript(window.raw_text, window_req)
            texts.append(parsed.text)
            language = language or parsed.language

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
                "temperature": req.temperature or 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
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


def _error_event(chunk) -> str:
    return sse({
        "type": "error",
        "error": {
            "message": chunk.data.decode("utf-8", "replace"),
            "type": "server_error",
            "code": (chunk.metadata or {}).get("status") or 500,
        },
    })


async def _stream(api, adapter, req, windows: list[Window], mode: str, audio_path: str):
    """Deltas from every window in order; sequential mode submits the next
    window only once the previous one finished (its text is the prompt)."""
    if mode == "parallel":
        for window in windows:
            _submit(api, adapter, req, window, streaming=True)
    language = req.language
    texts: list[str] = []
    for i, window in enumerate(windows):
        window_req = req
        if mode != "parallel":
            window_req = _window_request(req, language, _carry_over(texts) if i else None)
            _submit(api, adapter, window_req, window, streaming=True)
        raw_parts: list[str] = []
        async for c in api.iter_result_chunks(window.request_id):
            if c.modality == "error":
                yield _error_event(c)
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
        parsed = adapter.parse_transcript(window.raw_text, window_req)
        texts.append(parsed.text)
        language = language or parsed.language
        if i + 1 < len(windows):
            # windows are joined by a space in the final text; say so mid-stream
            yield sse({"type": "transcript.text.delta", "delta": " "})
        await asyncio.sleep(0)
    transcript = merge_windows(windows, adapter, req)
    yield sse({
        "type": "transcript.text.done",
        "text": transcript.text,
        "usage": _usage(audio_duration_seconds(audio_path)),
    })
    yield SSE_DONE
